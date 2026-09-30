"""
LLM integration layer of the SMSA system.

Four pre-trained Transformer engines are supported:

0. "groq"    - openai/gpt-oss-120b served by the Groq API (default when GROQ_API_KEY is set) and
   "gemini"  - Google Gemini API (automatic fallback when Groq fails). Both use the same
               engineered prompt with the output enforced by a JSON schema.
1. "llm"     - Qwen2.5-1.5B-Instruct run locally, a general instruction-tuned LLM that is
               steered with prompt engineering (role prompt, label definitions,
               domain rules, few-shot examples and a strict JSON output format).
2. "finbert" - ProsusAI/finbert, a BERT model further trained on financial text
               for 3-class sentiment classification (used as a domain baseline).

How the LLM produces a label + a calibrated confidence
------------------------------------------------------
The prompt asks the model to answer in JSON whose FIRST key is "sentiment".
We append the beginning of that answer ( {"sentiment": " ) to the prompt and
read the model's next-token probability distribution. The probabilities of the
tokens that start "Positive", "Negative" and "Neutral" are re-normalised to give
P(Positive), P(Negative), P(Neutral). The arg-max is the predicted sentiment,
its probability is the confidence and P(Pos) - P(Neg) is a score in [-1, 1].
Generation then continues (greedy decoding) to obtain the company name, the key
factors and a one-sentence reasoning. Because the label is taken from the
logits, the output is always one of the three valid labels, even if the rest of
the JSON produced by the small model is imperfect.
"""
import json
import os
import re
import threading
import time

import config

# ============================================================== PROMPTS
SYSTEM_PROMPT = """You are a senior equity research analyst who specialises in \
securities-market sentiment analysis. You read financial news, company announcements, \
analyst commentary, investor comments and social-media posts and decide what the text \
implies for the market sentiment towards the company (its share price / outlook) from \
the point of view of an investor in that company.

Sentiment labels (use exactly one):
- Positive : the text is favourable for the company - e.g. earnings beat, revenue or \
profit growth, new orders or contracts, upgrades, higher targets, dividends/buy-backs, \
approvals, expansion, bullish investor mood.
- Negative : the text is unfavourable - e.g. losses, earnings miss, falling sales, \
downgrades, lower targets, debt stress or default, fraud, penalties, lawsuits, \
resignations of key people, plant shutdowns, bearish investor mood.
- Neutral  : factual or procedural information with no clear favourable or unfavourable \
implication (meeting dates, routine filings, appointments of auditors, in-line results), \
or a balanced mix where good and bad points cancel out, or a question without an opinion.

Rules:
1. Judge the implication for the company that the text is about, not the tone of the words alone.
2. Compare results with expectations: "profit rose but missed estimates" is Negative; \
"loss narrowed more than expected" is Positive.
3. Understand market slang and sarcasm in social-media posts: "to the moon", "rocket", \
"bullish", "buying the dip" are Positive; "bagholder", "rug pull", "dumping", \
"stay away" are Negative; sarcastic praise of bad news is Negative.
4. If the text contains both good and bad news, choose the side that dominates; choose \
Neutral only if they are genuinely balanced.
5. Respond ONLY with a JSON object, no other text, in this exact format:
{"sentiment": "Positive|Negative|Neutral", "company": "<company name or Unknown>", \
"key_factors": ["<factor 1>", "<factor 2>"], "reasoning": "<one short sentence>"}"""

ZERO_SHOT_SYSTEM_PROMPT = (
    "You are a financial sentiment classifier. Classify the market sentiment of the "
    "text towards the company as Positive, Negative or Neutral. Respond ONLY with JSON: "
    '{"sentiment": "Positive|Negative|Neutral", "company": "<company name or Unknown>", '
    '"key_factors": ["..."], "reasoning": "<one short sentence>"}'
)

# Few-shot demonstrations. They are deliberately NOT taken from the test data set
# (to avoid leaking test answers into the prompt) and cover every text type,
# every label, sarcasm and a mixed / in-line case.
FEW_SHOT_EXAMPLES = [
    (
        "news",
        "Shares of Coral Textiles Ltd climbed 7% after the company won a Rs 1,200 crore "
        "export order from a large European retailer.",
        {"sentiment": "Positive", "company": "Coral Textiles Ltd",
         "key_factors": ["Rs 1,200 crore export order", "share price up 7%"],
         "reasoning": "A large new order boosts future revenue and the stock reacted positively."},
    ),
    (
        "announcement",
        "Harbor Logistics Ltd hereby informs the exchange that a meeting of the Board of "
        "Directors will be held on 14 November to consider the unaudited results for Q2.",
        {"sentiment": "Neutral", "company": "Harbor Logistics Ltd",
         "key_factors": ["routine board meeting intimation"],
         "reasoning": "A procedural disclosure with no information on performance."},
    ),
    (
        "analyst",
        "We downgrade Maple Cement to SELL. Weak demand and rising fuel costs are likely "
        "to compress EBITDA margins over the next two quarters.",
        {"sentiment": "Negative", "company": "Maple Cement",
         "key_factors": ["downgrade to SELL", "weak demand", "margin compression"],
         "reasoning": "The analyst expects falling margins and recommends selling."},
    ),
    (
        "social",
        "Great, another 'strategic restructuring' at Nimbus Telecom. Bagholders keep "
        "bleeding lol #NIMB",
        {"sentiment": "Negative", "company": "Nimbus Telecom",
         "key_factors": ["sarcasm about restructuring", "bagholders losing money"],
         "reasoning": "Sarcastic post expressing frustration over continued losses."},
    ),
    (
        "investor",
        "Been holding Orchid Foods for three years; the new CEO's cost discipline is "
        "finally showing in the margins. Adding more on every dip.",
        {"sentiment": "Positive", "company": "Orchid Foods",
         "key_factors": ["improving margins", "investor adding to position"],
         "reasoning": "The investor sees improving fundamentals and is buying more."},
    ),
    (
        "news",
        "Vega Motors reported a 4% rise in revenue while operating margin slipped 60 basis "
        "points; overall results were in line with street estimates.",
        {"sentiment": "Neutral", "company": "Vega Motors",
         "key_factors": ["revenue +4%", "margin -60 bps", "in line with estimates"],
         "reasoning": "Small positives and negatives balance out and results met expectations."},
    ),
]

# ---- Cloud LLM APIs (Groq, Gemini): same role, label definitions, rules and demonstrations,
# but the output format is enforced by the API's structured outputs (JSON schema).
API_SYSTEM_PROMPT = SYSTEM_PROMPT.split("5. Respond ONLY")[0] + (
    "5. Also estimate the probability of each label as a whole-number percentage "
    "(0-100, the three numbers must sum to 100); the chosen sentiment must have the "
    "highest probability.\n\n"
    "Worked examples:\n" + "\n".join(
        f'- [{config.TEXT_TYPES[t]}] "{txt}" -> {a["sentiment"]} ({a["reasoning"]})'
        for t, txt, a in FEW_SHOT_EXAMPLES
    )
)

API_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["Positive", "Negative", "Neutral"]},
        "company": {"type": "string"},
        "probabilities": {
            "type": "object",
            "properties": {k: {"type": "integer"} for k in ("Positive", "Negative", "Neutral")},
            "required": ["Positive", "Negative", "Neutral"],
            "additionalProperties": False,
        },
        "key_factors": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["sentiment", "company", "probabilities", "key_factors", "reasoning"],
    "additionalProperties": False,
}

PROMPT_VERSIONS = {
    "fewshot-v2": "Role prompt + label definitions + domain rules + 6 few-shot examples + JSON output",
    "zeroshot-v1": "Short zero-shot instruction + JSON output",
}


def _user_message(text, text_type=None, company=None):
    parts = []
    if text_type and text_type in config.TEXT_TYPES and text_type != "other":
        parts.append(f"Text type: {config.TEXT_TYPES[text_type]}")
    if company:
        parts.append(f"Company: {company}")
    parts.append(f'Text: """{text.strip()}"""')
    parts.append("Return the JSON answer.")
    return "\n".join(parts)


def build_messages(text, text_type=None, company=None, prompt_version="fewshot-v2"):
    """Builds the chat messages sent to the LLM for a given prompt version."""
    if prompt_version == "zeroshot-v1":
        return [
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(text)},
        ]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_type, ex_text, ex_answer in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": _user_message(ex_text, ex_type)})
        messages.append({"role": "assistant", "content": json.dumps(ex_answer)})
    messages.append({"role": "user", "content": _user_message(text, text_type, company)})
    return messages


# ============================================================== ERRORS
def api_error_message(exc):
    """
    Extracts the human-readable message from an SDK exception (Groq, Gemini).
    SDK exceptions often stringify to a long Python dict; users should see one sentence.
    """
    body = getattr(exc, "body", None) or getattr(exc, "details", None)
    msg = None
    if isinstance(body, dict):
        err = body.get("error", body)
        msg = err.get("message") if isinstance(err, dict) else str(err)
    if not msg:
        msg = getattr(exc, "message", None) or str(exc)
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= 160 else msg[:157] + "..."


# ============================================================== ENGINE
class ModelNotReady(RuntimeError):
    pass


class SentimentEngine:
    """Loads the models once and serves thread-safe sentiment predictions."""

    ANSWER_PREFIX = '{"sentiment": "'

    def __init__(self):
        self._lock = threading.Lock()          # one inference at a time
        self._load_lock = threading.Lock()
        self.device = None
        self.llm = None
        self.tokenizer = None
        self.finbert = None
        self.status = {"llm": "not loaded", "finbert": "not loaded"}
        # Cloud engines: "unchecked" | "ready" | "no API key" | "unavailable: <reason>"
        self.cloud = {
            "groq": "unchecked" if config.GROQ_ENABLED else "no API key",
            "gemini": "unchecked" if config.GEMINI_ENABLED else "no API key",
        }
        self.errors = {}

    # ------------------------------------------------------------ loading
    def _pick_device(self):
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_llm(self):
        with self._load_lock:
            if self.llm is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.status["llm"] = "loading"
            try:
                self.device = self.device or self._pick_device()
                dtype = torch.float32 if self.device == "cpu" else torch.float16
                self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
                model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL_ID, dtype=dtype)
                self.llm = model.to(self.device).eval()
                self._label_token_ids = self._compute_label_token_ids()
                self.status["llm"] = "ready"
            except Exception as exc:                       # pragma: no cover
                self.status["llm"] = "error"
                self.errors["llm"] = str(exc)
                raise

    def load_finbert(self):
        with self._load_lock:
            if self.finbert is not None:
                return
            from transformers import pipeline
            self.status["finbert"] = "loading"
            try:
                self.device = self.device or self._pick_device()
                self.finbert = pipeline(
                    "text-classification", model=config.FINBERT_MODEL_ID,
                    tokenizer=config.FINBERT_MODEL_ID, device=self.device, top_k=None,
                )
                self.status["finbert"] = "ready"
            except Exception as exc:                       # pragma: no cover
                self.status["finbert"] = "error"
                self.errors["finbert"] = str(exc)
                raise

    def check_cloud_engines(self):
        """
        Verifies every configured API key once, with a free "list models" call that uses no
        generation quota. An engine that fails is marked unavailable: the frontend disables it
        and the fallback chain skips it without making a request.
        """
        checks = {"groq": self._check_groq, "gemini": self._check_gemini}
        for name, check in checks.items():
            if self.cloud[name] == "no API key":
                continue
            try:
                check()
                self.cloud[name] = "ready"
            except Exception as exc:
                self.cloud[name] = f"unavailable: {api_error_message(exc)}"
        return dict(self.cloud)

    def _check_groq(self):
        import groq
        groq.Groq(timeout=15.0, max_retries=0).models.list()

    def _check_gemini(self):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                              http_options=types.HttpOptions(timeout=15_000))
        next(iter(client.models.list(config={"page_size": 1})))

    def load_all(self):
        for loader in (self.load_llm, self.load_finbert):
            try:
                loader()
            except Exception:
                pass

    def _compute_label_token_ids(self):
        """
        Finds the first token of every label *in context* (right after the answer
        prefix), so that BPE merges at the boundary are handled correctly.
        """
        probe = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True
        ) + self.ANSWER_PREFIX
        base = self.tokenizer(probe, add_special_tokens=False)["input_ids"]
        ids = {}
        for label in config.SENTIMENT_LABELS:
            full = self.tokenizer(probe + label, add_special_tokens=False)["input_ids"]
            if full[: len(base)] != base:
                raise RuntimeError("Tokenizer merges the answer prefix with the label")
            ids[label] = full[len(base)]
        if len(set(ids.values())) != len(ids):
            raise RuntimeError(f"Labels share the same first token: {ids}")
        return ids

    # ------------------------------------------------------------ inference
    def available_engines(self):
        return [
            {"engine": "groq", "model_name": config.GROQ_MODEL_ID,
             "status": self.cloud["groq"],
             "prompt_versions": {"fewshot-v2": PROMPT_VERSIONS["fewshot-v2"] + " (JSON schema enforced)"},
             "description": "Open-weight LLM served by the Groq API with structured outputs"},
            {"engine": "gemini", "model_name": config.GEMINI_MODEL_ID,
             "status": self.cloud["gemini"],
             "prompt_versions": {"fewshot-v2": PROMPT_VERSIONS["fewshot-v2"] + " (JSON schema enforced)"},
             "description": "Google Gemini API with structured outputs; fallback for Groq"},
            {"engine": "llm", "model_name": config.LLM_MODEL_ID, "status": self.status["llm"],
             "prompt_versions": PROMPT_VERSIONS,
             "description": "Instruction-tuned LLM with prompt engineering (explains its answer)"},
            {"engine": "finbert", "model_name": config.FINBERT_MODEL_ID,
             "status": self.status["finbert"], "prompt_versions": {"n/a": "Classifier - no prompt"},
             "description": "BERT pre-trained on financial text (classifier baseline)"},
        ]

    def _engine_problem(self, name):
        """Returns why a cloud engine cannot be used, or None if it may be tried."""
        state = self.cloud.get(name)
        if state is None or state in ("ready", "unchecked"):
            return None
        return state.replace("unavailable: ", "")

    def analyze(self, text, text_type=None, company=None, engine=None, prompt_version=None,
                fallback=None):
        """
        Analyses one text. For the cloud engines, if the call fails and fallback is enabled,
        the next engine of config.FALLBACK_CHAIN answers instead; result["fallback_from"]
        lists the engines that failed and why.
        """
        engine = engine or config.DEFAULT_ENGINE
        if fallback is None:
            fallback = config.FALLBACK_ENABLED
        chain = [engine] + (config.FALLBACK_CHAIN.get(engine, []) if fallback else [])
        chain = [e for i, e in enumerate(chain) if e not in chain[:i]]
        failures = []
        for i, name in enumerate(chain):
            problem = self._engine_problem(name)
            if problem:                                  # known to be unusable - do not call it
                failures.append(f"{name}: {problem}")
                continue
            try:
                result = self._analyze_one(text, text_type, company, name, prompt_version)
            except ValueError:
                raise                                   # invalid request - never fall back
            except Exception as exc:
                failures.append(f"{name}: {exc}"[:300])
                continue
            result["fallback_from"] = failures
            return result
        raise RuntimeError("; ".join(failures) or "No engine available")

    def _analyze_one(self, text, text_type, company, engine, prompt_version):
        if engine == "gemini":
            return self._analyze_gemini(text, text_type, company)
        if engine == "groq":
            return self._analyze_groq(text, text_type, company)
        if engine == "finbert":
            return self._analyze_finbert(text)
        if engine == "llm":
            return self._analyze_llm(text, text_type, company,
                                     prompt_version or config.DEFAULT_PROMPT_VERSION)
        raise ValueError(f"Unknown engine '{engine}'")

    def _analyze_llm(self, text, text_type, company, prompt_version):
        import torch
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"Unknown prompt_version '{prompt_version}'")
        if self.llm is None:
            self.load_llm()
        start = time.perf_counter()
        messages = build_messages(text, text_type, company, prompt_version)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + self.ANSWER_PREFIX

        with self._lock, torch.inference_mode():
            enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
            # Step 1 - one forward pass: probability of each label as the next token.
            out = self.llm(**enc, use_cache=True)
            next_probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
            raw = {lab: next_probs[tid].item() for lab, tid in self._label_token_ids.items()}
            mass = sum(raw.values()) or 1e-9
            probs = {lab: v / mass for lab, v in raw.items()}
            label = max(probs, key=probs.get)

            # Step 2 - force the chosen label and let the model write the explanation.
            cont_prompt = prompt + label + '"'
            enc2 = self.tokenizer(cont_prompt, return_tensors="pt",
                                  add_special_tokens=False).to(self.device)
            gen = self.llm.generate(
                **enc2, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated = self.tokenizer.decode(gen[0, enc2["input_ids"].shape[1]:],
                                              skip_special_tokens=True)

        raw_response = self.ANSWER_PREFIX + label + '"' + generated
        parsed = parse_llm_json(raw_response)
        detected = (parsed.get("company") or "").strip()
        if detected.lower() in ("", "unknown", "n/a", "none"):
            detected = None
        return {
            "engine": "llm",
            "model_name": config.LLM_MODEL_ID,
            "prompt_version": prompt_version,
            "model_description": PROMPT_VERSIONS[prompt_version],
            "sentiment": label,
            "confidence": round(probs[label], 4),
            "score": round(probs["Positive"] - probs["Negative"], 4),
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
            "label_probability_mass": round(mass, 4),
            "company": detected or company,
            "key_factors": [str(f) for f in (parsed.get("key_factors") or [])][:6],
            "reasoning": parsed.get("reasoning"),
            "raw_response": raw_response.strip(),
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }

    def _analyze_gemini(self, text, text_type, company):
        from google import genai
        from google.genai import errors, types
        if not config.GEMINI_ENABLED:
            raise RuntimeError("Gemini engine needs GEMINI_API_KEY in the .env file")
        if getattr(self, "_gemini", None) is None:
            self._gemini = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                http_options=types.HttpOptions(timeout=60_000))      # milliseconds
        start = time.perf_counter()
        try:
            response = self._gemini.models.generate_content(
                model=config.GEMINI_MODEL_ID,
                contents=_user_message(text, text_type, company),
                config=types.GenerateContentConfig(
                    system_instruction=API_SYSTEM_PROMPT,
                    temperature=0,
                    response_mime_type="application/json",
                    response_json_schema=API_OUTPUT_SCHEMA,
                ),
            )
        except errors.ClientError as exc:           # 4xx: invalid key, quota, bad request
            raise RuntimeError(f"Gemini API error {exc.code}: {api_error_message(exc)}")
        except errors.ServerError as exc:           # 5xx: overloaded / unavailable
            raise RuntimeError(f"Gemini API unavailable ({exc.code}): {api_error_message(exc)}")
        except Exception as exc:                    # network problems
            raise RuntimeError(f"Cannot reach the Gemini API: {exc}")
        if not response.candidates or not response.text:
            raise RuntimeError("Gemini returned no answer (possibly blocked)")
        usage = response.usage_metadata
        return _result_from_api_json(
            response.text, "gemini", response.model_version or config.GEMINI_MODEL_ID, company,
            start, {"input_tokens": getattr(usage, "prompt_token_count", None),
                    "output_tokens": getattr(usage, "candidates_token_count", None)})

    def _analyze_groq(self, text, text_type, company):
        import groq
        if not config.GROQ_ENABLED:
            raise RuntimeError("Groq engine needs GROQ_API_KEY in the .env file")
        if getattr(self, "_groq", None) is None:
            self._groq = groq.Groq(timeout=60.0, max_retries=2)   # key read from GROQ_API_KEY
        start = time.perf_counter()
        # Groq validates the generated JSON against the schema; a rare malformed token
        # (e.g. "0. nine") is rejected with "json_validate_failed", so retry a few times.
        for attempt in range(3):
            try:
                response = self._groq.chat.completions.create(
                    model=config.GROQ_MODEL_ID,
                    messages=[
                        {"role": "system", "content": API_SYSTEM_PROMPT},
                        {"role": "user", "content": _user_message(text, text_type, company)},
                    ],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "market_sentiment", "strict": True, "schema": API_OUTPUT_SCHEMA}},
                    reasoning_effort="low",     # short classification task
                    temperature=0 if attempt == 0 else 0.3,
                    max_completion_tokens=2000,
                )
                break
            except groq.BadRequestError as exc:
                if "json_validate_failed" not in str(exc) or attempt == 2:
                    raise RuntimeError(f"Groq API error 400: {api_error_message(exc)}")
            except groq.AuthenticationError:
                raise RuntimeError("Groq API key is invalid")
            except groq.RateLimitError:
                raise RuntimeError("Groq API rate limit reached - try again shortly")
            except groq.APIStatusError as exc:
                raise RuntimeError(f"Groq API error {exc.status_code}: {api_error_message(exc)}")
            except groq.APIConnectionError:
                raise RuntimeError("Cannot reach the Groq API - check the internet connection")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise RuntimeError(f"Groq response incomplete (finish_reason={choice.finish_reason})")
        return _result_from_api_json(
            choice.message.content, "groq", response.model or config.GROQ_MODEL_ID, company, start,
            {"input_tokens": response.usage.prompt_tokens,
             "output_tokens": response.usage.completion_tokens})

    def _analyze_finbert(self, text):
        if self.finbert is None:
            self.load_finbert()
        start = time.perf_counter()
        with self._lock:
            scores = self.finbert(text[:2000], truncation=True)[0]
        probs = {s["label"].capitalize(): float(s["score"]) for s in scores}
        label = max(probs, key=probs.get)
        return {
            "engine": "finbert",
            "model_name": config.FINBERT_MODEL_ID,
            "prompt_version": "n/a",
            "model_description": "FinBERT 3-class financial sentiment classifier",
            "sentiment": label,
            "confidence": round(probs[label], 4),
            "score": round(probs["Positive"] - probs["Negative"], 4),
            "probabilities": {k: round(probs[k], 4) for k in config.SENTIMENT_LABELS},
            "company": None,
            "key_factors": [],
            "reasoning": "FinBERT is a classifier; it returns class probabilities only.",
            "raw_response": json.dumps(scores),
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }


def _result_from_api_json(raw, engine_name, model_name, company, start, usage):
    """Converts the schema-constrained JSON answer of a cloud LLM into the common result format."""
    try:
        data = json.loads(raw)
        label = data["sentiment"]
        if label not in config.SENTIMENT_LABELS:
            raise KeyError(label)
    except (ValueError, KeyError, TypeError) as exc:
        # A model failure, not a bad request - raised as RuntimeError so the fallback chain applies.
        raise RuntimeError(f"{engine_name} returned an invalid answer: {exc}")
    p = {k: max(0.0, float(data["probabilities"].get(k, 0))) for k in config.SENTIMENT_LABELS}
    total = sum(p.values()) or 1.0
    probs = {k: v / total for k, v in p.items()}
    detected = (data.get("company") or "").strip()
    if detected.lower() in ("", "unknown", "n/a", "none"):
        detected = None
    return {
        "engine": engine_name,
        "model_name": model_name,
        "prompt_version": "fewshot-v2",
        "model_description": PROMPT_VERSIONS["fewshot-v2"] + " (JSON schema enforced)",
        "sentiment": label,
        "confidence": round(probs[label], 4),
        "score": round(probs["Positive"] - probs["Negative"], 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "company": detected or company,
        "key_factors": [str(f) for f in data.get("key_factors", [])][:6],
        "reasoning": data.get("reasoning"),
        "raw_response": raw,
        "latency_ms": int((time.perf_counter() - start) * 1000),
        "usage": usage,
    }


# ============================================================== PARSING
def parse_llm_json(raw):
    """
    Robustly extracts the JSON object produced by the LLM. Small models sometimes
    add text after the object or truncate it, so we fall back to regexes.
    """
    raw = raw.strip()
    end = raw.find("}")
    while end != -1:
        try:
            return json.loads(raw[: end + 1])
        except ValueError:
            end = raw.find("}", end + 1)
    result = {}
    m = re.search(r'"company"\s*:\s*"([^"]*)"', raw)
    if m:
        result["company"] = m.group(1)
    m = re.search(r'"reasoning"\s*:\s*"([^"]*)', raw)
    if m:
        result["reasoning"] = m.group(1)
    m = re.search(r'"key_factors"\s*:\s*\[([^\]]*)', raw)
    if m:
        result["key_factors"] = re.findall(r'"([^"]+)"', m.group(1))
    return result


engine = SentimentEngine()
