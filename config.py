"""
Central configuration for the Securities Market Sentiment Analysis (SMSA) system.
All values can be overridden through environment variables.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path):
    """Minimal .env loader (KEY=VALUE lines) so secrets never live in source code."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---------------------------------------------------------------- server
HOST = os.environ.get("SMSA_HOST", "127.0.0.1")
# 5050 by default: on macOS port 5000 is normally used by the AirPlay Receiver.
PORT = int(os.environ.get("SMSA_PORT", "5050"))

# ---------------------------------------------------------------- database
DATABASE_PATH = os.environ.get(
    "SMSA_DB_PATH", os.path.join(BASE_DIR, "database", "smsa.db")
)
SCHEMA_PATH = os.path.join(BASE_DIR, "database", "schema.sql")

# ---------------------------------------------------------------- models
# Engine "groq": open-weight LLM served by the Groq API (needs GROQ_API_KEY in .env).
GROQ_MODEL_ID = os.environ.get("SMSA_GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_ENABLED = bool(os.environ.get("GROQ_API_KEY"))
# Engine "gemini": Google Gemini API (needs GEMINI_API_KEY in .env) - also the fallback for Groq.
GEMINI_MODEL_ID = os.environ.get("SMSA_GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_ENABLED = bool(os.environ.get("GEMINI_API_KEY"))
# Engine "llm": an instruction-tuned, pre-trained open LLM run locally, driven by prompt engineering.
LLM_MODEL_ID = os.environ.get("SMSA_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
# Engine "finbert": a Transformer pre-trained / fine-tuned on financial text.
FINBERT_MODEL_ID = os.environ.get("SMSA_FINBERT_MODEL", "ProsusAI/finbert")

ENGINES = ("groq", "gemini", "llm", "finbert")
# Groq is the default when its API key is configured, then Gemini, otherwise the local LLM.
DEFAULT_ENGINE = os.environ.get(
    "SMSA_DEFAULT_ENGINE", "groq" if GROQ_ENABLED else "gemini" if GEMINI_ENABLED else "llm")

# Automatic fallback: if a cloud LLM fails (outage, rate limit, invalid key, no internet),
# the request is answered by the next engine of the chain. The stored result always names
# the engine that actually answered and records the failures in fallback_from.
FALLBACK_ENABLED = os.environ.get("SMSA_FALLBACK", "1") == "1"
FALLBACK_CHAIN = {"groq": ["gemini", "llm"], "gemini": ["groq", "llm"]}
DEFAULT_PROMPT_VERSION = os.environ.get("SMSA_PROMPT_VERSION", "fewshot-v2")

MAX_NEW_TOKENS = int(os.environ.get("SMSA_MAX_NEW_TOKENS", "220"))
MAX_INPUT_CHARS = 4000     # guard-rail for the API
MIN_INPUT_CHARS = 10

# Load the models in a background thread when the server starts, so the
# first user request does not wait for model loading.
PRELOAD_MODELS = os.environ.get("SMSA_PRELOAD", "1") == "1"

# ---------------------------------------------------------------- domain
SENTIMENT_LABELS = ["Positive", "Negative", "Neutral"]
TEXT_TYPES = {
    "news": "Financial news",
    "announcement": "Company announcement",
    "analyst": "Analyst commentary",
    "investor": "Investor comment",
    "social": "Social-media market comment",
    "other": "Other",
}

TEST_DATASET_CSV = os.path.join(BASE_DIR, "data", "test_dataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
