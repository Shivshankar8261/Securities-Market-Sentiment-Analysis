# Securities Market Sentiment Analysis (SMSA)
Applied Deep Learning – Assignment 2 – Group 3

A Flask web application that accepts financial text (news, company announcements,
analyst commentary, investor comments, social-media posts), sends it to a
pre-trained LLM with an engineered prompt, returns the market sentiment
(Positive / Negative / Neutral) towards the company and stores every response in
an SQLite database.

## Folder structure
```
Assignment-2-Group-3/
├── app.py                  Flask backend – pages + REST API
├── sentiment_engine.py     LLM integration + prompt engineering (Groq / Gemini / Qwen / FinBERT)
├── database.py             Data-access layer (SQLite)
├── config.py               Configuration (models, paths, defaults)
├── run_tests.py            Runs the labelled test data set through POST /api/analyze
├── requirements.txt
├── .env.example            Template for the API keys (copy to .env)
├── templates/              Front end (HTML – Jinja2)
├── static/                 Front end (CSS + JavaScript)
├── data/
│   ├── build_test_dataset.py
│   └── test_dataset.csv    60 labelled financial texts (5 types × 12)
├── database/
│   ├── schema.sql          Database structure (DDL)
│   └── smsa.db             SQLite database with all stored results and test runs
├── results/                Predictions, summaries and confusion matrices of each test run
└── report/                 Assignment report (.docx / .pdf)
```

## Setup and run
```bash
pip install -r requirements.txt
python app.py                 # open http://127.0.0.1:5050
```
On first start the models are downloaded from Hugging Face
(Qwen2.5-1.5B-Instruct ≈ 3 GB, FinBERT ≈ 0.4 GB) and afterwards work offline.
The server listens on port 5050 (port 5000 is used by AirPlay Receiver on macOS); change it with `SMSA_PORT`.
Apple-silicon GPUs (MPS) and NVIDIA GPUs (CUDA) are used automatically; the CPU works too (slower).

### LLM engines and API keys
| Engine | Model | Needs |
|---|---|---|
| `groq` (default) | openai/gpt-oss-120b via the Groq API | `GROQ_API_KEY` in `.env`, internet |
| `gemini` (fallback) | gemini-3.5-flash via the Google Gemini API | `GEMINI_API_KEY` in `.env` |
| `llm` | Qwen2.5-1.5B-Instruct, local | nothing (works offline) |
| `finbert` | ProsusAI/finbert, local (baseline) | nothing |

Copy `.env.example` to `.env` and fill in the keys. API keys are never stored in the source code.
**Automatic fallback:** if Groq fails (outage, rate limit, invalid key, no internet) the request is
answered by Gemini, and if Gemini also fails, by the local Qwen model. The stored result names the model
that actually answered (`fallback_from` records the failures). Disable with `SMSA_FALLBACK=0`, or per
request with `"fallback": false`. Without any cloud key the app uses the local Qwen model. For an offline demo set
`SMSA_DEFAULT_ENGINE=llm` in `.env`. The model can also be chosen per request on the Analyse page.

## Testing with the test data set
```bash
python run_tests.py --engine groq                      # default configuration
python run_tests.py --engine llm --prompt fewshot-v2   # local LLM, engineered prompt
python run_tests.py --engine llm --prompt zeroshot-v1  # prompt-engineering baseline
python run_tests.py --engine finbert                   # domain classifier baseline
```
Results appear on the **Test Results** page and in `results/`.

## REST API (summary)
| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/analyze` | Analyse one text, store and return the result |
| POST | `/api/analyze/batch` | Analyse up to 20 texts |
| GET | `/api/results` | List stored results (filters: sentiment, company, text_type, source, q, limit, offset) |
| GET / DELETE | `/api/results/<id>` | Read / delete one stored result |
| GET | `/api/results/export.csv` | Download all results |
| GET | `/api/stats` | Aggregated statistics |
| GET | `/api/companies` | Companies in the database |
| GET | `/api/models` | Available models and prompt versions |
| GET | `/api/testset` | Labelled test data set |
| GET | `/api/test-runs`, `/api/test-runs/<id>` | Evaluation runs and their predictions |
| GET | `/api/health` | Service / model status |

Example:
```bash
curl -X POST http://127.0.0.1:5050/api/analyze -H "Content-Type: application/json" \
  -d '{"text": "The M/s XYZ Limited reported strong quarterly earnings, with revenue increasing significantly. Analysts expressed optimism about future growth.", "text_type": "news"}'
```
