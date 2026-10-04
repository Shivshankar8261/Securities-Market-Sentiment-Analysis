"""
Securities Market Sentiment Analysis (SMSA) - Flask backend.

Run:  python app.py        ->  http://127.0.0.1:5050

Pages                       REST API (JSON)
-----                       ---------------
/            Analyse text   POST   /api/analyze              analyse a text, store result
/history     Stored results POST   /api/analyze/batch        analyse up to 20 texts
/dashboard   Statistics     GET    /api/results              list stored results (filters)
/test-results Test runs     GET    /api/results/<id>         one stored result
                            DELETE /api/results/<id>         delete a stored result
                            GET    /api/results/export.csv   download all results
                            GET    /api/stats                aggregate statistics
                            GET    /api/companies            companies in the database
                            GET    /api/models               available models / prompts
                            GET    /api/testset              labelled test data set
                            GET    /api/test-runs            evaluation runs
                            GET    /api/test-runs/<id>       one run with predictions
                            GET    /api/health               service health
"""
import csv
import io
import threading

from flask import Flask, Response, jsonify, render_template, request

import config
import database as db
from sentiment_engine import engine

app = Flask(__name__)
app.json.sort_keys = False
db.init_db()


# ================================================================== helpers
class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


@app.errorhandler(ApiError)
def handle_api_error(err):
    return jsonify({"status": "error", "error": err.message}), err.status


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"status": "error", "error": "Resource not found"}), 404
    return render_template("base.html", missing=True), 404


@app.errorhandler(500)
def server_error(err):
    return jsonify({"status": "error", "error": f"Internal server error: {err}"}), 500


def validate_payload(payload):
    """Validates one analysis request and returns the cleaned fields."""
    if not isinstance(payload, dict):
        raise ApiError("Request body must be a JSON object")
    text = (payload.get("text") or "").strip()
    if len(text) < config.MIN_INPUT_CHARS:
        raise ApiError(f"'text' is required and must have at least {config.MIN_INPUT_CHARS} characters")
    if len(text) > config.MAX_INPUT_CHARS:
        raise ApiError(f"'text' must not exceed {config.MAX_INPUT_CHARS} characters")
    text_type = (payload.get("text_type") or "other").strip().lower()
    if text_type not in config.TEXT_TYPES:
        raise ApiError(f"'text_type' must be one of {list(config.TEXT_TYPES)}")
    eng = (payload.get("engine") or config.DEFAULT_ENGINE).strip().lower()
    if eng not in config.ENGINES:
        raise ApiError(f"'engine' must be one of {list(config.ENGINES)}")
    prompt_version = payload.get("prompt_version") or config.DEFAULT_PROMPT_VERSION
    company = (payload.get("company") or "").strip()[:120] or None
    source = payload.get("source") if payload.get("source") in ("web", "api", "test") else "api"
    fallback = payload.get("fallback", config.FALLBACK_ENABLED)
    if not isinstance(fallback, bool):
        raise ApiError("'fallback' must be true or false")
    return text, text_type, eng, prompt_version, company, source, fallback


def run_analysis(payload):
    text, text_type, eng, prompt_version, company, source, fallback = validate_payload(payload)
    try:
        result = engine.analyze(text, text_type=text_type, company=company, engine=eng,
                                prompt_version=prompt_version, fallback=fallback)
    except ValueError as exc:
        raise ApiError(str(exc))
    except Exception as exc:                                   # model failure
        raise ApiError(f"LLM inference failed: {exc}", 503)
    record = db.save_analysis(text, text_type, source, company, result,
                              client_ip=request.remote_addr)
    record["probabilities"] = result["probabilities"]
    record["requested_engine"] = eng
    return record


# ================================================================== pages
@app.route("/")
def index():
    return render_template("index.html", text_types=config.TEXT_TYPES,
                           default_engine=config.DEFAULT_ENGINE,
                           groq_model=config.GROQ_MODEL_ID,
                           groq_enabled=config.GROQ_ENABLED,
                           gemini_model=config.GEMINI_MODEL_ID,
                           gemini_enabled=config.GEMINI_ENABLED)


@app.route("/history")
def history():
    return render_template("history.html", text_types=config.TEXT_TYPES)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/test-results")
def test_results():
    return render_template("test_results.html")


# ================================================================== API
@app.post("/api/analyze")
def api_analyze():
    record = run_analysis(request.get_json(silent=True))
    return jsonify({"status": "success", "result": record}), 201


@app.post("/api/analyze/batch")
def api_analyze_batch():
    body = request.get_json(silent=True) or {}
    items = body.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 20:
        raise ApiError("'items' must be a list of 1 to 20 objects")
    results = []
    for i, item in enumerate(items):
        try:
            results.append({"index": i, "status": "success", "result": run_analysis(item)})
        except ApiError as err:
            results.append({"index": i, "status": "error", "error": err.message})
    return jsonify({"status": "success", "count": len(results), "results": results}), 201


@app.get("/api/results")
def api_results():
    try:
        limit = max(1, min(int(request.args.get("limit", 50)), 500))
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        raise ApiError("'limit' and 'offset' must be integers")
    total, rows = db.list_results(
        limit=limit, offset=offset,
        sentiment=request.args.get("sentiment") or None,
        company=request.args.get("company") or None,
        text_type=request.args.get("text_type") or None,
        source=request.args.get("source") or None,
        search=request.args.get("q") or None,
    )
    return jsonify({"status": "success", "total": total, "limit": limit,
                    "offset": offset, "results": rows})


@app.get("/api/results/<int:result_id>")
def api_result(result_id):
    row = db.get_result(result_id)
    if not row:
        raise ApiError("Result not found", 404)
    return jsonify({"status": "success", "result": row})


@app.delete("/api/results/<int:result_id>")
def api_delete_result(result_id):
    if not db.delete_result(result_id):
        raise ApiError("Result not found", 404)
    return jsonify({"status": "success", "deleted": result_id})


@app.get("/api/results/export.csv")
def api_export():
    _, rows = db.list_results(limit=100000)
    buf = io.StringIO()
    fields = ["result_id", "submitted_at", "source", "text_type", "company_name",
              "detected_company", "input_text", "sentiment", "confidence",
              "sentiment_score", "reasoning", "engine", "model_name", "prompt_version",
              "latency_ms", "fallback_from"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=smsa_results.csv"})


@app.get("/api/stats")
def api_stats():
    return jsonify({"status": "success", "stats": db.get_stats()})


@app.get("/api/companies")
def api_companies():
    return jsonify({"status": "success", "companies": db.list_companies()})


@app.get("/api/models")
def api_models():
    return jsonify({"status": "success", "default_engine": config.DEFAULT_ENGINE,
                    "default_prompt_version": config.DEFAULT_PROMPT_VERSION,
                    "models": engine.available_engines()})


@app.get("/api/testset")
def api_testset():
    rows = db.get_test_dataset(request.args.get("text_type") or None)
    return jsonify({"status": "success", "count": len(rows), "items": rows})


@app.get("/api/test-runs")
def api_test_runs():
    return jsonify({"status": "success", "runs": db.list_test_runs()})


@app.get("/api/test-runs/<int:run_id>")
def api_test_run(run_id):
    data = db.get_test_run(run_id)
    if not data:
        raise ApiError("Test run not found", 404)
    return jsonify({"status": "success", **data})


@app.get("/api/health")
def api_health():
    # On serverless hosting (Vercel) the start-up thread never runs, so verify the API keys
    # on the first health request instead (free "list models" calls, once per instance).
    if "unchecked" in engine.cloud.values():
        engine.check_cloud_engines()
    return jsonify({"status": "ok", "models": engine.status, "errors": engine.errors,
                    "groq": engine.cloud["groq"],
                    "gemini": engine.cloud["gemini"],
                    "fallback_chain": config.FALLBACK_CHAIN if config.FALLBACK_ENABLED else {},
                    "database": config.DATABASE_PATH})


if __name__ == "__main__":
    if config.PRELOAD_MODELS:
        threading.Thread(target=engine.load_all, daemon=True).start()
    # Verify the cloud API keys in the background (free "list models" calls).
    threading.Thread(target=engine.check_cloud_engines, daemon=True).start()
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
