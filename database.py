"""
Data-access layer for the SMSA system (SQLite).

Every function opens its own short-lived connection, which keeps the module
safe to use from Flask's multi-threaded development server.
"""
import csv
import json
import os
import shutil
import sqlite3
from contextlib import contextmanager

import config


@contextmanager
def get_connection(db_path=None):
    conn = sqlite3.connect(db_path or config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path=None):
    """Create all tables / indexes / views (idempotent) and load the test data set."""
    path = db_path or config.DATABASE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Serverless: start the writable copy from the database bundled with the code.
    if (not os.path.exists(path) and path != config.BUNDLED_DATABASE_PATH
            and os.path.exists(config.BUNDLED_DATABASE_PATH)):
        shutil.copyfile(config.BUNDLED_DATABASE_PATH, path)
    with open(config.SCHEMA_PATH, encoding="utf-8") as fh:
        schema = fh.read()
    _migrate(path)
    with get_connection(path) as conn:
        conn.executescript(schema)
    if os.path.exists(config.TEST_DATASET_CSV):
        load_test_dataset(config.TEST_DATASET_CSV, path)


def _migrate(path):
    """
    Upgrades a database created by an earlier version of the schema, keeping all data:
    - llm_models.engine CHECK constraint gains new engines (SQLite needs a table rebuild);
    - sentiment_results gains the fallback_from column.
    """
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='llm_models'").fetchone()
        if row is None:                      # new database - the schema creates everything
            return
        conn.execute("DROP VIEW IF EXISTS v_sentiment_history")   # recreated by schema.sql
        if "CHECK (engine IN ('groq','gemini','llm','finbert'))" not in " ".join(row[0].split()):
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.executescript("""
                BEGIN;
                CREATE TABLE llm_models_new (
                    model_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    engine          TEXT    NOT NULL
                                    CHECK (engine IN ('groq','gemini','llm','finbert')),
                    model_name      TEXT    NOT NULL,
                    prompt_version  TEXT    NOT NULL DEFAULT 'n/a',
                    description     TEXT,
                    created_at      TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                    UNIQUE (engine, model_name, prompt_version)
                );
                INSERT INTO llm_models_new SELECT * FROM llm_models;
                DROP TABLE llm_models;
                ALTER TABLE llm_models_new RENAME TO llm_models;
                COMMIT;""")
            problems = conn.execute("PRAGMA foreign_key_check").fetchall()
            conn.execute("PRAGMA foreign_keys = ON")
            if problems:
                raise RuntimeError(f"Migration left broken foreign keys: {problems[:5]}")
        cols = [c[1] for c in conn.execute("PRAGMA table_info(sentiment_results)")]
        if cols and "fallback_from" not in cols:
            conn.execute("ALTER TABLE sentiment_results ADD COLUMN fallback_from TEXT")
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------------ helpers
def _row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    if d.get("key_factors"):
        try:
            d["key_factors"] = json.loads(d["key_factors"])
        except (TypeError, ValueError):
            pass
    return d


_COMPANY_PREFIXES = ("m/s ", "m/s. ", "the ")
_COMPANY_SUFFIXES = (" limited", " ltd.", " ltd", " inc.", " inc", " plc", " corporation", " corp.", " corp")


def company_key(name):
    """Normalised key so that 'Aurora Steel', 'Aurora Steel Ltd' and 'M/s Aurora Steel Limited' match."""
    key = " ".join((name or "").lower().replace(",", " ").split())
    for pre in _COMPANY_PREFIXES:
        if key.startswith(pre):
            key = key[len(pre):]
    for suf in _COMPANY_SUFFIXES:
        if key.endswith(suf):
            key = key[: -len(suf)]
            break
    return key.strip()


def get_or_create_company(conn, company_name):
    name = (company_name or "").strip()
    if not name:
        return None
    key = company_key(name)
    for row in conn.execute("SELECT company_id, company_name FROM companies"):
        if company_key(row["company_name"]) == key:
            return row["company_id"]
    cur = conn.execute("INSERT INTO companies (company_name) VALUES (?)", (name,))
    return cur.lastrowid


def get_or_create_model(conn, engine, model_name, prompt_version, description=None):
    row = conn.execute(
        "SELECT model_id FROM llm_models WHERE engine=? AND model_name=? AND prompt_version=?",
        (engine, model_name, prompt_version),
    ).fetchone()
    if row:
        return row["model_id"]
    cur = conn.execute(
        "INSERT INTO llm_models (engine, model_name, prompt_version, description) VALUES (?,?,?,?)",
        (engine, model_name, prompt_version, description),
    )
    return cur.lastrowid


# ------------------------------------------------------------------ writes
def save_analysis(input_text, text_type, source, company_name, result, client_ip=None):
    """
    Persist one request + the LLM response in a single transaction.
    `result` is the dict produced by sentiment_engine.SentimentEngine.analyze().
    Returns the stored record (as served by the history view).
    """
    with get_connection() as conn:
        # Prefer the company typed by the user; otherwise the one detected by the LLM.
        company_id = get_or_create_company(conn, company_name or result.get("company"))
        model_id = get_or_create_model(
            conn, result["engine"], result["model_name"], result["prompt_version"],
            result.get("model_description"),
        )
        cur = conn.execute(
            """INSERT INTO analysis_requests (company_id, input_text, text_type, source, client_ip)
               VALUES (?,?,?,?,?)""",
            (company_id, input_text, text_type, source, client_ip),
        )
        request_id = cur.lastrowid
        p = result["probabilities"]
        cur = conn.execute(
            """INSERT INTO sentiment_results
               (request_id, model_id, sentiment, confidence, sentiment_score,
                prob_positive, prob_negative, prob_neutral, detected_company,
                reasoning, key_factors, raw_response, latency_ms, fallback_from)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                request_id, model_id, result["sentiment"], result["confidence"],
                result["score"], p["Positive"], p["Negative"], p["Neutral"],
                result.get("company"), result.get("reasoning"),
                json.dumps(result.get("key_factors") or []),
                result.get("raw_response"), result.get("latency_ms"),
                "; ".join(result.get("fallback_from") or []) or None,
            ),
        )
        result_id = cur.lastrowid
        row = conn.execute(
            "SELECT * FROM v_sentiment_history WHERE result_id = ?", (result_id,)
        ).fetchone()
        return _row_to_dict(row)


def delete_result(result_id):
    """Deletes the request (the result is removed through ON DELETE CASCADE)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT request_id FROM sentiment_results WHERE result_id = ?", (result_id,)
        ).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM analysis_requests WHERE request_id = ?", (row["request_id"],))
        return True


# ------------------------------------------------------------------ reads
def get_result(result_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM v_sentiment_history WHERE result_id = ?", (result_id,)
        ).fetchone()
        return _row_to_dict(row)


def list_results(limit=50, offset=0, sentiment=None, company=None, text_type=None,
                 source=None, search=None):
    sql = "SELECT * FROM v_sentiment_history WHERE 1=1"
    args = []
    if sentiment:
        sql += " AND sentiment = ?"
        args.append(sentiment)
    if company:
        sql += " AND (company_name LIKE ? OR detected_company LIKE ?)"
        args += [f"%{company}%", f"%{company}%"]
    if text_type:
        sql += " AND text_type = ?"
        args.append(text_type)
    if source:
        sql += " AND source = ?"
        args.append(source)
    if search:
        sql += " AND input_text LIKE ?"
        args.append(f"%{search}%")
    count_sql = sql.replace("SELECT *", "SELECT COUNT(*)", 1)
    sql += " ORDER BY result_id DESC LIMIT ? OFFSET ?"
    with get_connection() as conn:
        total = conn.execute(count_sql, args).fetchone()[0]
        rows = conn.execute(sql, args + [limit, offset]).fetchall()
        return total, [_row_to_dict(r) for r in rows]


def get_stats():
    with get_connection() as conn:
        by_sent = {
            r["sentiment"]: r["n"]
            for r in conn.execute(
                "SELECT sentiment, COUNT(*) n FROM sentiment_results GROUP BY sentiment")
        }
        by_type = [
            dict(r) for r in conn.execute(
                """SELECT text_type, sentiment, COUNT(*) n FROM v_sentiment_history
                   GROUP BY text_type, sentiment ORDER BY text_type""")
        ]
        # Aggregate per company in Python so that name variants are merged (see company_key).
        groups = {}
        for r in conn.execute(
                """SELECT COALESCE(company_name, detected_company) company, sentiment, sentiment_score
                   FROM v_sentiment_history
                   WHERE COALESCE(company_name, detected_company) IS NOT NULL ORDER BY result_id"""):
            g = groups.setdefault(company_key(r["company"]), {
                "company": r["company"], "n": 0, "score": 0.0, "pos": 0, "neg": 0, "neu": 0})
            g["n"] += 1
            g["score"] += r["sentiment_score"]
            g[{"Positive": "pos", "Negative": "neg", "Neutral": "neu"}[r["sentiment"]]] += 1
        by_company = sorted(
            ({"company": g["company"], "n": g["n"], "avg_score": round(g["score"] / g["n"], 3),
              "pos": g["pos"], "neg": g["neg"], "neu": g["neu"]} for g in groups.values()),
            key=lambda g: -g["n"])[:15]
        totals = conn.execute(
            """SELECT COUNT(*) n, ROUND(AVG(confidence),3) avg_conf,
                      ROUND(AVG(latency_ms)) avg_latency FROM sentiment_results""").fetchone()
        return {
            "total_analyses": totals["n"],
            "average_confidence": totals["avg_conf"],
            "average_latency_ms": totals["avg_latency"],
            "by_sentiment": {k: by_sent.get(k, 0) for k in config.SENTIMENT_LABELS},
            "by_text_type": by_type,
            "by_company": by_company,
        }


def list_companies():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT c.company_id, c.company_name, COUNT(q.request_id) analyses
               FROM companies c LEFT JOIN analysis_requests q ON q.company_id = c.company_id
               GROUP BY c.company_id ORDER BY c.company_name""")]


# ------------------------------------------------------------------ test data set
def load_test_dataset(csv_path, db_path=None):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with get_connection(db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO test_dataset
               (test_id, text_type, company_name, input_text, expected_sentiment)
               VALUES (:test_id, :text_type, :company_name, :input_text, :expected_sentiment)""",
            rows,
        )
    return len(rows)


def get_test_dataset(text_type=None):
    sql = "SELECT * FROM test_dataset"
    args = []
    if text_type:
        sql += " WHERE text_type = ?"
        args.append(text_type)
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(sql + " ORDER BY test_id", args)]


def create_test_run(engine, model_name, prompt_version):
    with get_connection() as conn:
        model_id = get_or_create_model(conn, engine, model_name, prompt_version)
        cur = conn.execute("INSERT INTO test_runs (model_id) VALUES (?)", (model_id,))
        return cur.lastrowid


def add_test_run_item(run_id, test_id, result_id, predicted, expected):
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO test_run_items
               (run_id, test_id, result_id, predicted, expected, is_correct)
               VALUES (?,?,?,?,?,?)""",
            (run_id, test_id, result_id, predicted, expected, int(predicted == expected)),
        )


def finish_test_run(run_id, accuracy, macro_f1, avg_latency_ms):
    with get_connection() as conn:
        conn.execute(
            """UPDATE test_runs SET
                 total_items   = (SELECT COUNT(*) FROM test_run_items WHERE run_id = ?),
                 correct_items = (SELECT COALESCE(SUM(is_correct),0) FROM test_run_items WHERE run_id = ?),
                 accuracy = ?, macro_f1 = ?, avg_latency_ms = ?,
                 finished_at = datetime('now','localtime')
               WHERE run_id = ?""",
            (run_id, run_id, accuracy, macro_f1, avg_latency_ms, run_id),
        )


def list_test_runs():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            """SELECT t.*, m.engine, m.model_name, m.prompt_version
               FROM test_runs t JOIN llm_models m ON m.model_id = t.model_id
               ORDER BY t.run_id DESC""")]


def get_test_run(run_id):
    with get_connection() as conn:
        run = conn.execute(
            """SELECT t.*, m.engine, m.model_name, m.prompt_version
               FROM test_runs t JOIN llm_models m ON m.model_id = t.model_id
               WHERE t.run_id = ?""", (run_id,)).fetchone()
        if not run:
            return None
        items = conn.execute(
            """SELECT i.test_id, d.text_type, d.company_name, d.input_text,
                      i.expected, i.predicted, i.is_correct,
                      r.confidence, r.sentiment_score, r.reasoning, r.latency_ms
               FROM test_run_items i
               JOIN test_dataset d ON d.test_id = i.test_id
               LEFT JOIN sentiment_results r ON r.result_id = i.result_id
               WHERE i.run_id = ? ORDER BY i.test_id""", (run_id,)).fetchall()
        return {"run": dict(run), "items": [dict(i) for i in items]}
