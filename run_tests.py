"""
Tests the SMSA system with the labelled test data set (data/test_dataset.csv).

Every text is sent to the backend endpoint POST /api/analyze (exactly as the web
front end does), the LLM response is stored in the database, and the prediction
is compared with the expected sentiment. A test run with accuracy / macro-F1 is
stored in the tables test_runs and test_run_items, and a CSV, JSON summary and
charts are written to results/.

Usage
    python run_tests.py                                   # default engine
    python run_tests.py --engine gemini
    python run_tests.py --engine llm --prompt zeroshot-v1
    python run_tests.py --engine finbert
    python run_tests.py --url http://127.0.0.1:5050       # against a running server
"""
import argparse
import csv
import json
import os
import time
import urllib.request

import config
import database as db


def post_via_http(base_url, payload):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/analyze", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.status, json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=config.DEFAULT_ENGINE, choices=config.ENGINES)
    ap.add_argument("--prompt", default=config.DEFAULT_PROMPT_VERSION,
                    choices=["fewshot-v2", "zeroshot-v1"])
    ap.add_argument("--url", help="base URL of a running server (default: in-process Flask test client)")
    ap.add_argument("--limit", type=int, help="only test the first N texts")
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    db.init_db()
    items = db.get_test_dataset()[: args.limit] if args.limit else db.get_test_dataset()
    prompt = "n/a" if args.engine == "finbert" else (
        "fewshot-v2" if args.engine in ("groq", "gemini") else args.prompt)
    model_name = {"groq": config.GROQ_MODEL_ID, "gemini": config.GEMINI_MODEL_ID, "llm": config.LLM_MODEL_ID,
                  "finbert": config.FINBERT_MODEL_ID}[args.engine]

    if args.url:
        send = lambda p: post_via_http(args.url, p)
    else:
        from app import app
        client = app.test_client()

        def send(p):
            r = client.post("/api/analyze", json=p)
            return r.status_code, r.get_json()

    run_id = None
    rows, y_true, y_pred, latencies = [], [], [], []
    print(f"Testing {len(items)} texts  engine={args.engine}  model={model_name}  prompt={prompt}")
    t0 = time.time()
    for n, it in enumerate(items, 1):
        # fallback is disabled so that every prediction of a run comes from the model under test
        payload = {"text": it["input_text"], "text_type": it["text_type"], "engine": args.engine,
                   "prompt_version": args.prompt, "source": "test", "fallback": False}
        status, body = send(payload)
        if status != 201:
            print(f"  #{it['test_id']:>2}  ERROR {status}: {body.get('error')}")
            continue
        res = body["result"]
        if run_id is None:   # model name as reported by the backend (e.g. the exact Gemini version)
            model_name = res["model_name"]
            run_id = db.create_test_run(args.engine, model_name, res["prompt_version"])
        db.add_test_run_item(run_id, it["test_id"], res["result_id"], res["sentiment"],
                             it["expected_sentiment"])
        ok = res["sentiment"] == it["expected_sentiment"]
        y_true.append(it["expected_sentiment"]); y_pred.append(res["sentiment"])
        latencies.append(res["latency_ms"])
        rows.append({
            "test_id": it["test_id"], "text_type": it["text_type"], "company": it["company_name"],
            "input_text": it["input_text"], "expected": it["expected_sentiment"],
            "predicted": res["sentiment"], "correct": int(ok), "confidence": res["confidence"],
            "score": res["sentiment_score"], "detected_company": res.get("detected_company"),
            "reasoning": res.get("reasoning"), "latency_ms": res["latency_ms"],
        })
        print(f"  #{it['test_id']:>2} {it['text_type']:<12} expected={it['expected_sentiment']:<8} "
              f"predicted={res['sentiment']:<8} conf={res['confidence']:.2f} {'OK' if ok else 'WRONG'}")

    if not rows:
        raise SystemExit("No successful predictions - check the model / API configuration.")

    labels = config.SENTIMENT_LABELS
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    avg_lat = sum(latencies) / len(latencies)
    db.finish_test_run(run_id, acc, f1, avg_lat)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_type = {}
    for r in rows:
        t = per_type.setdefault(r["text_type"], {"n": 0, "correct": 0})
        t["n"] += 1; t["correct"] += r["correct"]
    for t in per_type.values():
        t["accuracy"] = t["correct"] / t["n"]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    tag = f"run{run_id}_{args.engine}" + (f"_{prompt}" if prompt != "n/a" else "")
    with open(os.path.join(config.RESULTS_DIR, f"{tag}_predictions.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = {
        "run_id": run_id, "engine": args.engine, "model": model_name, "prompt_version": prompt,
        "items": len(rows), "correct": sum(r["correct"] for r in rows), "accuracy": acc,
        "macro_f1": f1, "avg_latency_ms": avg_lat, "wall_time_s": round(time.time() - t0, 1),
        "per_class": {k: report[k] for k in labels}, "per_text_type": per_type,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }
    with open(os.path.join(config.RESULTS_DIR, f"{tag}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    save_confusion_png(cm, labels, os.path.join(config.RESULTS_DIR, f"{tag}_confusion.png"),
                       f"{model_name} ({prompt})  accuracy {acc:.1%}")

    print(f"\nRun #{run_id}: accuracy {acc:.1%}  macro-F1 {f1:.3f}  "
          f"avg latency {avg_lat/1000:.2f}s  ({summary['correct']}/{len(rows)} correct)")
    for t, v in per_type.items():
        print(f"  {config.TEXT_TYPES[t]:<30} {v['correct']}/{v['n']}")
    print(f"Results written to {config.RESULTS_DIR}/{tag}_*")


def save_confusion_png(cm, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Expected")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
