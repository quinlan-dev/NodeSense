"""
NodeSense latency benchmark.

Times the two request shapes the server actually handles: a plain
prediction, and a prediction with a SHAP explanation. Reports p50/p95/p99
in milliseconds so "real-time" is a measured claim, not an assertion.

Talks directly to the loaded model/explainer (via app.py's module-level
helpers) rather than over HTTP, so the numbers reflect inference cost
without also measuring the local network stack. Run scripts/http_smoke.sh
or hit the deployed URL with curl -w if end-to-end HTTP latency (including
a cold Hugging Face Space) is what you want instead.

Usage:
    python benchmark.py                 # 200 predict-only + 50 predict+explain
    python benchmark.py --n 500 --n-explain 100
"""

import argparse
import json
import os
import time

import numpy as np

import app as nodesense_app
from data import FEATURE_NAMES, generate_sessions

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
EVAL_DIR = os.path.join(ARTIFACT_DIR, "eval")


def percentiles(times_ms: list[float]) -> dict:
    arr = np.array(times_ms)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "mean": round(float(arr.mean()), 2),
        "n": len(arr),
    }


def sample_flows(n: int) -> np.ndarray:
    X, _, _ = generate_sessions(n_sessions=n, seed=123)
    return X[:, 0, :]  # one representative flow per session


def main(n_predict: int, n_explain: int):
    if not nodesense_app.load_artifacts():
        raise SystemExit("No model at artifacts/model.onnx — run train.py first.")

    os.makedirs(EVAL_DIR, exist_ok=True)
    flows = sample_flows(max(n_predict, n_explain))

    predict_times = []
    for flow in flows[:n_predict]:
        t0 = time.perf_counter()
        nodesense_app.real_prediction(list(map(float, flow)), explain=False)
        predict_times.append((time.perf_counter() - t0) * 1000)

    explain_times = []
    for flow in flows[:n_explain]:
        t0 = time.perf_counter()
        nodesense_app.real_prediction(list(map(float, flow)), explain=True)
        explain_times.append((time.perf_counter() - t0) * 1000)

    result = {
        "predict_only_ms": percentiles(predict_times),
        "predict_with_explanation_ms": percentiles(explain_times),
    }
    with open(os.path.join(EVAL_DIR, "latency.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nWrote latency.json -> {EVAL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="predict-only requests")
    parser.add_argument("--n-explain", type=int, default=50, help="predict+explain requests")
    args = parser.parse_args()
    main(args.n, args.n_explain)
