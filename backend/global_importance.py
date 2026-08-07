"""
NodeSense global feature importance.

Per-alert SHAP (explain.py / the /predict endpoint) answers "why was THIS
flow flagged". This script answers the aggregate question: across the
held-out test set, which of the 20 features actually drives the model's
decisions? Computed as mean |SHAP value| per feature over a sample of test
flows, each explained toward its own predicted class.

KernelSHAP is the cost center here (a full explanation per sampled flow),
so this defaults to a modest sample rather than the whole test set.

Usage:
    python global_importance.py                # 30 test flows, ~1-2 min
    python global_importance.py --sample-size 60 --nsamples 150
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort

from data import CLASS_NAMES, FEATURE_NAMES
from explain import FlowExplainer
from train import build_dataset

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
EVAL_DIR = os.path.join(ARTIFACT_DIR, "eval")


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def plot_importance(feature_names: list[str], values: list[float], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = range(len(feature_names))
    ax.barh(list(y_pos), values, color="#3987e5")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("NodeSense global feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(sample_size: int, nsamples: int, seed: int):
    os.makedirs(EVAL_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACT_DIR, "model.onnx")
    if not os.path.exists(model_path):
        raise SystemExit("No model at artifacts/model.onnx — run train.py first.")

    ds = build_dataset()
    session = ort.InferenceSession(model_path)
    background = np.load(os.path.join(ARTIFACT_DIR, "background.npy"))

    with open(os.path.join(ARTIFACT_DIR, "preprocess.json")) as f:
        pre = json.load(f)
    explainer = FlowExplainer(session, background, pre["seq_len"])

    # Sample flows from the test set (one flow per session, its predicted
    # class decides which target SHAP explains toward).
    rng = np.random.default_rng(seed)
    n_sessions = len(ds["X_te"])
    idx = rng.choice(n_sessions, size=min(sample_size, n_sessions), replace=False)
    flows = ds["X_te"][idx, 0, :]  # (sample_size, n_feat)

    seqs = np.repeat(flows[:, None, :], pre["seq_len"], axis=1).astype(np.float32)
    logits = session.run(None, {"input": seqs})[0]
    preds = _softmax(logits).argmax(axis=1)

    result = explainer.global_importance(flows, preds, FEATURE_NAMES, nsamples=nsamples)

    with open(os.path.join(EVAL_DIR, "global_importance.json"), "w") as f:
        json.dump(result, f, indent=2)
    plot_importance(
        result["feature_names"], result["mean_abs_shap"],
        os.path.join(EVAL_DIR, "global_importance.png"),
    )

    print(json.dumps(result, indent=2))
    print(f"\nWrote global_importance.json/png -> {EVAL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--nsamples", type=int, default=100,
                         help="KernelSHAP background perturbation count per flow")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(args.sample_size, args.nsamples, args.seed)
