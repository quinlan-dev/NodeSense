"""
NodeSense zero-day (leave-one-attack-out) evaluation.

The core research question: does NodeSense generalize to attack types it
never saw in training, the way a real zero-day would look to a deployed
model? For each attack class in turn, this trains a fresh transformer with
every session of that class removed from training, then measures recall
on the held-out class's test sessions — scored as "did the model flag
this as *any* anomaly", since a model that never saw Botnet during
training has no way to specifically name it "Botnet", only to notice the
traffic looks wrong.

This needs the training dependencies (torch, onnx, onnxscript) — it is not
part of the lightweight test/eval path evaluate.py uses.

Usage:
    python zero_day_eval.py                  # all 5 attack classes
    python zero_day_eval.py --epochs 12      # more training per fold
"""

import argparse
import json
import os

import numpy as np
import torch

from data import CLASS_NAMES
from train import build_dataset, train_transformer

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
EVAL_DIR = os.path.join(ARTIFACT_DIR, "eval")


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_fold(held_out_idx: int, sessions: int, epochs: int) -> dict:
    held_out_name = CLASS_NAMES[held_out_idx]
    print(f"\n=== Leave-one-out fold: holding out {held_out_name!r} from training ===")

    ds = build_dataset(sessions=sessions, exclude_class=held_out_idx)
    model = train_transformer(
        ds["X_tr"], ds["y_tr"], ds["X_te"], ds["y_te"], epochs=epochs
    )

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(ds["X_te"]))
        probs = _softmax(logits.numpy())
    y_pred = probs.argmax(axis=1)

    held_mask = ds["y_te"] == held_out_idx
    n_held = int(held_mask.sum())
    # Recall = fraction of held-out-class sessions flagged as *some*
    # anomaly (correct class credit isn't possible — the model was never
    # given that class label to predict).
    recall = float((y_pred[held_mask] != 0).mean()) if n_held else None

    # For context: how the model does on everything it WAS trained on,
    # in the same fold, so a low zero-day recall isn't confused with a
    # generally broken model.
    other_mask = ~held_mask
    other_acc = float((y_pred[other_mask] == ds["y_te"][other_mask]).mean())

    return {
        "held_out_class": held_out_name,
        "n_held_out_test_sessions": n_held,
        "zero_day_anomaly_recall": round(recall, 4) if recall is not None else None,
        "in_distribution_accuracy": round(other_acc, 4),
    }


def main(sessions: int, epochs: int):
    os.makedirs(EVAL_DIR, exist_ok=True)
    results = [run_fold(i, sessions, epochs) for i in range(1, len(CLASS_NAMES))]

    summary = {
        "description": (
            "Each row trains a model with the named class fully removed "
            "from training, then reports what fraction of that class's "
            "held-out test sessions were still flagged as anomalous."
        ),
        "folds": results,
        "mean_zero_day_recall": round(
            float(np.mean([r["zero_day_anomaly_recall"] for r in results
                           if r["zero_day_anomaly_recall"] is not None])), 4
        ),
    }
    with open(os.path.join(EVAL_DIR, "zero_day.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Zero-day summary ===")
    for r in results:
        print(f"  {r['held_out_class']:<14} recall={r['zero_day_anomaly_recall']}"
              f"  (in-distribution acc this fold: {r['in_distribution_accuracy']})")
    print(f"  mean recall across all held-out classes: {summary['mean_zero_day_recall']}")
    print(f"\nWrote zero_day.json -> {EVAL_DIR}")
    print(
        "\nNote: folds retrain from scratch and are NOT saved over the "
        "shipped artifacts/model.onnx — this script only measures "
        "generalization, it does not change what's deployed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    main(args.sessions, args.epochs)
