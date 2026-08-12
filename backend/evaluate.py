"""
NodeSense evaluation script.

Reports the numbers the paper needs, computed against the exact held-out
test split the shipped model's artifacts were produced from (via
train.build_dataset, same seed/session-count/split as train.py's default
run) so these are honest held-out numbers, not train-set numbers.

Writes to backend/artifacts/eval/:
    metrics.json           precision/recall/F1 per class, macro/weighted,
                            per-class ROC AUC (one-vs-rest), benign FPR
    confusion_matrix.csv   raw counts
    confusion_matrix.png   plotted confusion matrix
    calibration.json       confidence calibration (ECE) before/after
                            temperature scaling — reporting only, the
                            serving model is unaffected

Usage:
    python evaluate.py                      # evaluate the committed model
    python evaluate.py --sessions 8000       # match a different train run
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
)

from data import CLASS_NAMES, FEATURE_NAMES, SEQ_LEN
from dataset import build_dataset

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
EVAL_DIR = os.path.join(ARTIFACT_DIR, "eval")


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_model(session: ort.InferenceSession, X: np.ndarray) -> np.ndarray:
    """Batched inference over (n, seq_len, n_feat) -> (n, n_classes) probs."""
    logits = session.run(None, {"input": X.astype(np.float32)})[0]
    return _softmax(logits)


def expected_calibration_error(probs_max: np.ndarray, correct: np.ndarray,
                                n_bins: int = 10) -> float:
    """Standard ECE: bin predictions by confidence, compare each bin's
    average confidence to its actual accuracy, weight by bin size."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs_max)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_max > lo) & (probs_max <= hi)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = probs_max[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Grid-search temperature scaling: T minimizing NLL on the test set.

    A proper implementation fits T on a held-out validation split distinct
    from the test set; here test doubles as both because the dataset is
    synthetic and small. Treat this as a calibration *demonstration*, not
    a production calibration fit — the model card notes this.
    """
    best_T, best_nll = 1.0, float("inf")
    y_idx = y_true
    for T in np.arange(0.5, 5.01, 0.1):
        probs = _softmax(logits / T)
        nll = -np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_nll, best_T = nll, T
    return round(float(best_T), 2)


def plot_confusion(cm: np.ndarray, class_names: list[str], path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("NodeSense confusion matrix (held-out test set)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(sessions: int, data_glob: str | None):
    os.makedirs(EVAL_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACT_DIR, "model.onnx")
    if not os.path.exists(model_path):
        raise SystemExit(
            "No model at artifacts/model.onnx — run train.py first, or "
            "evaluate.py has nothing to score."
        )

    ds = build_dataset(data_glob=data_glob, sessions=sessions)
    X_te, y_te = ds["X_te"], ds["y_te"]

    session = ort.InferenceSession(model_path)
    probs = run_model(session, X_te)
    y_pred = probs.argmax(axis=1)

    report = classification_report(
        y_te, y_pred, target_names=CLASS_NAMES, output_dict=True,
        digits=3, zero_division=0,
    )

    # One-vs-rest ROC AUC per class (only for classes present in y_te).
    present = sorted(set(y_te.tolist()))
    auc_per_class = {}
    for c in present:
        y_bin = (y_te == c).astype(int)
        try:
            auc_per_class[CLASS_NAMES[c]] = round(float(roc_auc_score(y_bin, probs[:, c])), 4)
        except ValueError:
            auc_per_class[CLASS_NAMES[c]] = None

    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(CLASS_NAMES))))

    # Benign false positive rate: fraction of true-benign sessions the
    # model flagged as any attack. This is the number that determines
    # analyst alert fatigue in a real deployment.
    benign_mask = y_te == 0
    benign_fpr = float((y_pred[benign_mask] != 0).mean()) if benign_mask.any() else None

    metrics = {
        "n_test_sessions": int(len(y_te)),
        "class_names": CLASS_NAMES,
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES if name in report
        },
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
        "roc_auc_ovr": auc_per_class,
        "benign_false_positive_rate": benign_fpr,
    }
    with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.savetxt(
        os.path.join(EVAL_DIR, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",",
        header=",".join(CLASS_NAMES), comments="",
    )
    plot_confusion(cm, CLASS_NAMES, os.path.join(EVAL_DIR, "confusion_matrix.png"))

    # Calibration: is softmax confidence trustworthy as-is?
    logits = np.log(np.clip(probs, 1e-12, 1.0))  # recover logit-like scores for re-scaling
    conf = probs.max(axis=1)
    correct = (y_pred == y_te).astype(int)
    ece_before = expected_calibration_error(conf, correct)
    T = fit_temperature(logits, y_te)
    probs_t = _softmax(logits / T)
    conf_t = probs_t.max(axis=1)
    ece_after = expected_calibration_error(conf_t, correct)
    calibration = {
        "temperature": T,
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "note": (
            "Fit and evaluated on the same held-out split (synthetic data is "
            "small); treat as a demonstration of the calibration gap, not a "
            "production calibration fit. Serving still returns raw softmax "
            "confidence — see docs/MODEL_CARD.md."
        ),
    }
    with open(os.path.join(EVAL_DIR, "calibration.json"), "w") as f:
        json.dump(calibration, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nCalibration: ECE {ece_before:.4f} -> {ece_after:.4f} at T={T}")
    print(f"\nWrote metrics.json, confusion_matrix.csv/png, calibration.json -> {EVAL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=4000,
                         help="must match the session count train.py used")
    parser.add_argument("--data", help="glob of CICIDS-2018 CSVs, if evaluating a real-data model")
    args = parser.parse_args()
    main(args.sessions, args.data)
