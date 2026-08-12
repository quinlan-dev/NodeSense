"""
NodeSense dataset assembly and preprocessing.

Split out of train.py so that importing "how the held-out split and
scaling work" doesn't drag in torch. evaluate.py, global_importance.py,
and backend/tests/test_preprocessing.py only need this module — none of
them train anything, so none of them should need requirements-train.txt.
train.py and zero_day_eval.py (which do train) import build_dataset from
here too, so there is still exactly one definition of "the held-out split."
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import FEATURE_NAMES, generate_sessions, load_cicids

# Heavy-tailed features get log1p before scaling; flag counts and ratios don't.
LOG_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Mean", "Bwd Packet Length Max",
    "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Mean", "Bwd IAT Mean",
    "Average Packet Size", "Idle Mean",
]
LOG_MASK = np.array([f in LOG_FEATURES for f in FEATURE_NAMES])


def preprocess_fit(X):
    """Fit log+standard scaling on (n_sessions, SEQ_LEN, n_feat) raw flows."""
    flat = X.reshape(-1, X.shape[-1]).copy()
    flat[:, LOG_MASK] = np.log1p(np.clip(flat[:, LOG_MASK], 0, None))
    scaler = StandardScaler().fit(flat)
    return scaler


def preprocess_apply(X, scaler):
    shape = X.shape
    flat = X.reshape(-1, shape[-1]).astype(np.float64)
    flat[:, LOG_MASK] = np.log1p(np.clip(flat[:, LOG_MASK], 0, None))
    flat = scaler.transform(flat)
    return flat.reshape(shape).astype(np.float32)


def build_dataset(data_glob: str | None = None, sessions: int = 4000, seed: int = 42,
                   exclude_class: int | None = None):
    """Load raw data, split, and fit/apply preprocessing — the exact steps
    the shipped model's artifacts were produced by. Shared by train.py's
    main, evaluate.py, and zero_day_eval.py so "held-out test set" always
    means the same rows and the same scaler for every script that reports
    numbers about this model.

    exclude_class drops all sessions of that class index from the training
    split only (used by the leave-one-attack-out zero-day evaluation); the
    test split is untouched so held-out recall is measured against the
    same sessions every other script reports on.
    """
    import glob as glob_module

    if data_glob:
        X, y, y_flow = load_cicids(sorted(glob_module.glob(data_glob)))
    else:
        X, y, y_flow = generate_sessions(n_sessions=sessions, seed=seed)

    X_tr, X_te, y_tr, y_te, yf_tr, yf_te = train_test_split(
        X, y, y_flow, test_size=0.2, random_state=42, stratify=y
    )

    if exclude_class is not None:
        keep = y_tr != exclude_class
        X_tr, y_tr, yf_tr = X_tr[keep], y_tr[keep], yf_tr[keep]

    scaler = preprocess_fit(X_tr)
    Xs_tr = preprocess_apply(X_tr, scaler)
    Xs_te = preprocess_apply(X_te, scaler)
    return {
        "X_tr": Xs_tr, "X_te": Xs_te,
        "y_tr": y_tr, "y_te": y_te,
        "yf_tr": yf_tr, "yf_te": yf_te,
        "scaler": scaler,
    }
