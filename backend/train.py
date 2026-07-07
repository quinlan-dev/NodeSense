"""
NodeSense training pipeline.

Trains and benchmarks all three models, then exports the transformer plus
its preprocessing state to artifacts/ for the inference server:

    artifacts/model.onnx        transformer, input (batch, 16, 20)
    artifacts/preprocess.json   scaler mean/scale, feature names, classes
    artifacts/background.npy    scaled flow sample for KernelSHAP

Usage:
    python train.py                          # synthetic data (default)
    python train.py --sessions 8000 --epochs 15
    python train.py --data "../data/*.csv"   # real CICIDS-2018 CSVs
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import CLASS_NAMES, FEATURE_NAMES, SEQ_LEN, generate_sessions, load_cicids
from models import Autoencoder, NetworkTransformer, build_random_forest

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

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


def eval_binary(name, y_true, y_pred, y_score):
    """All models are compared on the shared task both can do:
    anomaly vs benign."""
    yt = (np.asarray(y_true) > 0).astype(int)
    yp = (np.asarray(y_pred) > 0).astype(int)
    print(f"\n=== {name} (binary anomaly detection) ===")
    print(classification_report(yt, yp, target_names=["Benign", "Attack"], digits=3))
    if y_score is not None:
        print(f"AUC-ROC: {roc_auc_score(yt, y_score):.4f}")


def train_rf(Xf_train, yf_train, Xf_test, yf_test):
    rf = build_random_forest()
    rf.fit(Xf_train, yf_train)
    y_pred = rf.predict(Xf_test)
    y_score = 1.0 - rf.predict_proba(Xf_test)[:, 0]
    eval_binary("Random Forest (per flow)", yf_test, y_pred, y_score)
    return rf


def train_autoencoder(Xf_train, yf_train, Xf_test, yf_test, epochs=20):
    benign = torch.tensor(Xf_train[yf_train == 0], dtype=torch.float32)
    model = Autoencoder(input_dim=benign.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(benign), batch_size=256, shuffle=True
    )
    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), xb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        err_train = ((model(benign) - benign) ** 2).mean(dim=1).numpy()
        xt = torch.tensor(Xf_test, dtype=torch.float32)
        err_test = ((model(xt) - xt) ** 2).mean(dim=1).numpy()
    threshold = np.percentile(err_train, 99)
    eval_binary("Autoencoder (per flow)", yf_test,
                (err_test > threshold).astype(int), err_test)
    return model, threshold


def train_transformer(X_train, y_train, X_test, y_test, epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NetworkTransformer(
        input_dim=X_train.shape[-1], seq_len=SEQ_LEN,
        num_classes=len(CLASS_NAMES),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_train), torch.tensor(y_train)
        ),
        batch_size=64, shuffle=True,
    )
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}  loss: {total / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = probs.argmax(axis=1)
    eval_binary("Transformer (per session)", y_test, y_pred, 1.0 - probs[:, 0])
    print("=== Transformer per-class report ===")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                digits=3, zero_division=0))
    return model.cpu()


def export_artifacts(model, scaler, X_train_scaled):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, SEQ_LEN, len(FEATURE_NAMES))
    # The dynamo exporter specializes internal attention reshapes to the
    # dummy batch size, breaking batched inference; the legacy exporter
    # handles dynamic batch correctly for nn.TransformerEncoder.
    torch.onnx.export(
        model, (dummy,), os.path.join(ARTIFACT_DIR, "model.onnx"),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    with open(os.path.join(ARTIFACT_DIR, "preprocess.json"), "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "class_names": CLASS_NAMES,
            "seq_len": SEQ_LEN,
            "log_mask": LOG_MASK.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }, f, indent=2)
    # Random scaled flows as the SHAP background distribution
    flows = X_train_scaled.reshape(-1, len(FEATURE_NAMES))
    idx = np.random.default_rng(0).choice(len(flows), 50, replace=False)
    np.save(os.path.join(ARTIFACT_DIR, "background.npy"), flows[idx])
    print(f"\nExported model.onnx, preprocess.json, background.npy -> {ARTIFACT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="glob of CICIDS-2018 CSVs; omit for synthetic")
    parser.add_argument("--sessions", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    if args.data:
        X, y, y_flow = load_cicids(sorted(glob.glob(args.data)))
    else:
        X, y, y_flow = generate_sessions(n_sessions=args.sessions)
    print(f"{len(X)} sessions of {SEQ_LEN} flows, "
          f"class counts: {np.bincount(y, minlength=len(CLASS_NAMES)).tolist()}")

    X_tr, X_te, y_tr, y_te, yf_tr, yf_te = train_test_split(
        X, y, y_flow, test_size=0.2, random_state=42, stratify=y
    )
    scaler = preprocess_fit(X_tr)
    Xs_tr, Xs_te = preprocess_apply(X_tr, scaler), preprocess_apply(X_te, scaler)

    if not args.skip_baselines:
        flat_tr = Xs_tr.reshape(-1, Xs_tr.shape[-1])
        flat_te = Xs_te.reshape(-1, Xs_te.shape[-1])
        train_rf(flat_tr, yf_tr.ravel(), flat_te, yf_te.ravel())
        train_autoencoder(flat_tr, yf_tr.ravel(), flat_te, yf_te.ravel())

    model = train_transformer(Xs_tr, y_tr, Xs_te, y_te, epochs=args.epochs)
    export_artifacts(model, scaler, Xs_tr)
