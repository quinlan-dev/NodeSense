"""
NodeSense training pipeline.

Usage:
    python train.py --data ../data/cicids2018.csv --model transformer
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import build_random_forest, Autoencoder, NetworkTransformer

SKEWED_FEATURES = [
    "Flow Duration", "Fwd Packet Length Max",
    "Bwd Packet Length Max", "Flow Bytes/s",
]


def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    present = [c for c in SKEWED_FEATURES if c in df.columns]
    df[present] = np.log1p(df[present].clip(lower=0))

    df["Label"] = (df["Label"] != "BENIGN").astype(int)
    X = df.drop("Label", axis=1).select_dtypes(include=[np.number]).values
    y = df["Label"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler


def train_rf(X_train, y_train, X_test, y_test):
    rf = build_random_forest()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
    return rf


def train_transformer(X_train, y_train, X_test, y_test, epochs=10, seq_len=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X_train.shape[1]

    def to_sequences(X, y):
        n = (len(X) // seq_len) * seq_len
        Xs = torch.tensor(X[:n], dtype=torch.float32).reshape(-1, seq_len, input_dim)
        # Label a sequence as attack if any flow in it is an attack
        ys = torch.tensor(y[:n].reshape(-1, seq_len).max(axis=1), dtype=torch.long)
        return Xs, ys

    Xtr, ytr = to_sequences(X_train, y_train)
    Xte, yte = to_sequences(X_test, y_test)

    model = NetworkTransformer(input_dim=input_dim, seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, ytr), batch_size=64, shuffle=True
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}  loss: {total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    print(classification_report(yte.numpy(), y_pred))
    print(f"AUC-ROC: {roc_auc_score(yte.numpy(), y_prob):.4f}")
    return model


def export_onnx(model, input_dim, seq_len, path="model.onnx"):
    model.eval()
    dummy = torch.randn(1, seq_len, input_dim)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
    )
    print(f"Exported to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", choices=["rf", "transformer"], default="transformer")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    X, y, scaler = load_and_preprocess(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if args.model == "rf":
        train_rf(X_train, y_train, X_test, y_test)
    else:
        model = train_transformer(X_train, y_train, X_test, y_test, epochs=args.epochs)
        export_onnx(model, X.shape[1], 16)
