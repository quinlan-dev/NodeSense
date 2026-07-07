"""
NodeSense model architectures.

Three candidates trained and benchmarked against each other:
1. Random forest (supervised per-flow baseline)
2. Autoencoder (unsupervised baseline, detects via reconstruction error)
3. Transformer (primary model, classifies flow sequences, multi-class)

The transformer is deliberately compact (about 150k parameters): it trains
in minutes on CPU, exports to a sub-megabyte ONNX file that can live in
git, and is fast enough for KernelSHAP's repeated evaluations at serve
time. On the demo data it saturates accuracy anyway; scale d_model/layers
up when training on the full CICIDS-2018.
"""

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


def build_random_forest(n_estimators: int = 200) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


class Autoencoder(nn.Module):
    """Trained on benign traffic only. Flags anomalies when
    reconstruction error exceeds a threshold set on validation data."""

    def __init__(self, input_dim: int = 20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class NetworkTransformer(nn.Module):
    """Classifies sequences of network flows into attack classes.

    Each flow is one timestep, so the model can learn temporal attack
    patterns like scan sweeps or beaconing that single-flow classifiers
    miss. Mean pooling over timesteps (rather than a last-token readout)
    keeps predictions stable when the server tiles a single flow into a
    sequence for one-off /predict calls.
    """

    def __init__(
        self,
        input_dim: int = 20,
        seq_len: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 6,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        positions = torch.arange(x.size(1), device=x.device)
        x = self.input_proj(x) + self.pos_encoding(positions)
        out = self.transformer(x)
        return self.classifier(out.mean(dim=1))
