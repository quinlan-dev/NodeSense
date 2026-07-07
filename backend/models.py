"""
NodeSense model architectures.

Three candidates trained and benchmarked against each other:
1. Random forest (supervised baseline)
2. Autoencoder (unsupervised baseline, detects via reconstruction error)
3. Transformer (primary model, classifies flow sequences)
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

    def __init__(self, input_dim: int = 78):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class NetworkTransformer(nn.Module):
    """Classifies sequences of network flows. Each flow is one timestep,
    so the model can learn temporal attack patterns like scan sweeps
    that single flow classifiers miss."""

    def __init__(
        self,
        input_dim: int = 78,
        seq_len: int = 16,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
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
        return self.classifier(out[:, -1, :])
