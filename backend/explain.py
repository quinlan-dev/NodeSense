"""
NodeSense explanation module.

Wraps a trained model with SHAP to produce feature level explanations
for every prediction. Build the explainer once at server startup and
reuse it, since constructing it is expensive.
"""

import numpy as np
import shap
import torch


def build_explainer(model, background_data: np.ndarray, n_background: int = 200):
    """Build a SHAP DeepExplainer for a PyTorch model.

    background_data should be a sample of training data. Larger backgrounds
    give more stable values but slow computation. 100 to 200 rows is a
    reasonable tradeoff for real time explanation.
    """
    background = torch.tensor(
        background_data[:n_background], dtype=torch.float32
    )
    return shap.DeepExplainer(model, background)


def explain_prediction(explainer, x: np.ndarray, feature_names: list[str], top_k: int = 10):
    """Return the top_k features driving a single prediction.

    Positive contributions push toward the anomaly class,
    negative contributions push toward benign.
    """
    x_tensor = torch.tensor(x, dtype=torch.float32)
    if x_tensor.dim() == 1:
        x_tensor = x_tensor.unsqueeze(0)

    shap_values = explainer.shap_values(x_tensor)
    # Index 1 holds contributions toward the anomaly class
    values = shap_values[1][0]

    pairs = sorted(
        zip(feature_names, values),
        key=lambda p: abs(p[1]),
        reverse=True,
    )[:top_k]

    return [
        {"feature": name, "contribution": round(float(val), 4)}
        for name, val in pairs
    ]
