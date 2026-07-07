"""
NodeSense explanation module.

Produces per-feature SHAP attributions for a single flow using
KernelSHAP over the exported ONNX model, so the server never needs
PyTorch at inference time.

The model consumes sequences, but explanations are computed at the flow
level: the flow under inspection is tiled into a full sequence, and
KernelSHAP perturbs its 20 features against a k-means summary of the
training background. Positive contributions push toward the target
attack class, negative toward benign.
"""

import numpy as np
import shap


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class FlowExplainer:
    def __init__(self, session, background: np.ndarray, seq_len: int,
                 n_background: int = 10):
        """
        session: onnxruntime InferenceSession, input (batch, seq_len, n_feat)
        background: (n, n_feat) scaled training flows; summarized with
            k-means because KernelSHAP cost is linear in background size.
        """
        self.session = session
        self.seq_len = seq_len
        self.background = shap.kmeans(background, n_background)

    def _predict_class_prob(self, class_idx: int):
        def f(X: np.ndarray) -> np.ndarray:
            seqs = np.repeat(
                X.astype(np.float32)[:, None, :], self.seq_len, axis=1
            )
            logits = self.session.run(None, {"input": seqs})[0]
            return _softmax(logits)[:, class_idx]
        return f

    def explain(self, flow_scaled: np.ndarray, class_idx: int,
                feature_names: list[str], top_k: int = 8,
                nsamples: int = 150) -> list[dict]:
        """Top_k features driving P(class_idx) for one scaled flow."""
        explainer = shap.KernelExplainer(
            self._predict_class_prob(class_idx), self.background
        )
        values = explainer.shap_values(
            flow_scaled.reshape(1, -1), nsamples=nsamples, silent=True
        )[0]
        pairs = sorted(
            zip(feature_names, values),
            key=lambda p: abs(p[1]),
            reverse=True,
        )[:top_k]
        return [
            {"feature": name, "contribution": round(float(val), 4)}
            for name, val in pairs
        ]
