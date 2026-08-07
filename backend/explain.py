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
                nsamples: int = 150, raw_values: np.ndarray | None = None) -> list[dict]:
        """Top_k features driving P(class_idx) for one scaled flow.

        raw_values, if given, are the unscaled feature readings (same order
        as feature_names) attached to each row so the UI can show "Flow
        Duration was 3006us" next to the contribution, not just a bar.
        """
        explainer = shap.KernelExplainer(
            self._predict_class_prob(class_idx), self.background
        )
        values = explainer.shap_values(
            flow_scaled.reshape(1, -1), nsamples=nsamples, silent=True
        )[0]
        indexed = sorted(
            enumerate(values), key=lambda p: abs(p[1]), reverse=True
        )[:top_k]
        return [
            {
                "feature": feature_names[i],
                "contribution": round(float(val), 4),
                "value": round(float(raw_values[i]), 4) if raw_values is not None else 0.0,
            }
            for i, val in indexed
        ]

    def global_importance(self, flows_scaled: np.ndarray, class_indices: np.ndarray,
                           feature_names: list[str], nsamples: int = 100) -> dict:
        """Mean absolute SHAP contribution per feature, averaged over a
        sample of flows, each explained toward its own predicted class.
        This is the aggregate view: which features matter across the whole
        test set, not just for one alert.
        """
        totals = np.zeros(len(feature_names))
        for flow, cls in zip(flows_scaled, class_indices):
            explainer = shap.KernelExplainer(
                self._predict_class_prob(int(cls)), self.background
            )
            values = explainer.shap_values(
                flow.reshape(1, -1), nsamples=nsamples, silent=True
            )[0]
            totals += np.abs(values)
        mean_abs = totals / max(len(flows_scaled), 1)
        order = np.argsort(mean_abs)[::-1]
        return {
            "feature_names": [feature_names[i] for i in order],
            "mean_abs_shap": [round(float(mean_abs[i]), 5) for i in order],
        }
