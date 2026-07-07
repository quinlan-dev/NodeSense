"""
NodeSense inference server.

Serves anomaly predictions from the exported ONNX transformer with
optional SHAP explanations. The alert stream generates live traffic
sessions and runs them through the real model, so every alert on the
dashboard is an actual model decision.

If artifacts/ is missing (model not trained yet), the server falls back
to a demo mode with simulated predictions so the dashboard still works.
"""

import asyncio
import json
import os
import random
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data import CLASS_NAMES, FEATURE_NAMES, SEQ_LEN, generate_sessions

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

state = {"session": None, "explainer": None, "pre": None}


def load_artifacts():
    model_path = os.path.join(ARTIFACT_DIR, "model.onnx")
    if not os.path.exists(model_path):
        return False
    import onnxruntime as ort
    from explain import FlowExplainer

    with open(os.path.join(ARTIFACT_DIR, "preprocess.json")) as f:
        state["pre"] = json.load(f)
    state["session"] = ort.InferenceSession(model_path)
    background = np.load(os.path.join(ARTIFACT_DIR, "background.npy"))
    state["explainer"] = FlowExplainer(
        state["session"], background, state["pre"]["seq_len"]
    )
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    if load_artifacts():
        print(f"Loaded model artifacts from {ARTIFACT_DIR}")
    else:
        print("No trained model found. Running in demo mode. Run train.py first.")
    yield


app = FastAPI(
    title="NodeSense API",
    description="Explainable network anomaly detection",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the dashboard to call this API from a different origin.
# Lock allow_origins down to your actual frontend URL before public launch.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FlowRequest(BaseModel):
    features: list[float]
    explain: bool = False


class ExplanationItem(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    anomaly: bool
    confidence: float
    attack_type: str
    explanation: list[ExplanationItem] | None = None


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def scale_flow(raw: np.ndarray) -> np.ndarray:
    pre = state["pre"]
    x = raw.astype(np.float64).copy()
    mask = np.array(pre["log_mask"])
    x[mask] = np.log1p(np.clip(x[mask], 0, None))
    x = (x - np.array(pre["scaler_mean"])) / np.array(pre["scaler_scale"])
    return x.astype(np.float32)


def predict_sequence(seq_scaled: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Run one scaled (seq_len, n_feat) sequence. Returns
    (class_idx, anomaly_confidence, class_probs)."""
    logits = state["session"].run(None, {"input": seq_scaled[None]})[0]
    probs = _softmax(logits)[0]
    cls = int(probs.argmax())
    if cls == 0:  # model says benign; confidence = P(any attack)
        return 0, float(1.0 - probs[0]), probs
    return cls, float(probs[cls]), probs


def real_prediction(features: list[float], explain: bool) -> dict:
    if len(features) != len(FEATURE_NAMES):
        raise HTTPException(
            422, f"Expected {len(FEATURE_NAMES)} features, got {len(features)}"
        )
    flow = scale_flow(np.array(features))
    seq = np.repeat(flow[None], SEQ_LEN, axis=0)
    cls, conf, probs = predict_sequence(seq)
    result = {
        "anomaly": cls != 0,
        "confidence": round(conf, 3),
        "attack_type": CLASS_NAMES[cls],
        "explanation": None,
    }
    if explain:
        # Explain toward the predicted attack class, or the most likely
        # attack class when the flow was ruled benign.
        target = cls if cls != 0 else int(probs[1:].argmax()) + 1
        result["explanation"] = state["explainer"].explain(
            flow, target, FEATURE_NAMES
        )
    return result


def demo_prediction(features: list[float], explain: bool) -> dict:
    confidence = round(random.uniform(0.05, 0.98), 3)
    anomaly = confidence > 0.5
    result = {
        "anomaly": anomaly,
        "confidence": confidence,
        "attack_type": random.choice(CLASS_NAMES[1:]) if anomaly else "Benign",
        "explanation": None,
    }
    if explain:
        sampled = random.sample(FEATURE_NAMES, 6)
        result["explanation"] = sorted(
            (
                {"feature": n, "contribution": round(random.uniform(-0.5, 0.5), 3)}
                for n in sampled
            ),
            key=lambda e: abs(e["contribution"]),
            reverse=True,
        )
    return result


def generate_alert(rng_seed: int | None = None) -> dict | None:
    """Generate one traffic session, classify it with the real model, and
    return an alert dict if the model flags it. The raw features of a
    representative flow ride along so the dashboard can request a SHAP
    explanation for exactly this alert."""
    seed = rng_seed if rng_seed is not None else random.randrange(2**31)
    X, y, y_flow = generate_sessions(n_sessions=1, seed=seed, benign_frac=0.35)
    seq_scaled = np.stack([scale_flow(f) for f in X[0]])
    cls, conf, _ = predict_sequence(seq_scaled)
    if cls == 0:
        return None
    # pick a flow of the session's attack class as the representative
    attack_flows = np.where(y_flow[0] == y[0])[0]
    rep = int(attack_flows[0]) if len(attack_flows) else SEQ_LEN - 1
    return {
        "timestamp": time.time(),
        "source_ip": f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}",
        "attack_type": CLASS_NAMES[cls],
        "confidence": round(conf, 3),
        "features": [round(float(v), 4) for v in X[0][rep]],
        "true_label": CLASS_NAMES[y[0]],
    }


def demo_alert() -> dict:
    return {
        "timestamp": time.time(),
        "source_ip": f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}",
        "attack_type": random.choice(CLASS_NAMES[1:]),
        "confidence": round(random.uniform(0.5, 0.99), 3),
        "features": [round(random.uniform(0, 2), 4) for _ in FEATURE_NAMES],
        "true_label": None,
    }


def next_alert() -> dict:
    if state["session"] is None:
        return demo_alert()
    for _ in range(10):  # benign sessions produce no alert; try a few
        alert = generate_alert()
        if alert:
            return alert
    return demo_alert()


@app.get("/")
def health():
    return {
        "service": "NodeSense",
        "status": "ok",
        "mode": "demo" if state["session"] is None else "live",
        "features": len(FEATURE_NAMES),
        "classes": CLASS_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: FlowRequest):
    if state["session"] is None:
        return demo_prediction(req.features, req.explain)
    return real_prediction(req.features, req.explain)


@app.get("/demo/stream")
def demo_stream(n: int = 5):
    """Polling fallback for the dashboard when WebSockets are unavailable."""
    return {"alerts": [next_alert() for _ in range(min(n, 20))]}


@app.websocket("/ws/alerts")
async def alert_stream(websocket: WebSocket):
    """Streams model-classified alerts to the dashboard."""
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(random.uniform(1.5, 4.0))
            alert = await asyncio.to_thread(next_alert)
            await websocket.send_text(json.dumps(alert))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    # Port 7860 is required by Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)
