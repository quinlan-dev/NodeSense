"""
NodeSense inference server.

Serves anomaly predictions with optional SHAP explanations.
Runs a demo mode with simulated traffic when no trained model is present,
so the dashboard works end to end before training is complete.
"""

import os
import random
import asyncio
import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "model.onnx")
DEMO_MODE = not os.path.exists(MODEL_PATH)

# Feature names for the CICIDS-2018 flow features used in explanations.
# Trim or extend this list to match your final feature set after preprocessing.
FEATURE_NAMES = [
    "Flow Duration", "Total Fwd Packets", "Total Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Mean", "Bwd Packet Length Max",
    "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Mean", "Bwd IAT Mean",
    "Fwd PSH Flags", "SYN Flag Count", "ACK Flag Count", "URG Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Idle Mean",
]

sess = None
explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sess
    if not DEMO_MODE:
        import onnxruntime as ort
        sess = ort.InferenceSession(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("No model found. Running in demo mode with simulated predictions.")
    yield


app = FastAPI(
    title="NodeSense API",
    description="Explainable network anomaly detection",
    version="0.1.0",
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
    explanation: list[ExplanationItem] | None = None


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)


def demo_prediction(features: list[float], explain: bool) -> dict:
    """Simulated prediction used before a trained model exists."""
    confidence = round(random.uniform(0.05, 0.98), 3)
    result = {"anomaly": confidence > 0.5, "confidence": confidence, "explanation": None}
    if explain:
        sampled = random.sample(FEATURE_NAMES, 6)
        result["explanation"] = [
            {"feature": name, "contribution": round(random.uniform(-0.5, 0.5), 3)}
            for name in sampled
        ]
        result["explanation"].sort(key=lambda e: abs(e["contribution"]), reverse=True)
    return result


def real_prediction(features: list[float], explain: bool) -> dict:
    x = np.array(features, dtype=np.float32).reshape(1, -1)
    logits = sess.run(None, {"input": x})[0]
    prob = float(softmax(logits)[0][1])
    result = {"anomaly": prob > 0.5, "confidence": round(prob, 3), "explanation": None}
    if explain and explainer is not None:
        # See explain.py for building the SHAP explainer against the trained model
        shap_values = explainer.shap_values(x)
        pairs = sorted(
            zip(FEATURE_NAMES, shap_values[1][0]),
            key=lambda p: abs(p[1]),
            reverse=True,
        )[:10]
        result["explanation"] = [
            {"feature": name, "contribution": round(float(val), 3)}
            for name, val in pairs
        ]
    return result


@app.get("/")
def health():
    return {
        "service": "NodeSense",
        "status": "ok",
        "mode": "demo" if DEMO_MODE else "live",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: FlowRequest):
    if DEMO_MODE:
        return demo_prediction(req.features, req.explain)
    return real_prediction(req.features, req.explain)


@app.websocket("/ws/alerts")
async def alert_stream(websocket: WebSocket):
    """Streams simulated alerts so the dashboard has live data to render."""
    await websocket.accept()
    attack_types = ["DDoS", "Port Scan", "Brute Force", "Botnet", "Infiltration"]
    try:
        while True:
            await asyncio.sleep(random.uniform(1.0, 4.0))
            confidence = round(random.uniform(0.5, 0.99), 3)
            alert = {
                "timestamp": asyncio.get_event_loop().time(),
                "source_ip": f"10.0.{random.randint(0,255)}.{random.randint(1,254)}",
                "attack_type": random.choice(attack_types),
                "confidence": confidence,
                "top_feature": random.choice(FEATURE_NAMES),
            }
            await websocket.send_text(json.dumps(alert))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    # Port 7860 is required by Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)
