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
import logging
import math
import os
import random
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from data import CLASS_NAMES, FEATURE_NAMES, SEQ_LEN, generate_sessions

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nodesense")
# shap logs its internal linear algebra (kernel weights, phi arrays) at
# INFO, which would otherwise drown out the one-line predict log below.
logging.getLogger("shap").setLevel(logging.WARNING)

state = {"session": None, "explainer": None, "pre": None}

# SHAP is the expensive path (~120ms of CPU per call); bound how many can
# run at once so a burst of explain requests can't starve the host.
_explain_slots = threading.Semaphore(2)

# In-memory sliding-window rate limit, keyed by client IP. Deliberately not
# a separate dependency (slowapi etc.) for a single-process demo API; a
# multi-worker or multi-instance deployment would need a shared backend
# (Redis) instead since this state does not cross processes.
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_S = int(os.environ.get("RATE_LIMIT_WINDOW_S", "60"))
_rate_state: dict[str, deque] = defaultdict(deque)
_rate_lock = threading.Lock()

MAX_BODY_BYTES = 8 * 1024  # a 20-float request is ~a few hundred bytes


def check_rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        q = _rate_state[ip]
        while q and now - q[0] > RATE_LIMIT_WINDOW_S:
            q.popleft()
        if len(q) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(429, "Rate limit exceeded, please slow down")
        q.append(now)


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

# Only the deployed dashboard and local dev servers may call this API by
# default. Override with ALLOWED_ORIGINS="https://a.example,https://b.example"
# (comma separated, no spaces) to lock this down further for another deployment.
_DEFAULT_ORIGINS = [
    "https://quinlan-dev.github.io",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
]
_origins_env = os.environ.get("ALLOWED_ORIGINS")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",")] if _origins_env else _DEFAULT_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_headers_and_body_limit(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "Request body too large"}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def _sanitize_non_finite(obj):
    """Recursively replace NaN/Infinity floats with their string form.

    FastAPI's default 422 handler echoes the rejected input back in the
    error body, and Starlette's JSONResponse renders with allow_nan=False
    — so without this, submitting NaN/Infinity (exactly the input our own
    validator rejects) crashes the error handler itself with a 500 instead
    of cleanly returning 422.
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = _sanitize_non_finite(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": errors})


class FlowRequest(BaseModel):
    features: list[float]
    explain: bool = False

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: list[float]) -> list[float]:
        if len(v) != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected exactly {len(FEATURE_NAMES)} features, got {len(v)}"
            )
        for x in v:
            if math.isnan(x) or math.isinf(x):
                raise ValueError("Feature values must be finite (no NaN or inf)")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [
                        3006.6, 1, 0, 59.6, 25.4, 113.4, 39.8, 8453.0, 332.6,
                        3006.6, 384.6, 3578.5, 2439.0, 0, 1, 0, 0, 0, 25.4, 228.7,
                    ],
                    "explain": True,
                }
            ]
        }
    }


class ExplanationItem(BaseModel):
    feature: str
    contribution: float
    value: float


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
    raw = np.array(features)
    flow = scale_flow(raw)
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
        if not _explain_slots.acquire(timeout=10):
            raise HTTPException(503, "Explanation workers busy, retry shortly")
        try:
            result["explanation"] = state["explainer"].explain(
                flow, target, FEATURE_NAMES, raw_values=raw
            )
        finally:
            _explain_slots.release()
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
        sampled_idx = random.sample(range(len(FEATURE_NAMES)), 6)
        result["explanation"] = sorted(
            (
                {
                    "feature": FEATURE_NAMES[i],
                    "contribution": round(random.uniform(-0.5, 0.5), 3),
                    "value": round(features[i], 4) if i < len(features) else 0.0,
                }
                for i in sampled_idx
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


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(check_rate_limit)])
def predict(req: FlowRequest):
    t0 = time.perf_counter()
    if state["session"] is None:
        result = demo_prediction(req.features, req.explain)
    else:
        result = real_prediction(req.features, req.explain)
    latency_ms = (time.perf_counter() - t0) * 1000
    # Log the decision and timing, never the raw feature vector.
    logger.info(
        "predict class=%s anomaly=%s confidence=%.3f explain=%s latency_ms=%.1f mode=%s",
        result["attack_type"], result["anomaly"], result["confidence"],
        req.explain, latency_ms, "demo" if state["session"] is None else "live",
    )
    return result


# Cap concurrent WebSocket clients so a burst of connections cannot exhaust
# the single free-tier instance this normally runs on.
MAX_WS_CONNECTIONS = int(os.environ.get("MAX_WS_CONNECTIONS", "50"))
_ws_connections = 0
_ws_lock = threading.Lock()


@app.get("/demo/stream", dependencies=[Depends(check_rate_limit)])
def demo_stream(n: int = 5):
    """Polling fallback for the dashboard when WebSockets are unavailable."""
    return {"alerts": [next_alert() for _ in range(min(max(n, 0), 20))]}


@app.websocket("/ws/alerts")
async def alert_stream(websocket: WebSocket):
    """Streams model-classified alerts to the dashboard."""
    global _ws_connections
    with _ws_lock:
        if _ws_connections >= MAX_WS_CONNECTIONS:
            await websocket.close(code=1013)  # 1013 = "try again later"
            return
        _ws_connections += 1
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(random.uniform(1.5, 4.0))
            alert = await asyncio.to_thread(next_alert)
            await websocket.send_text(json.dumps(alert))
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            _ws_connections -= 1


if __name__ == "__main__":
    import uvicorn
    # Port 7860 is required by Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)
