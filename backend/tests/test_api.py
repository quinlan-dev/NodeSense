"""Tests for the FastAPI server. Runs against whatever artifacts/ contains —
live mode if a trained model is committed (the normal case), demo mode
otherwise — so these pass either way without requiring training first.
"""

import pytest
from fastapi.testclient import TestClient

from app import app
from data import FEATURE_NAMES

VALID_FEATURES = [
    3006.6, 1, 0, 59.6, 25.4, 113.4, 39.8, 8453.0, 332.6,
    3006.6, 384.6, 3578.5, 2439.0, 0, 1, 0, 0, 0, 25.4, 228.7,
]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["mode"] in ("live", "demo")
    assert body["features"] == len(FEATURE_NAMES)
    assert len(body["classes"]) == 6


def test_predict_valid_input(client):
    res = client.post("/predict", json={"features": VALID_FEATURES, "explain": False})
    assert res.status_code == 200
    body = res.json()
    assert "anomaly" in body
    assert "confidence" in body
    assert "attack_type" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_with_explanation(client):
    res = client.post("/predict", json={"features": VALID_FEATURES, "explain": True})
    assert res.status_code == 200
    body = res.json()
    assert body["explanation"] is not None
    assert len(body["explanation"]) > 0
    for item in body["explanation"]:
        assert set(item.keys()) == {"feature", "contribution", "value"}
        assert item["feature"] in FEATURE_NAMES


def test_predict_wrong_length_rejected(client):
    res = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert res.status_code == 422


def test_predict_nan_rejected(client):
    # httpx's json= helper refuses to serialize NaN/Infinity itself (a
    # client-side guard), so send the raw body to actually exercise the
    # server's own finite-value validator.
    body = '{"features": [' + ",".join(["NaN"] * len(FEATURE_NAMES)) + ']}'
    res = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert res.status_code == 422


def test_predict_inf_rejected(client):
    body = '{"features": [' + ",".join(["Infinity"] * len(FEATURE_NAMES)) + ']}'
    res = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert res.status_code == 422


def test_demo_stream(client):
    res = client.get("/demo/stream?n=3")
    assert res.status_code == 200
    alerts = res.json()["alerts"]
    assert len(alerts) == 3
    for a in alerts:
        assert "attack_type" in a
        assert "features" in a
        assert len(a["features"]) == len(FEATURE_NAMES)
