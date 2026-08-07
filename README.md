# NodeSense

**Explainable AI for Network Intrusion Detection**

NodeSense is a network anomaly detection system that uses a transformer based sequence model to identify malicious traffic patterns, including novel attacks that signature based tools like Snort cannot catch. Every detection comes with a plain language explanation of which network flow features drove the decision, so analysts can understand and act on alerts instead of treating the model as a black box.

Built as a graduate independent study project at UC Santa Cruz.

## Why NodeSense

Traditional intrusion detection systems match traffic against databases of known attack signatures. This works well for known threats but fails completely against zero day exploits. Machine learning based detection can catch novel attacks by learning what normal traffic looks like, but most ML systems cannot explain their decisions, which makes security teams reluctant to trust them.

NodeSense addresses both problems. A transformer model trained on the CICIDS-2018 dataset detects anomalies in network flow sequences, and a SHAP based explanation layer attaches feature level reasoning to every alert.

## Features

- Transformer sequence classifier over 20 network flow features, 6 classes
  (Benign, DDoS, Port Scan, Brute Force, Botnet, Infiltration)
- Random forest and autoencoder baselines benchmarked in the same pipeline
- Real KernelSHAP explanations computed against the served ONNX model
- FastAPI inference server with ONNX Runtime (no PyTorch at serve time)
- React dashboard with a live model-classified alert feed and per-alert
  feature contribution charts (WebSocket, with REST polling fallback)
- Fully reproducible training pipeline with a synthetic CICIDS-style data
  generator, so everything runs end to end without the 70GB dataset

## Architecture

```
data.py  synthetic flow sessions (or real CICIDS-2018 CSVs)
   |
train.py  RF + autoencoder baselines, transformer training
   |
artifacts/  model.onnx + preprocess.json + background.npy  (committed)
   |
app.py  FastAPI: /predict (+SHAP via explain.py), /ws/alerts, /demo/stream
   |
frontend/  React dashboard: live alert feed + SHAP contribution chart
```

## Project Structure

```
nodesense/
├── backend/
│   ├── app.py                  FastAPI inference server
│   ├── data.py                 Synthetic flow generator + CICIDS loader
│   ├── models.py               Model architectures (RF, autoencoder, transformer)
│   ├── train.py                Training pipeline + ONNX export
│   ├── explain.py              KernelSHAP over the ONNX model
│   ├── evaluate.py             Held-out metrics, confusion matrix, calibration
│   ├── zero_day_eval.py        Leave-one-attack-out generalization study
│   ├── benchmark.py            Inference latency percentiles
│   ├── global_importance.py    Aggregate SHAP feature importance
│   ├── tests/                  pytest suite (API + preprocessing)
│   ├── artifacts/              Exported model + preprocessing state + eval/
│   ├── requirements.txt        Serving dependencies (used by Dockerfile)
│   ├── requirements-train.txt  Training extras (PyTorch, onnxscript)
│   ├── requirements-dev.txt    Test/eval extras (pytest, httpx, matplotlib)
│   └── Dockerfile
├── frontend/
│   ├── src/                    React dashboard
│   ├── package.json
│   └── vite.config.js
├── notebooks/                  EDA and experiments
├── .github/workflows/          CI (tests + build) and optional Pages deploy
└── docs/
    ├── research_log.md         Progress log
    └── MODEL_CARD.md           Intended use, metrics, limitations, ethics
```

## Quick Start

### Backend

```bash
cd backend
py -3.12 -m venv venv           # PyTorch/SHAP need Python <= 3.12
venv\Scripts\activate           # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt -r requirements-train.txt
python train.py                 # trains everything, exports artifacts/
uvicorn app:app --reload --port 7860
```

A trained model is already committed in `backend/artifacts/`, so the server
runs in live mode even if you skip `train.py`.

The API is now running at http://localhost:7860 with interactive docs at http://localhost:7860/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard is now running at http://localhost:5173

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check, reports live/demo mode |
| POST | `/predict` | Classify a network flow (20 raw features), optionally with SHAP explanation |
| WS | `/ws/alerts` | Live stream of model-classified alerts |
| GET | `/demo/stream` | REST polling fallback for the alert stream |

### Example Request

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [3006.6, 1, 0, 59.6, 25.4, 113.4, 39.8, 8453.0, 332.6, 3006.6, 384.6, 3578.5, 2439.0, 0, 1, 0, 0, 0, 25.4, 228.7], "explain": true}'
```

### Example Response

```json
{
  "anomaly": true,
  "confidence": 0.99,
  "attack_type": "Port Scan",
  "explanation": [
    {"feature": "Flow Duration", "contribution": 0.362},
    {"feature": "Total Fwd Packets", "contribution": 0.258}
  ]
}
```

## Dataset

The committed model is trained on synthetic flow sessions from
`backend/data.py`, whose class distributions are modeled on how each attack
looks at the flow level. To train on the real
[CICIDS-2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html) from the
Canadian Institute for Cybersecurity, download the CSVs into `data/` and run:

```bash
python train.py --data "../data/*.csv"
```

`data.py` maps the CICIDS column names onto the same 20-feature vector, so a
model trained on either source serves identically.

## Deployment

The backend is designed to deploy on Hugging Face Spaces using the included Dockerfile. The frontend builds to static files and deploys to GitHub Pages, Vercel, or Netlify.

```bash
# Frontend deploy to GitHub Pages
cd frontend
npm run deploy
```

### Backend environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | the GitHub Pages dashboard + local dev ports | Comma-separated list of origins allowed to call the API (CORS). Set this to lock the API to a different frontend origin. |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_S` | `30` / `60` | Per-IP request cap on `/predict` and `/demo/stream` (sliding window, in-memory — see the model card's limitations section for the multi-worker caveat). |
| `MAX_WS_CONNECTIONS` | `50` | Caps concurrent `/ws/alerts` clients per instance. |
| `LOG_LEVEL` | `INFO` | Python logging level for the `nodesense` logger. |

## Model card and evaluation

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for intended use, training
data caveats, and the full evaluation suite's results: held-out
precision/recall/F1/AUC, confidence calibration, a leave-one-attack-out
zero-day generalization study, and inference latency. Regenerate all of it
against the current model with:

```bash
cd backend
python evaluate.py            # metrics, confusion matrix, calibration
python zero_day_eval.py        # leave-one-attack-out (retrains 5 models)
python benchmark.py            # latency percentiles
python global_importance.py    # aggregate SHAP feature importance
```

These need `requirements-dev.txt` (evaluate.py, global_importance.py) and
additionally `requirements-train.txt` (zero_day_eval.py, which retrains
models).

## Tech Stack

PyTorch, scikit-learn, SHAP, FastAPI, ONNX Runtime, React, Vite, Recharts, Docker

## License

MIT

## Author

Quinlan Hoang, M.S. Computer Science, UC Santa Cruz
