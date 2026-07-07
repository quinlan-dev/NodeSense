# NodeSense

**Explainable AI for Network Intrusion Detection**

NodeSense is a network anomaly detection system that uses a transformer based sequence model to identify malicious traffic patterns, including novel attacks that signature based tools like Snort cannot catch. Every detection comes with a plain language explanation of which network flow features drove the decision, so analysts can understand and act on alerts instead of treating the model as a black box.

Built as a graduate independent study project at UC Santa Cruz.

## Why NodeSense

Traditional intrusion detection systems match traffic against databases of known attack signatures. This works well for known threats but fails completely against zero day exploits. Machine learning based detection can catch novel attacks by learning what normal traffic looks like, but most ML systems cannot explain their decisions, which makes security teams reluctant to trust them.

NodeSense addresses both problems. A transformer model trained on the CICIDS-2018 dataset detects anomalies in network flow sequences, and a SHAP based explanation layer attaches feature level reasoning to every alert.

## Features

- Transformer sequence classifier trained on 80 network flow features
- Random forest and autoencoder baselines for benchmarking
- SHAP explanations attached to every prediction
- FastAPI inference server with ONNX runtime for fast predictions
- React dashboard with a live alert feed and feature contribution charts
- Fully reproducible training pipeline

## Architecture

```
PCAP capture -> Flow features -> Transformer model -> Prediction + SHAP explanation
                                                              |
                                              FastAPI  ->  React dashboard
```

## Project Structure

```
nodesense/
├── backend/
│   ├── app.py              FastAPI inference server
│   ├── models.py           Model architectures (RF, autoencoder, transformer)
│   ├── train.py            Training pipeline
│   ├── explain.py          SHAP explanation module
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/                React dashboard
│   ├── package.json
│   └── vite.config.js
├── notebooks/              EDA and experiments
└── docs/
    └── research_log.md     Progress log
```

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

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
| GET | `/` | Health check |
| POST | `/predict` | Classify a network flow, optionally with SHAP explanation |
| GET | `/demo/stream` | Simulated alert stream for the dashboard demo |

### Example Request

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 1.2, ...], "explain": true}'
```

### Example Response

```json
{
  "anomaly": true,
  "confidence": 0.94,
  "explanation": [
    {"feature": "Flow Bytes/s", "contribution": 0.43},
    {"feature": "Fwd IAT Mean", "contribution": -0.31}
  ]
}
```

## Dataset

NodeSense is trained on the [CICIDS-2018 dataset](https://www.unb.ca/cic/datasets/ids-2018.html) from the Canadian Institute for Cybersecurity. The dataset is not included in this repository due to size. Download it separately and place the CSV files in a `data/` directory at the project root.

## Deployment

The backend is designed to deploy on Hugging Face Spaces using the included Dockerfile. The frontend builds to static files and deploys to GitHub Pages, Vercel, or Netlify.

```bash
# Frontend deploy to GitHub Pages
cd frontend
npm run deploy
```

## Tech Stack

PyTorch, scikit-learn, SHAP, FastAPI, ONNX Runtime, React, Vite, Recharts, Docker

## License

MIT

## Author

Quinlan Hoang, M.S. Computer Science, UC Santa Cruz
