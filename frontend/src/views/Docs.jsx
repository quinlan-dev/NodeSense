const FEATURES_20 = [
  'Flow Duration', 'Total Fwd Packets', 'Total Bwd Packets',
  'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Bwd Packet Length Max',
  'Bwd Packet Length Mean', 'Flow Bytes/s', 'Flow Packets/s',
  'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
  'Fwd PSH Flags', 'SYN Flag Count', 'ACK Flag Count', 'URG Flag Count',
  'Down/Up Ratio', 'Average Packet Size', 'Idle Mean',
]

function Docs() {
  return (
    <>
      <div className="page-head">
        <h1>Documentation</h1>
        <p>
          Everything needed to run NodeSense locally, call the API, and retrain
          the model — condensed from the repository README.
        </p>
      </div>

      <section className="card" style={{ marginBottom: 20 }}>
        <h2>Quick start</h2>
        <h3>Backend (Python 3.12)</h3>
        <pre className="codeblock">{`cd backend
py -3.12 -m venv venv            # PyTorch/SHAP need Python <= 3.12
venv\\Scripts\\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt -r requirements-train.txt
python train.py                  # optional — a trained model ships in artifacts/
uvicorn app:app --reload --port 7860`}</pre>
        <p style={{ fontSize: 14, color: 'var(--muted)', lineHeight: 1.6 }}>
          The API serves at <code>http://localhost:7860</code> with interactive
          docs at <code>/docs</code>. A trained model is committed in{' '}
          <code>backend/artifacts/</code>, so the server starts in live mode
          even if you skip training.
        </p>
        <h3>Frontend</h3>
        <pre className="codeblock">{`cd frontend
npm install
npm run dev                      # dashboard at http://localhost:5173`}</pre>
      </section>

      <section className="card" style={{ marginBottom: 20 }}>
        <h2>API reference</h2>
        <table className="docs">
          <thead>
            <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td>GET</td><td><code>/</code></td><td>Health check; reports whether a trained model is serving (live) or the fallback simulator (demo)</td></tr>
            <tr><td>POST</td><td><code>/predict</code></td><td>Classify one network flow from its 20 raw features; <code>explain: true</code> adds SHAP attributions</td></tr>
            <tr><td>WS</td><td><code>/ws/alerts</code></td><td>Live stream of model-classified alerts, each carrying its raw feature vector</td></tr>
            <tr><td>GET</td><td><code>/demo/stream</code></td><td>REST polling fallback for the alert stream (<code>?n=5</code>)</td></tr>
          </tbody>
        </table>

        <h3>Example: classify and explain a flow</h3>
        <pre className="codeblock">{`curl -X POST https://quinlan-dev-nodesense.hf.space/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [3006.6, 1, 0, 59.6, 25.4, 113.4, 39.8, 8453.0,
       332.6, 3006.6, 384.6, 3578.5, 2439.0, 0, 1, 0, 0, 0, 25.4, 228.7],
      "explain": true}'`}</pre>
        <pre className="codeblock">{`{
  "anomaly": true,
  "confidence": 0.99,
  "attack_type": "Port Scan",
  "explanation": [
    {"feature": "Flow Duration",     "contribution": 0.362},
    {"feature": "Total Fwd Packets", "contribution": 0.258}
  ]
}`}</pre>
      </section>

      <section className="card" style={{ marginBottom: 20 }}>
        <h2>The 20 flow features</h2>
        <p style={{ fontSize: 14, color: 'var(--muted)', marginBottom: 14, lineHeight: 1.6 }}>
          Each prediction consumes this fixed vector, in order. The names follow
          the CICIDS-2018 conventions, and <code>backend/data.py</code> maps the
          real dataset's columns onto the same vector.
        </p>
        <div className="feature-chips">
          {FEATURES_20.map((f) => <span key={f}>{f}</span>)}
        </div>
      </section>

      <section className="card" style={{ marginBottom: 20 }}>
        <h2>Training on real data</h2>
        <p style={{ fontSize: 14, color: 'var(--muted)', lineHeight: 1.6, marginBottom: 10 }}>
          The committed model is trained on synthetic sessions whose class
          distributions mirror each attack's flow-level behavior. To train on
          the real <a href="https://www.unb.ca/cic/datasets/ids-2018.html" target="_blank" rel="noreferrer">CICIDS-2018
          dataset</a>, download the CSVs into <code>data/</code> and run:
        </p>
        <pre className="codeblock">{`python train.py --data "../data/*.csv"`}</pre>
        <p style={{ fontSize: 14, color: 'var(--muted)', lineHeight: 1.6 }}>
          Training benchmarks a random forest and an autoencoder alongside the
          transformer on identical splits, then exports the transformer plus its
          preprocessing state to <code>artifacts/</code> for serving.
        </p>
      </section>

      <section className="card">
        <h2>Deployment</h2>
        <p style={{ fontSize: 14, color: 'var(--muted)', lineHeight: 1.6 }}>
          The backend deploys to Hugging Face Spaces with the included
          Dockerfile (listens on port 7860; the image installs serving
          dependencies only — no PyTorch). This dashboard builds to static files
          and deploys to GitHub Pages with <code>npm run deploy</code>. Full
          steps live in <code>docs/DEPLOY.md</code> in the{' '}
          <a href="https://github.com/quinlan-dev/NodeSense" target="_blank" rel="noreferrer">repository</a>.
        </p>
      </section>
    </>
  )
}

export default Docs
