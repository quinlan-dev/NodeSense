const FEATURES = [
  {
    ico: '🧠',
    title: 'Transformer sequence model',
    body: 'Flows are classified in sequences of 16, so temporal attack patterns like scan sweeps and C2 beaconing — invisible to single-flow classifiers — become learnable signal.',
  },
  {
    ico: '🔍',
    title: 'Every alert explained',
    body: 'KernelSHAP runs against the served model for each alert, attributing the decision to specific flow features. Analysts see why traffic was flagged, not just that it was.',
  },
  {
    ico: '🎯',
    title: 'Six traffic classes',
    body: 'Beyond benign vs. malicious: DDoS, port scans, brute force, botnet beaconing, and infiltration are distinguished, each with its own flow-level signature.',
  },
  {
    ico: '⚡',
    title: 'Fast ONNX inference',
    body: 'The trained transformer exports to a 335KB ONNX graph. Predictions run in milliseconds on CPU and full SHAP explanations complete in ~120ms.',
  },
  {
    ico: '📊',
    title: 'Benchmarked baselines',
    body: 'A random forest and an unsupervised autoencoder train in the same pipeline, so the transformer is always compared against credible baselines on identical splits.',
  },
  {
    ico: '🔬',
    title: 'Reproducible research',
    body: 'A synthetic CICIDS-style generator makes the full pipeline runnable end to end without the 70GB dataset, and real CICIDS-2018 CSVs drop in with one flag.',
  },
]

const PIPELINE = [
  {
    title: 'Capture flows',
    body: 'Network traffic is summarized into flows — 20 statistical features covering duration, packet counts, byte rates, timing, and TCP flags.',
  },
  {
    title: 'Model sequences',
    body: 'A compact transformer reads 16 consecutive flows and classifies the session, learning the temporal shape of each attack.',
  },
  {
    title: 'Explain decisions',
    body: 'SHAP attributes each detection to the features that drove it: a port scan flagged for its microsecond duration and lone SYN packet.',
  },
  {
    title: 'Alert analysts',
    body: 'The dashboard streams model-classified alerts live; clicking one renders its feature contributions as an interactive chart.',
  },
]

function Landing({ user }) {
  return (
    <>
      <div className="hero">
        <span className="eyebrow">Explainable AI · Network Security</span>
        <h1>
          Catch novel attacks.<br />
          <span className="grad">Understand every alert.</span>
        </h1>
        <p className="sub">
          NodeSense detects network intrusions that signature-based tools miss,
          and explains each detection in terms of the traffic features that
          drove it — so security teams can trust the model instead of guessing.
        </p>
        <div className="cta">
          <a className="btn primary" href={user ? '#/dashboard' : '#/login'}>
            {user ? 'Open dashboard' : 'Try the live demo'}
          </a>
          <a className="btn" href="#/docs">Read the docs</a>
        </div>

        <div className="statstrip">
          <div className="stat-tile"><div className="val">20</div><div className="lbl">flow features per prediction</div></div>
          <div className="stat-tile"><div className="val">6</div><div className="lbl">traffic classes distinguished</div></div>
          <div className="stat-tile"><div className="val">~120ms</div><div className="lbl">per SHAP explanation (CPU)</div></div>
          <div className="stat-tile"><div className="val">335KB</div><div className="lbl">deployed ONNX model</div></div>
        </div>
      </div>

      <main className="content">
        <section className="section">
          <h2>Why signature matching isn't enough</h2>
          <p className="lede">
            Traditional intrusion detection matches traffic against databases of
            known attack signatures — effective for known threats, blind to
            zero-days. Machine learning catches novel attacks by learning what
            normal looks like, but most ML detectors can't explain themselves,
            so analysts don't trust them. NodeSense is built to solve both
            problems at once.
          </p>
          <div className="feature-grid">
            {FEATURES.map((f) => (
              <div className="feature" key={f.title}>
                <div className="ico">{f.ico}</div>
                <h3>{f.title}</h3>
                <p>{f.body}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="section">
          <h2>How it works</h2>
          <p className="lede">From raw traffic to an explained alert in four stages.</p>
          <div className="pipeline">
            {PIPELINE.map((s, i) => (
              <div className="pipe-step" key={s.title}>
                <span className="num">{i + 1}</span>
                <h3>{s.title}</h3>
                <p>{s.body}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="section">
          <h2>Built with</h2>
          <p className="lede">
            An end-to-end ML systems project: research code, a production-style
            serving stack, and a live product surface.
          </p>
          <div className="stack-badges">
            {['PyTorch', 'scikit-learn', 'SHAP', 'FastAPI', 'ONNX Runtime',
              'React', 'Vite', 'Recharts', 'Docker', 'Hugging Face Spaces'].map((t) => (
              <span key={t}>{t}</span>
            ))}
          </div>
        </section>
      </main>
    </>
  )
}

export default Landing
