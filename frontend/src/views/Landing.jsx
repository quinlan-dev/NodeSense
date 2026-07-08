import { useEffect, useRef, useState } from 'react'
import { attackColor } from '../lib'

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

const STATS = [
  { to: 20, suffix: '', label: 'flow features per prediction' },
  { to: 6, suffix: '', label: 'traffic classes distinguished' },
  { to: 120, prefix: '~', suffix: 'ms', label: 'per SHAP explanation (CPU)' },
  { to: 335, suffix: 'KB', label: 'deployed ONNX model' },
]

const PREVIEW_ALERTS = [
  { type: 'Port Scan', ip: '10.0.202.156', conf: '99%' },
  { type: 'DDoS', ip: '10.0.44.218', conf: '99%' },
  { type: 'Botnet', ip: '10.0.117.42', conf: '97%' },
]

function CountUp({ to, prefix = '', suffix = '', duration = 1200 }) {
  const [val, setVal] = useState(0)
  const ref = useRef(null)
  useEffect(() => {
    let raf
    let started = false
    const start = () => {
      if (started) return
      started = true
      const t0 = performance.now()
      const tick = (t) => {
        const p = Math.min((t - t0) / duration, 1)
        setVal(Math.round(to * (1 - Math.pow(1 - p, 3)))) // ease-out cubic
        if (p < 1) raf = requestAnimationFrame(tick)
      }
      raf = requestAnimationFrame(tick)
    }
    const obs = new IntersectionObserver(
      (entries) => entries[0].isIntersecting && start(),
      { threshold: 0.4 }
    )
    if (ref.current) obs.observe(ref.current)
    return () => { obs.disconnect(); cancelAnimationFrame(raf) }
  }, [to, duration])
  return <span ref={ref}>{prefix}{val}{suffix}</span>
}

function HeroPreview() {
  // theme for attackColor: read current doc theme so preview matches
  const theme = document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'
  return (
    <div className="hero-preview" aria-hidden="true">
      <div className="hp-head">
        <span className="hp-dots"><i /><i /><i /></span>
        live detection · nodesense
        <span className="status live" style={{ marginLeft: 'auto' }}>live</span>
      </div>
      <div className="hp-body">
        {PREVIEW_ALERTS.map((a) => (
          <div className="hp-row" key={a.ip} style={{ borderLeftColor: attackColor(a.type, theme) }}>
            <span className="t">{a.type}</span>
            <span className="ip">{a.ip}</span>
            <span className="c">{a.conf}</span>
          </div>
        ))}
      </div>
      <div className="hp-shap">
        <div className="lbl">Why was this flagged?</div>
        {[
          ['Flow Duration', 78, true],
          ['Total Fwd Packets', 56, true],
          ['Fwd Packet Length Max', 22, false],
        ].map(([name, pct, attack]) => (
          <div className="hp-bar" key={name}>
            <span className="name">{name}</span>
            <span className="track">
              <span
                className="fill"
                style={{
                  width: `${pct}%`,
                  background: attack ? 'var(--danger)' : 'var(--accent-strong)',
                }}
              />
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

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

        <HeroPreview />

        <div className="statstrip">
          {STATS.map((s) => (
            <div className="stat-tile" key={s.label}>
              <div className="val"><CountUp to={s.to} prefix={s.prefix} suffix={s.suffix} /></div>
              <div className="lbl">{s.label}</div>
            </div>
          ))}
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

        <section className="cta-band">
          <h2>See it detect in real time</h2>
          <p>Open the live dashboard, watch the model classify traffic, and click any alert for its explanation.</p>
          <a className="btn primary" href={user ? '#/dashboard' : '#/login'}>
            {user ? 'Open dashboard' : 'Launch the demo'}
          </a>
        </section>
      </main>
    </>
  )
}

export default Landing
