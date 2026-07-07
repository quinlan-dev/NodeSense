import { useState, useEffect, useRef, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

// Local dev goes through the Vite proxy (/api) and a direct WebSocket.
// Production points at the Hugging Face Space running backend/.
const API_BASE = import.meta.env.PROD
  ? 'https://quinlan-dev-nodesense.hf.space'
  : '/api'

const WS_BASE = import.meta.env.PROD
  ? 'wss://quinlan-dev-nodesense.hf.space'
  : 'ws://localhost:7860'

function App() {
  const [alerts, setAlerts] = useState([])
  const [selected, setSelected] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [status, setStatus] = useState('connecting')
  const [mode, setMode] = useState(null)
  const pollTimer = useRef(null)

  const pushAlert = useCallback((alert) => {
    alert.id = Date.now() + Math.random()
    setAlerts((prev) => [alert, ...prev].slice(0, 50))
  }, [])

  // Backend health check reports whether a trained model is serving
  useEffect(() => {
    fetch(`${API_BASE}/`)
      .then((r) => r.json())
      .then((d) => setMode(d.mode))
      .catch(() => setMode(null))
  }, [])

  // Live alert stream over WebSocket, with REST polling as a fallback
  useEffect(() => {
    let ws
    let closed = false

    const startPolling = () => {
      if (pollTimer.current) return
      setStatus('polling')
      pollTimer.current = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/demo/stream?n=1`)
          const data = await res.json()
          data.alerts.forEach(pushAlert)
        } catch {
          setStatus('disconnected')
        }
      }, 3000)
    }

    try {
      ws = new WebSocket(`${WS_BASE}/ws/alerts`)
      ws.onopen = () => setStatus('live')
      ws.onmessage = (e) => pushAlert(JSON.parse(e.data))
      ws.onerror = () => { if (!closed) startPolling() }
      ws.onclose = () => { if (!closed) startPolling() }
    } catch {
      startPolling()
    }

    return () => {
      closed = true
      if (ws) ws.close()
      if (pollTimer.current) clearInterval(pollTimer.current)
      pollTimer.current = null
    }
  }, [pushAlert])

  // Fetch a SHAP explanation for the selected alert's actual flow features
  const explainAlert = useCallback(async (alert) => {
    setSelected(alert)
    setExplanation(null)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: alert.features, explain: true }),
      })
      const data = await res.json()
      setExplanation(data.explanation || [])
    } catch {
      setExplanation([])
    }
  }, [])

  const maxAbs = explanation?.length
    ? Math.max(...explanation.map((e) => Math.abs(e.contribution)), 0.1)
    : 0.5

  return (
    <div className="app">
      <header>
        <h1>NodeSense</h1>
        <span className={`status ${status}`}>{status}</span>
        {mode && <span className={`status mode-${mode}`}>{mode === 'live' ? 'model: live' : 'model: demo'}</span>}
      </header>

      <main>
        <section className="alert-feed">
          <h2>Live Alerts</h2>
          {alerts.length === 0 && <p className="empty">Waiting for traffic...</p>}
          <ul>
            {alerts.map((a) => (
              <li
                key={a.id}
                className={selected?.id === a.id ? 'selected' : ''}
                onClick={() => explainAlert(a)}
              >
                <span className="attack-type">{a.attack_type}</span>
                <span className="source">{a.source_ip}</span>
                <span className="confidence">
                  {(a.confidence * 100).toFixed(0)}%
                </span>
              </li>
            ))}
          </ul>
        </section>

        <section className="explanation-panel">
          <h2>Why was this flagged?</h2>
          {!selected && <p className="empty">Select an alert to see its explanation.</p>}
          {selected && !explanation && <p className="empty">Computing SHAP values...</p>}
          {selected && explanation && (
            <p className="alert-detail">
              <strong>{selected.attack_type}</strong> from {selected.source_ip}
              {' '}at {(selected.confidence * 100).toFixed(0)}% confidence
            </p>
          )}
          {explanation && explanation.length > 0 && (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={explanation} layout="vertical" margin={{ left: 40 }}>
                <XAxis type="number" domain={[-maxAbs * 1.1, maxAbs * 1.1]} tickFormatter={(v) => v.toFixed(2)} />
                <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 12 }} />
                <Tooltip
                  formatter={(v) => [v.toFixed(4), 'SHAP contribution']}
                  contentStyle={{ background: '#1a2029', border: '1px solid #2a323d' }}
                />
                <Bar dataKey="contribution">
                  {explanation.map((entry, i) => (
                    <Cell key={i} fill={entry.contribution > 0 ? '#d9534f' : '#5b8dd9'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
          {explanation && explanation.length > 0 && (
            <p className="legend">
              Red features pushed the model toward flagging this connection.
              Blue features pushed toward benign. Values are SHAP
              contributions to the predicted attack class probability.
            </p>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
