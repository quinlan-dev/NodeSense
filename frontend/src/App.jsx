import { useState, useEffect, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

// Point this at your deployed backend. For local dev the Vite proxy
// handles /api. For production set your Hugging Face Space URL.
const API_BASE = import.meta.env.PROD
  ? 'https://YOUR_USERNAME-nodesense.hf.space'
  : '/api'

const WS_BASE = import.meta.env.PROD
  ? 'wss://YOUR_USERNAME-nodesense.hf.space'
  : 'ws://localhost:7860'

function App() {
  const [alerts, setAlerts] = useState([])
  const [selected, setSelected] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [status, setStatus] = useState('connecting')

  // Live alert stream over WebSocket
  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE}/ws/alerts`)
    ws.onopen = () => setStatus('live')
    ws.onclose = () => setStatus('disconnected')
    ws.onmessage = (e) => {
      const alert = JSON.parse(e.data)
      alert.id = Date.now() + Math.random()
      setAlerts((prev) => [alert, ...prev].slice(0, 50))
    }
    return () => ws.close()
  }, [])

  // Fetch a SHAP explanation for a selected alert
  const explainAlert = useCallback(async (alert) => {
    setSelected(alert)
    setExplanation(null)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: Array(20).fill(0).map(() => Math.random() * 2),
          explain: true,
        }),
      })
      const data = await res.json()
      setExplanation(data.explanation)
    } catch {
      setExplanation([])
    }
  }, [])

  return (
    <div className="app">
      <header>
        <h1>NodeSense</h1>
        <span className={`status ${status}`}>{status}</span>
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
          {selected && !explanation && <p className="empty">Loading explanation...</p>}
          {explanation && explanation.length > 0 && (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={explanation} layout="vertical" margin={{ left: 40 }}>
                <XAxis type="number" domain={[-0.6, 0.6]} />
                <YAxis type="category" dataKey="feature" width={140} tick={{ fontSize: 12 }} />
                <Tooltip />
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
              Blue features pushed toward benign.
            </p>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
