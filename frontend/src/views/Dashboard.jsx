import { useCallback, useMemo, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { apiBase, CHART, ATTACK_TYPES, attackColor } from '../lib'

function Dashboard({ stream, settings, paused, setPaused }) {
  const { alerts, status, mode, clear } = stream
  const [selected, setSelected] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [typeFilter, setTypeFilter] = useState('all')
  const chart = CHART[settings.theme]

  const visible = useMemo(
    () => alerts.filter(
      (a) => a.confidence >= settings.threshold
        && (typeFilter === 'all' || a.attack_type === typeFilter)
    ),
    [alerts, settings.threshold, typeFilter]
  )

  const stats = useMemo(() => {
    const avg = visible.length
      ? visible.reduce((s, a) => s + a.confidence, 0) / visible.length
      : 0
    const counts = {}
    visible.forEach((a) => { counts[a.attack_type] = (counts[a.attack_type] || 0) + 1 })
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]
    return {
      total: visible.length,
      avg: (avg * 100).toFixed(0),
      top: top ? top[0] : '—',
    }
  }, [visible])

  const explainAlert = useCallback(async (alert) => {
    setSelected(alert)
    setExplanation(null)
    try {
      const res = await fetch(`${apiBase(settings)}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: alert.features, explain: true }),
      })
      const data = await res.json()
      setExplanation(data.explanation || [])
    } catch {
      setExplanation([])
    }
  }, [settings])

  const maxAbs = explanation?.length
    ? Math.max(...explanation.map((e) => Math.abs(e.contribution)), 0.1)
    : 0.5

  return (
    <>
      <div className="page-head">
        <h1>Live detection</h1>
        <p>
          Traffic sessions are generated continuously and classified by the
          deployed transformer. Click any alert to compute its SHAP explanation.
        </p>
      </div>

      <div className="dash-tiles">
        <div className="stat-tile"><div className="val">{stats.total}</div><div className="lbl">alerts in view</div></div>
        <div className="stat-tile"><div className="val">{stats.avg}%</div><div className="lbl">avg confidence</div></div>
        <div className="stat-tile"><div className="val">{stats.top}</div><div className="lbl">most frequent attack</div></div>
        <div className="stat-tile">
          <div className="val" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className={`status ${paused ? 'paused' : status}`}>{paused ? 'paused' : status}</span>
            {mode && <span className={`status mode-${mode}`}>{mode === 'live' ? 'model live' : 'model demo'}</span>}
          </div>
          <div className="lbl">stream · model</div>
        </div>
      </div>

      <div className="dash-grid">
        <section className="card alert-feed">
          <h2>Alert feed</h2>
          <div className="feed-controls">
            <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
              <option value="all">All attack types</option>
              {ATTACK_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
            <button className="btn small" onClick={() => setPaused(!paused)}>
              {paused ? 'Resume' : 'Pause'}
            </button>
            <button className="btn small" onClick={clear}>Clear</button>
          </div>
          {visible.length === 0 && (
            <p className="empty">
              {alerts.length ? 'No alerts match the current filters.' : 'Waiting for traffic...'}
            </p>
          )}
          <ul>
            {visible.map((a) => (
              <li
                key={a.id}
                className={selected?.id === a.id ? 'selected' : ''}
                style={{ borderLeftColor: attackColor(a.attack_type, settings.theme) }}
                onClick={() => explainAlert(a)}
              >
                <span className="attack-type">{a.attack_type}</span>
                <span className="source">{a.source_ip}</span>
                <span className="confidence">{(a.confidence * 100).toFixed(0)}%</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="card explanation-panel">
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
            <>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={explanation} layout="vertical" margin={{ left: 40, right: 10 }}>
                  <XAxis
                    type="number"
                    domain={[-maxAbs * 1.1, maxAbs * 1.1]}
                    tickFormatter={(v) => v.toFixed(2)}
                    stroke={chart.axis}
                    tick={{ fontSize: 11, fill: chart.axis }}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    width={150}
                    stroke={chart.axis}
                    tick={{ fontSize: 12, fill: chart.axis }}
                  />
                  <Tooltip
                    formatter={(v) => [v.toFixed(4), 'SHAP contribution']}
                    contentStyle={{
                      background: chart.tooltipBg,
                      border: `1px solid ${chart.tooltipBorder}`,
                      borderRadius: 8,
                    }}
                    cursor={{ fill: 'transparent' }}
                  />
                  <Bar dataKey="contribution" barSize={14} radius={[0, 4, 4, 0]}>
                    {explanation.map((entry, i) => (
                      <Cell key={i} fill={entry.contribution > 0 ? chart.attack : chart.benign} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <p className="legend">
                <span style={{ color: chart.attack, fontWeight: 700 }}>■</span> pushed
                the model toward flagging this connection ·{' '}
                <span style={{ color: chart.benign, fontWeight: 700 }}>■</span> pushed
                toward benign. Values are SHAP contributions to the predicted
                attack class probability.
              </p>
            </>
          )}
        </section>
      </div>
    </>
  )
}

export default Dashboard
