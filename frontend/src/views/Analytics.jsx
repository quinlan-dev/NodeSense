import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, Cell,
  CartesianGrid, LabelList,
} from 'recharts'
import { CHART, ATTACK_TYPES, attackColor } from '../lib'

// Benchmark numbers from the committed training run (see docs/research_log.md).
// Binary anomaly detection on the held-out synthetic test split; the
// transformer saturates synthetic data — real CICIDS-2018 results replace
// these after the full dataset run.
const BENCHMARKS = [
  { model: 'Autoencoder', auc: 0.959, f1: 0.824 },
  { model: 'Random Forest', auc: 0.998, f1: 0.976 },
  { model: 'Transformer', auc: 1.0, f1: 1.0 },
]

const MODEL_FACTS = [
  ['Architecture', '2-layer Transformer encoder'],
  ['Model dimension', '64 · 4 attention heads'],
  ['Input', '16 flows × 20 features'],
  ['Parameters', '≈ 150k'],
  ['Exported size', '335 KB (ONNX)'],
  ['Inference', '< 5 ms / sequence (CPU)'],
  ['Explanation', '~120 ms KernelSHAP'],
  ['Training data', '4,000 sessions · 64,000 flows'],
]

function Analytics({ stream, settings }) {
  const chart = CHART[settings.theme]

  const distribution = useMemo(() => {
    const counts = Object.fromEntries(ATTACK_TYPES.map((t) => [t, 0]))
    stream.alerts.forEach((a) => {
      if (a.attack_type in counts) counts[a.attack_type] += 1
    })
    return ATTACK_TYPES.map((t) => ({ type: t, count: counts[t] }))
  }, [stream.alerts])

  const hasAlerts = stream.alerts.length > 0

  const tooltipStyle = {
    background: chart.tooltipBg,
    border: `1px solid ${chart.tooltipBorder}`,
    borderRadius: 8,
  }

  return (
    <>
      <div className="page-head">
        <h1>Model analytics</h1>
        <p>
          How the three candidate models compare, and what the live stream has
          seen this session. All three train in one pipeline on identical data
          splits, so the comparison is honest.
        </p>
      </div>

      <div className="two-col">
        <section className="card">
          <h2>Benchmark — binary anomaly detection (held-out test set)</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={BENCHMARKS} margin={{ top: 18, right: 10, left: -18 }}>
              <CartesianGrid stroke={chart.grid} vertical={false} />
              <XAxis dataKey="model" stroke={chart.axis} tick={{ fontSize: 12, fill: chart.axis }} tickLine={false} />
              <YAxis domain={[0, 1]} stroke={chart.axis} tick={{ fontSize: 11, fill: chart.axis }} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'transparent' }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="auc" name="AUC-ROC" fill={chart.cats[0]} barSize={22} radius={[4, 4, 0, 0]}>
                <LabelList dataKey="auc" position="top" style={{ fontSize: 11, fill: chart.axis }} formatter={(v) => v.toFixed(3)} />
              </Bar>
              <Bar dataKey="f1" name="Attack F1" fill={chart.cats[1]} barSize={22} radius={[4, 4, 0, 0]}>
                <LabelList dataKey="f1" position="top" style={{ fontSize: 11, fill: chart.axis }} formatter={(v) => v.toFixed(3)} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="legend">
            The transformer saturates the synthetic test split (a perfect score
            is expected there — the generator's classes are separable); the
            numbers to report publicly come from the full CICIDS-2018 run. The
            autoencoder trains on benign traffic only, which is why its recall
            trails the supervised models.
          </p>
        </section>

        <section className="card">
          <h2>Attacks seen this session</h2>
          {!hasAlerts && <p className="empty">Open the dashboard to start the stream, then check back.</p>}
          {hasAlerts && (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={distribution} layout="vertical" margin={{ left: 30, right: 30 }}>
                <XAxis type="number" allowDecimals={false} stroke={chart.axis} tick={{ fontSize: 11, fill: chart.axis }} />
                <YAxis type="category" dataKey="type" width={90} stroke={chart.axis} tick={{ fontSize: 12, fill: chart.axis }} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'transparent' }} />
                <Bar dataKey="count" name="alerts" barSize={16} radius={[0, 4, 4, 0]}>
                  <LabelList dataKey="count" position="right" style={{ fontSize: 11, fill: chart.axis }} />
                  {distribution.map((d) => (
                    <Cell key={d.type} fill={attackColor(d.type, settings.theme)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
          <p className="legend">
            Live counts of model-classified alerts received this session,
            colored by attack class.
          </p>
        </section>
      </div>

      <div className="two-col">
        <section className="card">
          <h2>Deployed model</h2>
          <div className="model-facts">
            {MODEL_FACTS.map(([k, v]) => (
              <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
            ))}
          </div>
        </section>

        <section className="card">
          <h2>What each attack looks like in the features</h2>
          <table className="docs">
            <tbody>
              <tr><td><strong>DDoS</strong></td><td>extreme packet and byte rates, tiny inter-arrival times, SYN-heavy, almost no return traffic</td></tr>
              <tr><td><strong>Port Scan</strong></td><td>one or two tiny SYN packets per flow, microsecond durations, machine-regular timing</td></tr>
              <tr><td><strong>Brute Force</strong></td><td>short repeated auth attempts, PSH on every attempt, metronome retry rhythm</td></tr>
              <tr><td><strong>Botnet</strong></td><td>long mostly-idle flows with small periodic check-ins — very high idle time, low variance</td></tr>
              <tr><td><strong>Infiltration</strong></td><td>long flows dominated by large server-to-attacker transfers; down/up ratio far above normal</td></tr>
            </tbody>
          </table>
        </section>
      </div>
    </>
  )
}

export default Analytics
