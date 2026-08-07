import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, Cell,
  CartesianGrid, LabelList,
} from 'recharts'
import { CHART, ATTACK_TYPES, attackColor } from '../lib'
import metrics from '../data/metrics.json'
import zeroDay from '../data/zero_day.json'
import latency from '../data/latency.json'

// Real numbers from backend/evaluate.py, zero_day_eval.py, benchmark.py,
// bundled at build time (frontend/src/data/*.json). Regenerate those files
// and re-copy them here after retraining — see README "Model card and evaluation".
const PER_CLASS = metrics.class_names.map((name) => ({
  model: name,
  f1: metrics.per_class[name]?.f1 ?? 0,
  auc: metrics.roc_auc_ovr[name] ?? 0,
}))

const ZERO_DAY_ROWS = zeroDay.folds.map((f) => ({
  type: f.held_out_class,
  recall: f.zero_day_anomaly_recall ?? 0,
}))

const MODEL_FACTS = [
  ['Architecture', '2-layer Transformer encoder'],
  ['Model dimension', '64 · 4 attention heads'],
  ['Input', '16 flows × 20 features'],
  ['Parameters', '≈ 150k'],
  ['Exported size', '335 KB (ONNX)'],
  ['Predict latency (p50 / p99)', `${latency.predict_only_ms.p50} ms / ${latency.predict_only_ms.p99} ms`],
  ['Predict + SHAP (p50 / p99)', `${latency.predict_with_explanation_ms.p50} ms / ${latency.predict_with_explanation_ms.p99} ms`],
  ['Benign false positive rate', metrics.benign_false_positive_rate ?? 'n/a'],
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
          Real numbers from <code>backend/evaluate.py</code> and{' '}
          <code>zero_day_eval.py</code>, computed on a held-out test split —
          not hardcoded placeholders. See{' '}
          <a href="https://github.com/quinlan-dev/NodeSense/blob/main/docs/MODEL_CARD.md" target="_blank" rel="noreferrer">
            the model card
          </a>{' '}
          for the important caveat that this is still synthetic data.
        </p>
      </div>

      <div className="two-col">
        <section className="card">
          <h2>Per-class performance (held-out test set)</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={PER_CLASS} margin={{ top: 18, right: 10, left: -18 }}>
              <CartesianGrid stroke={chart.grid} vertical={false} />
              <XAxis dataKey="model" stroke={chart.axis} tick={{ fontSize: 11, fill: chart.axis }} tickLine={false} />
              <YAxis domain={[0, 1]} stroke={chart.axis} tick={{ fontSize: 11, fill: chart.axis }} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'transparent' }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="f1" name="F1" fill={chart.cats[0]} barSize={18} radius={[4, 4, 0, 0]}>
                <LabelList dataKey="f1" position="top" style={{ fontSize: 10, fill: chart.axis }} formatter={(v) => v.toFixed(2)} />
              </Bar>
              <Bar dataKey="auc" name="ROC-AUC" fill={chart.cats[1]} barSize={18} radius={[4, 4, 0, 0]}>
                <LabelList dataKey="auc" position="top" style={{ fontSize: 10, fill: chart.axis }} formatter={(v) => v.toFixed(2)} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="legend">
            The transformer saturates this synthetic test split — its
            classes are separable by construction. Report the same
            evaluation run against real CICIDS-2018 data as the headline
            result; see the model card.
          </p>
        </section>

        <section className="card">
          <h2>Zero-day generalization (leave-one-attack-out)</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={ZERO_DAY_ROWS} layout="vertical" margin={{ left: 30, right: 30 }}>
              <XAxis type="number" domain={[0, 1]} stroke={chart.axis} tick={{ fontSize: 11, fill: chart.axis }} />
              <YAxis type="category" dataKey="type" width={90} stroke={chart.axis} tick={{ fontSize: 12, fill: chart.axis }} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'transparent' }}
                       formatter={(v) => [v.toFixed(3), 'anomaly recall when class was never trained on']} />
              <Bar dataKey="recall" barSize={16} radius={[0, 4, 4, 0]}>
                <LabelList dataKey="recall" position="right" style={{ fontSize: 11, fill: chart.axis }} formatter={(v) => v.toFixed(2)} />
                {ZERO_DAY_ROWS.map((d) => (
                  <Cell key={d.type} fill={attackColor(d.type, settings.theme)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="legend">
            Each bar retrains a model with that class fully removed from
            training, then measures recall on its held-out sessions as "any
            anomaly." Infiltration generalizes poorly (designed to mimic
            benign large transfers — see the model card); the others
            generalize well to attack types the model never saw.
          </p>
        </section>
      </div>

      <div className="two-col">
        <section className="card">
          <h2>Attacks seen this session</h2>
          {!hasAlerts && <p className="empty">Open the dashboard to start the stream, then check back.</p>}
          {hasAlerts && (
            <ResponsiveContainer width="100%" height={240}>
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

        <section className="card">
          <h2>Deployed model</h2>
          <div className="model-facts">
            {MODEL_FACTS.map(([k, v]) => (
              <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
            ))}
          </div>
        </section>
      </div>

      <div className="two-col">
        <section className="card">
          <h2>Confusion matrix</h2>
          <img src={`${import.meta.env.BASE_URL}confusion_matrix.png`} alt="Confusion matrix" style={{ width: '100%', borderRadius: 8 }} />
        </section>
        <section className="card">
          <h2>Global feature importance</h2>
          <img src={`${import.meta.env.BASE_URL}global_importance.png`} alt="Global SHAP feature importance" style={{ width: '100%', borderRadius: 8 }} />
        </section>
      </div>

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
    </>
  )
}

export default Analytics
