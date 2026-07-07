import { useState } from 'react'
import { DEFAULT_API } from '../lib'

function Settings({ settings, update, reset, onLogout }) {
  const [apiDraft, setApiDraft] = useState(settings.apiUrl)

  return (
    <>
      <div className="page-head">
        <h1>Settings</h1>
        <p>
          Preferences are stored in your browser. The API endpoint override is
          useful for pointing the deployed dashboard at a local backend, or
          your own fork of the Space.
        </p>
      </div>

      <div className="settings-stack">
        <section className="card">
          <h2>Appearance</h2>
          <div className="setting-row">
            <label className="field">Theme</label>
            <div style={{ display: 'flex', gap: 10 }}>
              <button
                className={`btn small ${settings.theme === 'dark' ? 'primary' : ''}`}
                onClick={() => update({ theme: 'dark' })}
              >☾ Dark</button>
              <button
                className={`btn small ${settings.theme === 'light' ? 'primary' : ''}`}
                onClick={() => update({ theme: 'light' })}
              >☀ Light</button>
            </div>
          </div>
        </section>

        <section className="card">
          <h2>Alert feed</h2>
          <div className="setting-row">
            <label className="field">Minimum confidence to show an alert</label>
            <div className="range-row">
              <input
                type="range" min="0" max="0.95" step="0.05"
                value={settings.threshold}
                onChange={(e) => update({ threshold: parseFloat(e.target.value) })}
              />
              <span className="range-val">{(settings.threshold * 100).toFixed(0)}%</span>
            </div>
            <p className="desc">
              Alerts below this confidence are hidden from the dashboard feed
              and session statistics. The model still classifies everything.
            </p>
          </div>
          <div className="setting-row">
            <label className="field">Alerts kept in the feed</label>
            <select
              style={{ width: 140 }}
              value={settings.maxAlerts}
              onChange={(e) => update({ maxAlerts: parseInt(e.target.value, 10) })}
            >
              <option value="20">20</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
          </div>
        </section>

        <section className="card">
          <h2>Backend</h2>
          <div className="setting-row">
            <label className="field">API endpoint override</label>
            <input
              type="url"
              placeholder={DEFAULT_API}
              value={apiDraft}
              onChange={(e) => setApiDraft(e.target.value)}
            />
            <p className="desc">
              Leave empty to use the default ({DEFAULT_API}). WebSocket and REST
              calls both follow this setting. Changes reconnect the stream.
            </p>
            <div style={{ display: 'flex', gap: 10 }}>
              <button className="btn small primary" onClick={() => update({ apiUrl: apiDraft.trim() })}>
                Apply
              </button>
              <button className="btn small" onClick={() => { setApiDraft(''); update({ apiUrl: '' }) }}>
                Use default
              </button>
            </div>
          </div>
        </section>

        <section className="card">
          <h2>Session</h2>
          <div className="setting-row">
            <p className="desc">
              Signing out clears your demo session. Resetting preferences
              restores every setting above to its default.
            </p>
            <div style={{ display: 'flex', gap: 10 }}>
              <button className="btn small" onClick={reset}>Reset preferences</button>
              <button className="btn small" style={{ color: 'var(--danger)' }} onClick={onLogout}>
                Sign out
              </button>
            </div>
          </div>
        </section>
      </div>
    </>
  )
}

export default Settings
