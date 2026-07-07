import { useState } from 'react'
import { useAuth, useRoute, useSettings, useAlertStream, navigate } from './lib'
import Landing from './views/Landing'
import Login from './views/Login'
import Dashboard from './views/Dashboard'
import Analytics from './views/Analytics'
import Docs from './views/Docs'
import Settings from './views/Settings'

const NAV = [
  { route: 'dashboard', label: 'Dashboard', auth: true },
  { route: 'analytics', label: 'Analytics', auth: true },
  { route: 'docs', label: 'Docs', auth: false },
  { route: 'settings', label: 'Settings', auth: true },
]

function App() {
  const route = useRoute()
  const { user, login, logout } = useAuth()
  const { settings, update, reset } = useSettings()
  const [paused, setPaused] = useState(false)
  // The stream lives at the shell level so the feed survives navigation.
  const stream = useAlertStream(settings, paused)

  const protectedRoutes = ['dashboard', 'analytics', 'settings']
  const effective = protectedRoutes.includes(route) && !user ? 'login' : route

  let view
  switch (effective) {
    case 'login':
      view = <Login onLogin={(name) => { login(name); navigate('dashboard') }} />
      break
    case 'dashboard':
      view = <Dashboard stream={stream} settings={settings} paused={paused} setPaused={setPaused} />
      break
    case 'analytics':
      view = <Analytics stream={stream} settings={settings} />
      break
    case 'docs':
      view = <Docs />
      break
    case 'settings':
      view = <Settings settings={settings} update={update} reset={reset} onLogout={() => { logout(); navigate('') }} />
      break
    default:
      view = <Landing user={user} mode={stream.mode} />
  }

  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href="#/">
          <span className="brand-mark">N</span> NodeSense
        </a>
        <nav className="mainnav">
          {NAV.map((n) => (
            <a key={n.route} href={`#/${n.route}`} className={route === n.route ? 'active' : ''}>
              {n.label}
            </a>
          ))}
        </nav>
        <div className="topbar-right">
          <button
            className="icon-btn"
            title="Toggle theme"
            onClick={() => update({ theme: settings.theme === 'dark' ? 'light' : 'dark' })}
          >
            {settings.theme === 'dark' ? '☀' : '☾'}
          </button>
          {user ? (
            <span className="user-chip">
              <span className="avatar">{user.name.slice(0, 1).toUpperCase()}</span>
              {user.name}
              <button onClick={() => { logout(); navigate('') }}>sign out</button>
            </span>
          ) : (
            <a className="btn small primary" href="#/login">Sign in</a>
          )}
        </div>
      </header>

      {effective === '' ? view : <main className="content">{view}</main>}

      <footer className="site">
        NodeSense — explainable network anomaly detection · Graduate independent
        study, UC Santa Cruz ·{' '}
        <a href="https://github.com/quinlan-dev/NodeSense" target="_blank" rel="noreferrer">GitHub</a>
      </footer>
    </div>
  )
}

export default App
