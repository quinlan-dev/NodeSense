import { useState } from 'react'

function Login({ onLogin }) {
  const [name, setName] = useState('')
  const [pass, setPass] = useState('')

  const submit = (e) => {
    e.preventDefault()
    if (name.trim()) onLogin(name)
  }

  return (
    <div className="login-wrap">
      <div className="login-card">
        <h1>Sign in to NodeSense</h1>
        <p className="hint">
          Enter a display name to open the live detection dashboard.
        </p>
        <form onSubmit={submit}>
          <div>
            <label className="field" htmlFor="login-name">Display name</label>
            <input
              id="login-name"
              type="text"
              placeholder="e.g. Quinlan"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
              required
            />
          </div>
          <div>
            <label className="field" htmlFor="login-pass">Password</label>
            <input
              id="login-pass"
              type="password"
              placeholder="any password works in demo mode"
              value={pass}
              onChange={(e) => setPass(e.target.value)}
            />
          </div>
          <button className="btn primary" type="submit">Sign in</button>
        </form>
        <p className="demo-note">
          <strong>Demo authentication.</strong> This is a client-side session for
          the project demo — any credentials are accepted and nothing is sent to
          a server. A production deployment would put real auth (OAuth / SSO) in
          front of the API.
        </p>
      </div>
    </div>
  )
}

export default Login
