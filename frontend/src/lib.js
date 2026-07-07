// Shared app logic: settings, demo auth, hash routing, API resolution,
// and the live alert stream. All persistence is localStorage — this is a
// client-only demo with no account backend.

import { useCallback, useEffect, useRef, useState } from 'react'

export const DEFAULT_API = import.meta.env.PROD
  ? 'https://quinlan-dev-nodesense.hf.space'
  : '/api'

const DEFAULT_SETTINGS = {
  apiUrl: '',          // empty = use DEFAULT_API
  threshold: 0.5,      // hide alerts below this confidence
  maxAlerts: 50,
  theme: 'dark',
}

// ---------- settings ----------

export function loadSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem('ns_settings') || '{}') }
  } catch {
    return { ...DEFAULT_SETTINGS }
  }
}

export function useSettings() {
  const [settings, setSettings] = useState(loadSettings)
  const update = useCallback((patch) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch }
      localStorage.setItem('ns_settings', JSON.stringify(next))
      return next
    })
  }, [])
  const reset = useCallback(() => {
    localStorage.removeItem('ns_settings')
    setSettings({ ...DEFAULT_SETTINGS })
  }, [])
  useEffect(() => {
    document.documentElement.dataset.theme = settings.theme
  }, [settings.theme])
  return { settings, update, reset }
}

export function apiBase(settings) {
  return (settings.apiUrl || DEFAULT_API).replace(/\/$/, '')
}

export function wsBase(settings) {
  const api = settings.apiUrl || (import.meta.env.PROD ? DEFAULT_API : 'http://localhost:7860')
  return api.replace(/\/$/, '').replace(/^http/, 'ws')
}

// ---------- demo auth (client-side only, no real security) ----------

export function useAuth() {
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('ns_user')) } catch { return null }
  })
  const login = useCallback((name) => {
    const u = { name: name.trim(), since: Date.now() }
    localStorage.setItem('ns_user', JSON.stringify(u))
    setUser(u)
  }, [])
  const logout = useCallback(() => {
    localStorage.removeItem('ns_user')
    setUser(null)
  }, [])
  return { user, login, logout }
}

// ---------- hash router ----------

export function useRoute() {
  const parse = () => window.location.hash.replace(/^#\/?/, '') || ''
  const [route, setRoute] = useState(parse)
  useEffect(() => {
    const onHash = () => setRoute(parse())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  return route
}

export function navigate(route) {
  window.location.hash = `#/${route}`
}

// ---------- live alert stream ----------

export function useAlertStream(settings, paused) {
  const [alerts, setAlerts] = useState([])
  const [status, setStatus] = useState('connecting')
  const [mode, setMode] = useState(null)
  const pausedRef = useRef(paused)
  pausedRef.current = paused
  const maxRef = useRef(settings.maxAlerts)
  maxRef.current = settings.maxAlerts

  const pushAlert = useCallback((alert) => {
    if (pausedRef.current) return
    alert.id = Date.now() + Math.random()
    setAlerts((prev) => [alert, ...prev].slice(0, maxRef.current))
  }, [])

  const api = apiBase(settings)
  const ws = wsBase(settings)

  useEffect(() => {
    fetch(`${api}/`)
      .then((r) => r.json())
      .then((d) => setMode(d.mode))
      .catch(() => setMode(null))
  }, [api])

  useEffect(() => {
    let sock
    let closed = false
    let pollTimer = null

    const startPolling = () => {
      if (pollTimer || closed) return
      setStatus('polling')
      pollTimer = setInterval(async () => {
        try {
          const res = await fetch(`${api}/demo/stream?n=1`)
          const data = await res.json()
          data.alerts.forEach(pushAlert)
        } catch {
          setStatus('disconnected')
        }
      }, 3000)
    }

    try {
      sock = new WebSocket(`${ws}/ws/alerts`)
      sock.onopen = () => setStatus('live')
      sock.onmessage = (e) => pushAlert(JSON.parse(e.data))
      sock.onerror = () => { if (!closed) startPolling() }
      sock.onclose = () => { if (!closed) startPolling() }
    } catch {
      startPolling()
    }

    return () => {
      closed = true
      if (sock) sock.close()
      if (pollTimer) clearInterval(pollTimer)
    }
  }, [api, ws, pushAlert])

  const clear = useCallback(() => setAlerts([]), [])
  return { alerts, status, mode, clear }
}

// ---------- chart theme (values validated with the dataviz palette checker) ----------

export const CHART = {
  dark: {
    attack: '#e66767',
    benign: '#3987e5',
    cats: ['#3987e5', '#199e70', '#c98500', '#008300', '#9085e9'],
    grid: '#2c2f36',
    axis: '#8b96a5',
    tooltipBg: '#1a2029',
    tooltipBorder: '#2a323d',
  },
  light: {
    attack: '#d03b3b',
    benign: '#2a78d6',
    cats: ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7'],
    grid: '#e6e8ec',
    axis: '#5c6773',
    tooltipBg: '#ffffff',
    tooltipBorder: '#dfe3e8',
  },
}

export const ATTACK_TYPES = ['DDoS', 'Port Scan', 'Brute Force', 'Botnet', 'Infiltration']

// Fixed slot per attack class — color follows the entity, never the rank.
export function attackColor(type, theme) {
  const i = ATTACK_TYPES.indexOf(type)
  return CHART[theme].cats[i >= 0 ? i : 0]
}
