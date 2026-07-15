import { useEffect, useState } from 'react'
import { BellRing, Bot, CheckCircle2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import PageHeader from '../components/PageHeader.jsx'
import { api } from '../lib/api.js'

export default function AlertsPage() {
  const [alerts, setAlerts] = useState([])
  const navigate = useNavigate()
  const load = () => api.get('/api/alerts').then((response) => setAlerts(response.data.items || []))
  useEffect(() => { load(); const timer = window.setInterval(load, 5000); return () => window.clearInterval(timer) }, [])
  const investigate = async (alert) => {
    await api.put(`/api/alerts/${alert.id}/status`, { status: 'investigating' })
    const prompt = `Investigate this passive security alert in depth. Correlate it with current endpoint and SOC evidence, use as many additional compatible tools as necessary, explain whether it represents a credible threat, and recommend proportionate next actions. Do not execute containment without explicit authorization.\n\nPassive alert evidence:\n${JSON.stringify(alert, null, 2)}`
    navigate('/agent', { state: { initialPrompt: prompt, sourceAlertId: alert.id } })
  }
  const resolve = async (alert) => { await api.put(`/api/alerts/${alert.id}/status`, { status: 'resolved' }); load() }
  return <div className="space-y-6"><PageHeader title="Alerts" subtitle="Passive SOC and endpoint findings awaiting operator review." /><div className="space-y-3">{!alerts.length ? <div className="panel p-6 text-sm text-dim">No passive alerts have been emitted.</div> : alerts.map((alert) => <article key={alert.id} className="panel p-4"><div className="flex flex-wrap items-start justify-between gap-3"><div><div className="flex items-center gap-2"><BellRing className="h-4 w-4 text-amber-400" /><span className="badge badge-amber">{alert.severity}</span><span className="badge badge-dim">{alert.status}</span><span className="font-mono text-xs text-cyan">{alert.skill}</span></div><h2 className="mt-3 text-lg font-semibold text-text">{alert.title}</h2><div className="mt-1 text-sm text-dim">{new Date(alert.created_at).toLocaleString()} · {alert.count} finding(s)</div></div><div className="flex gap-2"><button className="btn btn-primary" onClick={() => investigate(alert)}><Bot className="h-4 w-4" /> Open investigation agent</button><button className="btn" onClick={() => resolve(alert)}><CheckCircle2 className="h-4 w-4" /> Resolve</button></div></div>{alert.analysis ? <pre className="mt-4 max-h-48 overflow-auto whitespace-pre-wrap rounded-lg bg-black/20 p-3 text-xs text-dim">{typeof alert.analysis === 'string' ? alert.analysis : JSON.stringify(alert.analysis, null, 2)}</pre> : null}<details className="mt-3"><summary className="cursor-pointer text-sm text-cyan">Evidence</summary><pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded-lg bg-black/20 p-3 text-xs text-dim">{JSON.stringify(alert.findings, null, 2)}</pre></details></article>)}</div></div>
}
