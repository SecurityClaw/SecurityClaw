import { useEffect, useState } from 'react'
import { api } from '../lib/api.js'
import PageHeader from '../components/PageHeader.jsx'

export default function CronsPage() {
  const [items, setItems] = useState([])
  const [working, setWorking] = useState(null)

  useEffect(() => {
    const load = () => api.get('/api/crons').then((res) => setItems(res.data.items || []))
    load()
    const timer = window.setInterval(load, 5000)
    return () => window.clearInterval(timer)
  }, [])

  const toggle = async (item) => {
    setWorking(item.name)
    try {
      await api.put(`/api/skills/${item.name}/enabled`, { enabled: !item.enabled })
      const res = await api.get('/api/crons')
      setItems(res.data.items || [])
    } finally { setWorking(null) }
  }

  const runNow = async (item) => {
    setWorking(item.name)
    try {
      await api.post(`/api/crons/${item.name}/run`)
      const res = await api.get('/api/crons')
      setItems(res.data.items || [])
    } finally { setWorking(null) }
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Crons" subtitle="Runtime schedules parsed from skill instruction frontmatter." />
      <div className="panel overflow-hidden">
        <div className="grid grid-cols-[1.2fr_0.9fr_1fr_1.2fr_1.5fr_1.4fr] gap-4 border-b border-border px-5 py-3 font-mono text-[11px] uppercase tracking-[0.18em] text-dim">
          <div>Skill</div>
          <div>State</div>
          <div>Type</div>
          <div>Schedule</div>
          <div>Last observation</div>
          <div>Controls</div>
        </div>
        <div>
          {items.length === 0 ? <div className="px-5 py-4 font-mono text-sm text-dim">No skill schedules detected.</div> : null}
          {items.map((item) => (
            <div key={item.name} className="grid grid-cols-[1.2fr_0.9fr_1fr_1.2fr_1.5fr_1.4fr] gap-4 border-b border-border/70 px-5 py-4 text-sm last:border-b-0">
              <div className="font-mono text-cyan">{item.name}</div>
              <div>
                <span className={`badge ${item.enabled ? 'badge-green' : 'badge-dim'}`}>{item.enabled ? 'active' : 'disabled'}</span>
              </div>
              <div>
                <span className={`badge ${item.type === 'cron' ? 'badge-amber' : item.type === 'interval' ? 'badge-green' : 'badge-dim'}`}>{item.type}</span>
              </div>
              <div className="font-mono text-text">{item.cron_expr || (item.interval_seconds !== null && item.interval_seconds !== undefined ? `every ${item.interval_seconds}s` : 'manual')}</div>
              <div className="text-dim"><div>{item.last_run ? new Date(item.last_run).toLocaleString() : 'Not run yet'}</div>{item.last_error ? <div className="mt-1 text-danger">{item.last_error}</div> : null}{item.last_result?.finding_count !== undefined ? <div className="mt-1 text-cyan">{item.last_result.finding_count} change finding(s)</div> : null}</div>
              <div className="flex flex-wrap gap-2"><button className={item.enabled ? 'btn btn-danger' : 'btn btn-primary'} onClick={() => toggle(item)} disabled={working === item.name}>{item.enabled ? 'Disable' : 'Enable'}</button>{item.type !== 'manual' && item.enabled ? <button className="btn" onClick={() => runNow(item)} disabled={working === item.name}>Run now</button> : null}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
