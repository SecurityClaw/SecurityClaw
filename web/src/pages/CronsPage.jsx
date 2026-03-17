import { useEffect, useState } from 'react'
import { api } from '../lib/api.js'
import PageHeader from '../components/PageHeader.jsx'

export default function CronsPage() {
  const [items, setItems] = useState([])

  useEffect(() => {
    api.get('/api/crons').then((res) => setItems(res.data.items || []))
  }, [])

  return (
    <div className="space-y-6">
      <PageHeader title="Crons" subtitle="Runtime schedules parsed from skill instruction frontmatter." />
      <div className="panel overflow-hidden">
        <div className="grid grid-cols-[1.2fr_0.9fr_1fr_1.2fr_2fr] gap-4 border-b border-border px-5 py-3 font-mono text-[11px] uppercase tracking-[0.18em] text-dim">
          <div>Skill</div>
          <div>State</div>
          <div>Type</div>
          <div>Schedule</div>
          <div>Description</div>
        </div>
        <div>
          {items.length === 0 ? <div className="px-5 py-4 font-mono text-sm text-dim">No skill schedules detected.</div> : null}
          {items.map((item) => (
            <div key={item.name} className="grid grid-cols-[1.2fr_0.9fr_1fr_1.2fr_2fr] gap-4 border-b border-border/70 px-5 py-4 text-sm last:border-b-0">
              <div className="font-mono text-cyan">{item.name}</div>
              <div>
                <span className={`badge ${item.enabled ? 'badge-green' : 'badge-dim'}`}>{item.enabled ? 'active' : 'disabled'}</span>
              </div>
              <div>
                <span className={`badge ${item.type === 'cron' ? 'badge-amber' : item.type === 'interval' ? 'badge-green' : 'badge-dim'}`}>{item.type}</span>
              </div>
              <div className="font-mono text-text">{item.cron_expr || (item.interval_seconds !== null && item.interval_seconds !== undefined ? `every ${item.interval_seconds}s` : 'manual')}</div>
              <div className="text-dim">{item.description}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
