import { useMemo } from 'react'
import { AlertTriangle, Boxes, Network, PackageSearch, ShieldCheck } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const COLORS = ['#dc2626', '#f97316', '#facc15', '#38bdf8', '#94a3b8']

function scoreSeverity(item) {
  if (item.severity_label) return item.severity_label
  const text = JSON.stringify(item.severity || []).toLowerCase()
  const score = Number((text.match(/(?:cvss[^0-9]*)?(10(?:\.0)?|[0-9](?:\.[0-9])?)/) || [])[1])
  if (score >= 9) return 'critical'
  if (score >= 7) return 'high'
  if (score >= 4) return 'medium'
  if (score > 0) return 'low'
  return 'unknown'
}

export default function EndpointInsights({ messages, liveSteps }) {
  const assistant = [...messages].reverse().find((message) => message.role === 'assistant') || {}
  const results = assistant.skill_results || {}
  const timeline = liveSteps.length ? liveSteps : (assistant.agent_timeline || [])
  const data = useMemo(() => {
    const values = Object.values(results)
    const flatten = (key) => values.flatMap((result) => result?.[key] || [])
    const vulnerabilities = flatten('vulnerabilities')
    const severityCounts = { critical: 0, high: 0, medium: 0, low: 0, unknown: 0 }
    vulnerabilities.forEach((item) => { severityCounts[scoreSeverity(item)] += 1 })
    const connections = flatten('connections')
    const protocols = {}
    connections.forEach((item) => { const key = item.protocol || item.State || 'unknown'; protocols[key] = (protocols[key] || 0) + 1 })
    const errors = values.flatMap((result) => result?.errors || []).filter(Boolean)
    const coverage = values.find((result) => result?.coverage)?.coverage || {}
    return {
      vulnerabilities, severityCounts, connections, protocols, errors, coverage,
      processes: flatten('processes'), packages: flatten('packages'), services: flatten('services'),
      persistence: flatten('persistence'), files: flatten('files'), checks: values.flatMap((result) => Object.entries(result?.checks || {})),
      neighbors: flatten('neighbors'), routes: flatten('routes'), passiveFindings: flatten('findings'),
    }
  }, [results])
  const severityData = Object.entries(data.severityCounts).map(([name, value]) => ({ name, value }))
  const protocolData = Object.entries(data.protocols).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value).slice(0, 10)
  const tools = new Set(timeline.flatMap((item) => item.skills || []))
  const cards = [
    ['Vulnerabilities', data.vulnerabilities.length, AlertTriangle], ['Installed packages', data.packages.length, PackageSearch],
    ['Processes', data.processes.length, Boxes], ['Connections', data.connections.length, Network],
    ['Services', data.services.length, ShieldCheck], ['Persistence entries', data.persistence.length, ShieldCheck],
    ['ARP/NDP neighbors', data.neighbors.length, Network], ['Network findings', data.passiveFindings.length, AlertTriangle],
    ['Integrity records', data.files.length, ShieldCheck], ['Agent steps', timeline.length, ShieldCheck],
  ]
  return (
    <div className="min-h-0 flex-1 overflow-auto p-5">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8">{cards.map(([label, value, Icon]) => <div key={label} className="rounded-xl border border-border bg-panel2 p-3"><Icon className="h-4 w-4 text-cyan" /><div className="mt-3 text-2xl font-semibold text-text">{value}</div><div className="text-xs text-dim">{label}</div></div>)}</div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <section className="rounded-xl border border-border bg-panel2 p-4"><h3 className="font-mono text-xs uppercase tracking-wider text-cyan">Vulnerability severity</h3><div className="h-64"><ResponsiveContainer><PieChart><Pie data={severityData} dataKey="value" nameKey="name" innerRadius={50} outerRadius={85} label>{severityData.map((entry, index) => <Cell key={entry.name} fill={COLORS[index]} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer></div></section>
        <section className="rounded-xl border border-border bg-panel2 p-4"><h3 className="font-mono text-xs uppercase tracking-wider text-cyan">Network protocol and state distribution</h3><div className="h-64"><ResponsiveContainer><BarChart data={protocolData}><CartesianGrid stroke="#243047" /><XAxis dataKey="name" stroke="#94a3b8" /><YAxis stroke="#94a3b8" /><Tooltip /><Bar dataKey="value" fill="#38bdf8" /></BarChart></ResponsiveContainer></div></section>
        <section className="rounded-xl border border-border bg-panel2 p-4"><h3 className="font-mono text-xs uppercase tracking-wider text-cyan">Coverage and data quality</h3><dl className="mt-3 grid grid-cols-2 gap-3 text-sm"><div><dt className="text-dim">Packages queried</dt><dd className="text-lg text-text">{data.coverage.queried ?? '—'}</dd></div><div><dt className="text-dim">Packages available</dt><dd className="text-lg text-text">{data.coverage.available ?? data.packages.length}</dd></div><div><dt className="text-dim">Advisory source</dt><dd className="text-lg text-text">{data.coverage.source || '—'}</dd></div><div><dt className="text-dim">Collection errors</dt><dd className="text-lg text-text">{data.errors.length}</dd></div></dl>{data.errors.length ? <ul className="mt-3 max-h-32 overflow-auto text-xs text-danger">{data.errors.map((error, index) => <li key={`${error}-${index}`}>{error}</li>)}</ul> : null}</section>
        <section className="rounded-xl border border-border bg-panel2 p-4"><h3 className="font-mono text-xs uppercase tracking-wider text-cyan">Investigation coverage</h3><div className="mt-3 flex flex-wrap gap-2">{[...tools].map((tool) => <span key={tool} className="badge badge-green">{tool}</span>)}{!tools.size ? <span className="text-sm text-dim">No tools have run.</span> : null}</div><div className="mt-4 text-sm text-dim">Defensive checks: {data.checks.length}. Findings are evidence-driven; missing telemetry is not treated as a clean result.</div></section>
      </div>
      <section className="mt-4 rounded-xl border border-border bg-panel2 p-4"><h3 className="font-mono text-xs uppercase tracking-wider text-cyan">Vulnerability findings</h3>{data.vulnerabilities.length ? <div className="mt-3 overflow-auto"><table className="w-full text-left text-sm"><thead className="text-dim"><tr><th className="p-2">Severity</th><th className="p-2">CVE / advisory</th><th className="p-2">Installed package</th><th className="p-2">Summary</th></tr></thead><tbody>{data.vulnerabilities.slice(0, 200).map((item, index) => <tr key={`${item.id}-${index}`} className="border-t border-border"><td className="p-2 uppercase">{scoreSeverity(item)}</td><td className="p-2 text-cyan">{item.cves?.join(', ') || item.id}</td><td className="p-2">{item.package} {item.installed_version}</td><td className="max-w-xl p-2 text-dim">{item.summary || 'No advisory summary provided.'}</td></tr>)}</tbody></table></div> : <div className="mt-3 text-sm text-dim">No grounded vulnerability records are available. Check correlation coverage before interpreting this as a clean result.</div>}</section>
    </div>
  )
}
