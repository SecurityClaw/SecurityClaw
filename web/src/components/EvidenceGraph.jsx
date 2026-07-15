import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { Box, Download, Focus, Maximize2, Minimize2, PanelRightClose, PanelRightOpen, Pencil, RotateCcw, Search, Tags } from 'lucide-react'

const ForceGraph3D = lazy(() => import('react-force-graph-3d'))

const COLORS = {
  host: '#67e8f9', skill: '#a3e635', process: '#c084fc', network: '#38bdf8',
  persistence: '#fb923c', file: '#facc15', package: '#94a3b8', service: '#2dd4bf',
  vulnerability: '#fb7185', control: '#4ade80', error: '#ef4444',
  neighbor: '#22d3ee', route: '#818cf8', interface: '#34d399', finding: '#f43f5e',
}
const SEVERITY_COLORS = { critical: '#dc2626', high: '#f97316', medium: '#facc15', low: '#38bdf8', info: '#94a3b8' }

function severityFromVulnerability(vulnerability) {
  return vulnerability.severity_label || 'unknown'
}

function buildEvidenceGraph(skillResults = {}, annotations = {}) {
  const nodes = [{ id: 'host', name: 'Endpoint', type: 'host', description: 'Endpoint under defensive investigation.', evidence: {} }]
  const links = []
  const seen = new Set(['host'])
  const packageByName = new Map()
  const addNode = (node, parent = 'host', relation = 'OBSERVED_BY') => {
    if (!node.id || seen.has(node.id)) return
    seen.add(node.id)
    const annotation = annotations[node.id] || {}
    nodes.push({ ...node, name: annotation.name || node.name, analystSeverity: annotation.severity, notes: annotation.notes })
    links.push({ source: parent, target: node.id, type: relation })
  }

  Object.entries(skillResults).forEach(([skill, result]) => {
    if (!result || typeof result !== 'object') return
    const skillId = `skill:${skill}`
    addNode({ id: skillId, name: skill.replaceAll('_', ' '), type: result.status === 'error' ? 'error' : 'skill', description: `Runtime evidence produced by ${skill}.`, evidence: { status: result.status, errors: result.errors } }, 'host', 'EXECUTED')
    if (result.hostname) addNode({ id: `host:${result.hostname}`, name: result.hostname, type: 'host', description: result.os || 'Discovered endpoint identity.', evidence: result }, skillId, 'IDENTIFIED')
    ;(result.processes || []).slice(0, 80).forEach((process) => addNode({ id: `process:${process.pid}`, name: `${process.name || 'process'} (${process.pid || '?'})`, type: 'process', description: process.command || process.executable || 'Running process.', evidence: process }, skillId, 'RUNNING'))
    ;(result.connections || []).slice(0, 80).forEach((connection, index) => {
      const remote = connection.remote || `${connection.RemoteAddress || '?'}:${connection.RemotePort || '?'}`
      addNode({ id: `connection:${remote}:${index}`, name: remote, type: 'network', description: `${connection.protocol || 'TCP'} ${connection.state || connection.State || 'connection'}`, evidence: connection }, skillId, 'CONNECTED_TO')
    })
    ;(result.persistence || []).slice(0, 60).forEach((entry, index) => {
      const name = entry.path || entry.TaskName || entry.type || `entry ${index + 1}`
      addNode({ id: `persistence:${name}:${index}`, name, type: 'persistence', description: 'Startup or persistence mechanism discovered on the endpoint.', evidence: entry }, skillId, 'PERSISTS_VIA')
    })
    ;(result.files || []).slice(0, 60).forEach((file, index) => addNode({ id: `file:${file.path || index}`, name: file.path || `file ${index + 1}`, type: 'file', description: 'File integrity evidence with cryptographic digest.', evidence: file }, skillId, 'HASHED'))
    ;(result.packages || []).slice(0, 120).forEach((pkg, index) => {
      const id = `package:${pkg.name}:${pkg.version}:${index}`
      packageByName.set(String(pkg.name || '').toLowerCase(), id)
      addNode({ id, name: `${pkg.name} ${pkg.version || ''}`.trim(), type: 'package', description: `${pkg.publisher || pkg.ecosystem || pkg.source || 'Installed software'} package.`, evidence: pkg }, skillId, 'INSTALLED')
    })
    ;(result.services || []).slice(0, 80).forEach((service, index) => addNode({ id: `service:${service.name || service.Name}:${index}`, name: service.name || service.Name || `service ${index + 1}`, type: 'service', description: service.description || service.DisplayName || service.PathName || 'Operating-system service.', evidence: service }, skillId, 'HOSTS_SERVICE'))
    ;(result.neighbors || []).slice(0, 120).forEach((neighbor, index) => {
      const ip = neighbor.dst || neighbor.IPAddress || `neighbor ${index + 1}`
      const mac = neighbor.lladdr || neighbor.LinkLayerAddress || 'unresolved'
      addNode({ id: `neighbor:${ip}:${mac}`, name: `${ip} · ${mac}`, type: 'neighbor', description: 'Observed ARP or NDP neighbor binding.', evidence: neighbor }, skillId, 'RESOLVES_TO')
    })
    ;(result.routes || []).slice(0, 80).forEach((route, index) => addNode({ id: `route:${route.dst || route.DestinationPrefix}:${index}`, name: route.dst || route.DestinationPrefix || `route ${index + 1}`, type: 'route', description: `Route via ${route.gateway || route.NextHop || 'local interface'}.`, evidence: route }, skillId, 'ROUTES_VIA'))
    ;(result.interfaces || []).slice(0, 40).forEach((item, index) => addNode({ id: `interface:${item.ifname || item.Name}:${index}`, name: item.ifname || item.Name || `interface ${index + 1}`, type: 'interface', description: item.InterfaceDescription || item.operstate || item.Status || 'Network interface.', evidence: item }, skillId, 'EXPOSES'))
    ;(result.findings || []).slice(0, 100).forEach((finding, index) => addNode({ id: `finding:${finding.type || 'finding'}:${index}`, name: String(finding.type || 'security finding').replaceAll('_', ' '), type: 'finding', severity: finding.severity || 'medium', description: finding.description || 'Passive security finding requiring investigation.', evidence: finding }, skillId, 'DETECTED'))
    Object.entries(result.checks || {}).forEach(([name, value]) => addNode({ id: `control:${name}`, name, type: 'control', description: 'Defensive posture check.', evidence: { value } }, skillId, 'CHECKED'))
    ;(result.vulnerabilities || []).forEach((vulnerability, index) => {
      const id = vulnerability.id || vulnerability.cves?.[0] || `advisory-${index}`
      const affectedPackage = packageByName.get(String(vulnerability.package || '').toLowerCase())
      addNode({ id: `vulnerability:${id}:${index}`, name: vulnerability.cves?.[0] || id, type: 'vulnerability', severity: severityFromVulnerability(vulnerability), description: vulnerability.summary || 'Published vulnerability affecting installed software.', evidence: vulnerability }, affectedPackage || skillId, affectedPackage ? 'HAS_VULNERABILITY' : 'AFFECTED_BY')
    })
  })
  return { nodes, links }
}

export default function EvidenceGraph({ skillResults = {}, storageKey = 'current' }) {
  const graph2DRef = useRef()
  const graph3DRef = useRef()
  const wrapperRef = useRef()
  const previousNodeCountRef = useRef(0)
  const fittedRendererRef = useRef({ two: false, three: false })
  const stableGraphRef = useRef({ signature: '', graph: { nodes: [], links: [] } })
  const rendererGraphsRef = useRef({ two: { nodes: [], links: [] }, three: { nodes: [], links: [] } })
  const [mode3d, setMode3d] = useState(true)
  const [labels, setLabels] = useState(false)
  const [query, setQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [selected, setSelected] = useState(null)
  const [editing, setEditing] = useState(false)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  const [size, setSize] = useState({ width: 900, height: 620 })
  const annotationKey = `securityclaw:graph-annotations:${storageKey}`
  const [annotations, setAnnotations] = useState(() => {
    try { return JSON.parse(localStorage.getItem(annotationKey) || '{}') } catch { return {} }
  })
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => {
      const next = { width: Math.max(620, Math.floor(entry.contentRect.width)), height: Math.max(520, Math.floor(entry.contentRect.height)) }
      setSize((current) => current.width === next.width && current.height === next.height ? current : next)
    })
    if (wrapperRef.current) observer.observe(wrapperRef.current)
    return () => observer.disconnect()
  }, [])
  useEffect(() => { localStorage.setItem(annotationKey, JSON.stringify(annotations)) }, [annotationKey, annotations])
  useEffect(() => {
    stableGraphRef.current = { signature: '', graph: { nodes: [], links: [] } }
    rendererGraphsRef.current = { two: { nodes: [], links: [] }, three: { nodes: [], links: [] } }
    fittedRendererRef.current = { two: false, three: false }
    previousNodeCountRef.current = 0
    setSelected(null)
    try { setAnnotations(JSON.parse(localStorage.getItem(annotationKey) || '{}')) } catch { setAnnotations({}) }
  }, [annotationKey])

  const evidenceSignature = JSON.stringify({ skillResults, annotations })
  const fullGraph = useMemo(() => {
    if (stableGraphRef.current.signature === evidenceSignature) return stableGraphRef.current.graph
    const next = buildEvidenceGraph(skillResults, annotations)
    const previousNodes = new Map(stableGraphRef.current.graph.nodes.map((node) => [node.id, node]))
    next.nodes = next.nodes.map((node) => {
      const previous = previousNodes.get(node.id)
      if (!previous) return node
      return {
        ...previous,
        ...node,
        x: previous.x,
        y: previous.y,
        z: previous.z,
        vx: previous.vx,
        vy: previous.vy,
        vz: previous.vz,
        fx: previous.fx,
        fy: previous.fy,
        fz: previous.fz,
      }
    })
    stableGraphRef.current = { signature: evidenceSignature, graph: next }
    return next
  }, [evidenceSignature])
  const types = useMemo(() => [...new Set(fullGraph.nodes.map((node) => node.type))].sort(), [fullGraph])
  const graph = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const visible = new Set(fullGraph.nodes.filter((node) => (typeFilter === 'all' || node.type === typeFilter || node.id === 'host') && (!needle || `${node.name} ${node.description} ${JSON.stringify(node.evidence)}`.toLowerCase().includes(needle))).map((node) => node.id))
    if (needle || typeFilter !== 'all') visible.add('host')
    const nodes = fullGraph.nodes.filter((node) => visible.has(node.id)).map((node) => ({ ...node }))
    const links = fullGraph.links.filter((link) => visible.has(typeof link.source === 'object' ? link.source.id : link.source) && visible.has(typeof link.target === 'object' ? link.target.id : link.target)).map((link) => ({ ...link, source: typeof link.source === 'object' ? link.source.id : link.source, target: typeof link.target === 'object' ? link.target.id : link.target }))
    return { nodes, links }
  }, [fullGraph, query, typeFilter])
  const rendererGraphs = useMemo(() => {
    const updateRenderer = (rendererKey) => {
      const previousNodes = new Map(rendererGraphsRef.current[rendererKey].nodes.map((node) => [node.id, node]))
      const next = {
        nodes: graph.nodes.map((node) => {
          const previous = previousNodes.get(node.id)
          return previous ? { ...node, x: previous.x, y: previous.y, z: previous.z, vx: previous.vx, vy: previous.vy, vz: previous.vz, fx: previous.fx, fy: previous.fy, fz: previous.fz } : { ...node }
        }),
        links: graph.links.map((link) => ({
          ...link,
          source: typeof link.source === 'object' ? link.source.id : link.source,
          target: typeof link.target === 'object' ? link.target.id : link.target,
        })),
      }
      rendererGraphsRef.current[rendererKey] = next
      return next
    }
    return { two: updateRenderer('two'), three: updateRenderer('three') }
  }, [graph])

  const activeGraphRef = () => mode3d ? graph3DRef.current : graph2DRef.current

  useEffect(() => {
    const firstEvidence = previousNodeCountRef.current <= 1 && graph.nodes.length > 1
    previousNodeCountRef.current = graph.nodes.length
    if (!firstEvidence) return undefined
    const timer = window.setTimeout(() => {
      graph2DRef.current?.zoomToFit?.(600, 70)
      graph3DRef.current?.zoomToFit?.(600, 70)
    }, 350)
    return () => window.clearTimeout(timer)
  }, [graph.nodes.length])

  useEffect(() => {
    const timer = window.setTimeout(() => activeGraphRef()?.zoomToFit?.(500, 70), 250)
    return () => window.clearTimeout(timer)
  }, [mode3d])

  const nodeColor = (node) => SEVERITY_COLORS[node.analystSeverity || node.severity] || COLORS[node.type] || '#94a3b8'
  const nodeLabel = (node) => `${node.name}\n${node.type}${node.severity ? ` · ${node.severity}` : ''}\n${node.description}`
  const focusNode = (node) => {
    setSelected(node)
    if (mode3d && node.x != null) {
      const distance = 110
      const ratio = 1 + distance / Math.hypot(node.x || 1, node.y || 1, node.z || 1)
      graph3DRef.current?.cameraPosition({ x: node.x * ratio, y: node.y * ratio, z: (node.z || 1) * ratio }, node, 900)
    } else {
      graph2DRef.current?.centerAt(node.x, node.y, 700)
      graph2DRef.current?.zoom(3.5, 700)
    }
  }
  const exportGraph = () => {
    const blob = new Blob([JSON.stringify({ ...fullGraph, annotations }, null, 2)], { type: 'application/json' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = 'securityclaw-evidence-graph.json'
    link.click()
    URL.revokeObjectURL(link.href)
  }
  const saveAnnotation = (form) => {
    const data = new FormData(form)
    setAnnotations((current) => ({ ...current, [selected.id]: { name: data.get('name'), severity: data.get('severity'), notes: data.get('notes') } }))
    setEditing(false)
  }
  const resetLayout = () => {
    const activeGraph = mode3d ? rendererGraphs.three : rendererGraphs.two
    activeGraph.nodes.forEach((node) => { node.fx = undefined; node.fy = undefined; node.fz = undefined })
    activeGraphRef()?.d3ReheatSimulation?.()
    window.setTimeout(() => activeGraphRef()?.zoomToFit?.(700, 60), 500)
  }

  if (fullGraph.nodes.length === 1) return <div className="p-6 text-sm text-dim">Run an endpoint investigation to generate an evidence graph.</div>
  const common = { width: size.width, height: size.height, nodeLabel, nodeColor, nodeVal: (node) => node.type === 'host' ? 8 : node.type === 'vulnerability' ? 6 : 3, linkLabel: (link) => link.type, linkColor: () => '#334155', linkDirectionalArrowLength: 4, linkDirectionalArrowRelPos: 1, onNodeClick: focusNode, backgroundColor: '#070d18', cooldownTicks: 120, cooldownTime: 4000, warmupTicks: 25, d3AlphaDecay: 0.06, d3VelocityDecay: 0.45 }

  return (
    <div className={`relative flex min-h-0 flex-1 overflow-hidden bg-[#070d18] ${fullscreen ? 'fixed inset-0 z-50' : ''}`}>
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex flex-wrap items-center gap-2 border-b border-border p-3">
          <div className="relative min-w-56 flex-1"><Search className="absolute left-3 top-2.5 h-4 w-4 text-dim" /><input className="input w-full pl-9" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search nodes, evidence or CVE…" /></div>
          <select className="input" value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)}><option value="all">All node types</option>{types.map((type) => <option key={type}>{type}</option>)}</select>
          <button className={`btn ${mode3d ? 'btn-primary' : ''}`} onClick={() => setMode3d((value) => !value)}><Box className="h-4 w-4" /> {mode3d ? '3D' : '2D'}</button>
          <button className={`btn ${labels ? 'btn-primary' : ''}`} onClick={() => setLabels((value) => !value)}><Tags className="h-4 w-4" /> Labels</button>
          <button className="btn" onClick={() => activeGraphRef()?.zoomToFit?.(700, 50)}><Focus className="h-4 w-4" /> Fit</button>
          <button className="btn" onClick={resetLayout}><RotateCcw className="h-4 w-4" /> Reset</button>
          <button className="btn" onClick={() => setDetailsOpen((value) => !value)}>{detailsOpen ? <PanelRightClose className="h-4 w-4" /> : <PanelRightOpen className="h-4 w-4" />} Details</button>
          <button className="btn" onClick={() => setFullscreen((value) => !value)}>{fullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />} {fullscreen ? 'Exit' : 'Expand'}</button>
          <button className="btn" onClick={exportGraph}><Download className="h-4 w-4" /> JSON</button>
        </div>
        <div ref={wrapperRef} className="relative min-h-[520px] flex-1 overflow-hidden">
          <div className={`absolute inset-0 ${mode3d ? 'visible pointer-events-auto' : 'invisible pointer-events-none'}`} aria-hidden={!mode3d}>
            <Suspense fallback={<div className="p-6 text-sm text-dim">Loading 3D renderer…</div>}><ForceGraph3D ref={graph3DRef} {...common} graphData={rendererGraphs.three} onNodeDragEnd={(node) => { node.fx = node.x; node.fy = node.y; node.fz = node.z }} onEngineStop={() => { const controls = graph3DRef.current?.controls?.(); if (controls) controls.autoRotate = false; if (!fittedRendererRef.current.three) { fittedRendererRef.current.three = true; graph3DRef.current?.zoomToFit?.(600, 70) } }} /></Suspense>
          </div>
          <div className={`absolute inset-0 ${mode3d ? 'invisible pointer-events-none' : 'visible pointer-events-auto'}`} aria-hidden={mode3d}>
            <ForceGraph2D ref={graph2DRef} {...common} graphData={rendererGraphs.two} onNodeDragEnd={(node) => { node.fx = node.x; node.fy = node.y }} nodeCanvasObjectMode={() => 'after'} nodeCanvasObject={(node, context, scale) => { if (!labels && rendererGraphs.two.nodes.length > 24 && selected?.id !== node.id) return; context.font = `${Math.max(10 / scale, 2)}px sans-serif`; context.fillStyle = '#cbd5e1'; context.textAlign = 'center'; context.fillText(node.name.slice(0, 34), node.x, node.y + 8) }} />
          </div>
        </div>
      </div>
      {detailsOpen ? <aside className="absolute inset-y-0 right-0 z-20 w-80 overflow-auto border-l border-border bg-panel2/95 p-4 shadow-2xl backdrop-blur">
        {!selected ? <div className="text-sm text-dim">Select a node to inspect its evidence and add analyst annotations.</div> : <>
          <div className="flex items-start justify-between gap-2"><div><span className="badge badge-green">{selected.type}</span><h3 className="mt-2 break-words text-lg font-semibold text-text">{selected.name}</h3></div><button className="btn" onClick={() => setEditing((value) => !value)}><Pencil className="h-4 w-4" /></button></div>
          <p className="mt-3 text-sm text-dim">{selected.description}</p>
          {editing ? <form className="mt-4 space-y-3" onSubmit={(event) => { event.preventDefault(); saveAnnotation(event.currentTarget) }}><input className="input w-full" name="name" defaultValue={selected.name} /><select className="input w-full" name="severity" defaultValue={selected.analystSeverity || selected.severity || 'info'}>{['critical', 'high', 'medium', 'low', 'info'].map((value) => <option key={value}>{value}</option>)}</select><textarea className="input min-h-28 w-full" name="notes" defaultValue={selected.notes || ''} placeholder="Analyst notes" /><button className="btn btn-primary" type="submit">Save annotation</button></form> : null}
          {selected.notes ? <div className="mt-4 rounded-lg border border-cyan/20 bg-cyan/5 p-3 text-sm text-text"><div className="mb-1 font-mono text-[10px] uppercase text-cyan">Analyst notes</div>{selected.notes}</div> : null}
          <div className="mt-4 font-mono text-[10px] uppercase tracking-wider text-dim">Collected evidence</div><pre className="mt-2 max-h-[480px] overflow-auto whitespace-pre-wrap break-all rounded-lg bg-black/30 p-3 text-xs text-dim">{JSON.stringify(selected.evidence, null, 2)}</pre>
        </>}
      </aside> : null}
    </div>
  )
}
