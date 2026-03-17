import { useEffect, useMemo, useState } from 'react'
import { api } from '../lib/api.js'
import PageHeader from '../components/PageHeader.jsx'
import EditorPanel from '../components/EditorPanel.jsx'

export default function SkillsPage() {
  const [skills, setSkills] = useState([])
  const [selected, setSelected] = useState(null)
  const [manifestRaw, setManifestRaw] = useState('')
  const [instructionRaw, setInstructionRaw] = useState('')
  const [savingManifest, setSavingManifest] = useState(false)
  const [savingInstruction, setSavingInstruction] = useState(false)
  const [savingToggle, setSavingToggle] = useState(false)

  const loadSkills = async () => {
    const res = await api.get('/api/skills')
    const items = res.data.items || []
    setSkills(items)
    if (!selected && items.length) setSelected(items[0].name)
  }

  useEffect(() => { loadSkills() }, [])

  const selectedSkill = useMemo(() => skills.find((skill) => skill.name === selected), [skills, selected])

  useEffect(() => {
    if (!selectedSkill) return
    setManifestRaw(selectedSkill.manifest_raw || '')
    setInstructionRaw(selectedSkill.instruction_raw || '')
  }, [selectedSkill])

  const saveManifest = async () => {
    if (!selected) return
    setSavingManifest(true)
    try {
      await api.put(`/api/skills/${selected}/manifest`, { content: manifestRaw })
      await loadSkills()
    } finally {
      setSavingManifest(false)
    }
  }

  const saveInstruction = async () => {
    if (!selected) return
    setSavingInstruction(true)
    try {
      await api.put(`/api/skills/${selected}/instruction`, { content: instructionRaw })
      await loadSkills()
    } finally {
      setSavingInstruction(false)
    }
  }

  const toggleSkill = async (enabled) => {
    if (!selected) return
    setSavingToggle(true)
    try {
      await api.put(`/api/skills/${selected}/enabled`, { enabled })
      await loadSkills()
    } finally {
      setSavingToggle(false)
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Skills" subtitle="Inspect and edit every skill's manifest and instruction prompt." />

      <div className="grid gap-6 xl:grid-cols-[320px_1fr]">
        <div className="panel overflow-hidden">
          <div className="border-b border-border px-4 py-3 font-mono text-xs uppercase tracking-[0.18em] text-cyan">Loaded Skills</div>
          <div className="space-y-2 p-3">
            {skills.length === 0 ? <div className="rounded-xl border border-border bg-panel2 p-3 font-mono text-xs text-dim">No skills discovered.</div> : null}
            {skills.map((skill) => (
              <button
                key={skill.name}
                className={`w-full rounded-xl border p-3 text-left ${selected === skill.name ? 'border-cyan bg-cyan/10' : 'border-border bg-panel2'} ${skill.enabled ? '' : 'opacity-70'}`}
                onClick={() => setSelected(skill.name)}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="font-mono text-xs uppercase tracking-[0.14em] text-cyan">{skill.name}</div>
                  <span className={`badge ${skill.enabled ? 'badge-green' : 'badge-dim'}`}>{skill.enabled ? 'active' : 'disabled'}</span>
                </div>
                <div className="mt-2 text-sm text-text">{skill.description}</div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {skill.schedule_cron_expr ? <span className="badge badge-amber">cron</span> : skill.schedule_interval_seconds !== null && skill.schedule_interval_seconds !== undefined ? <span className="badge badge-green">every {skill.schedule_interval_seconds}s</span> : <span className="badge badge-dim">manual</span>}
                </div>
              </button>
            ))}
          </div>
        </div>

        {selectedSkill ? (
          <div className="space-y-6">
            <div className="panel p-4">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <div className="font-mono text-xs uppercase tracking-[0.18em] text-cyan">Skill Metadata</div>
                  <div className="mt-1 text-lg font-semibold text-text">{selectedSkill.name}</div>
                </div>
                <div className="flex gap-2">
                  <button
                    className={selectedSkill.enabled ? 'btn btn-danger' : 'btn btn-primary'}
                    onClick={() => toggleSkill(!selectedSkill.enabled)}
                    disabled={savingToggle}
                  >
                    {savingToggle ? 'UPDATING' : selectedSkill.enabled ? 'DISABLE SKILL' : 'ENABLE SKILL'}
                  </button>
                  {selectedSkill.schedule_cron_expr ? <span className="badge badge-amber">{selectedSkill.schedule_cron_expr}</span> : selectedSkill.schedule_interval_seconds !== null && selectedSkill.schedule_interval_seconds !== undefined ? <span className="badge badge-green">every {selectedSkill.schedule_interval_seconds}s</span> : <span className="badge badge-dim">manual</span>}
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.16em] text-dim">Description</div>
                  <div className="text-sm text-text">{selectedSkill.description}</div>
                </div>
                <div>
                  <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.16em] text-dim">Runtime State</div>
                  <div className="flex flex-wrap gap-2">
                    <span className={`badge ${selectedSkill.enabled ? 'badge-green' : 'badge-dim'}`}>{selectedSkill.enabled ? 'enabled in chat and service' : 'disabled in chat and service'}</span>
                    {(selectedSkill.required_env_vars || []).length ? selectedSkill.required_env_vars.map((item, idx) => <span key={idx} className="badge badge-dim">{item.name || item}</span>) : <span className="badge badge-green">no env vars</span>}
                  </div>
                </div>
              </div>
            </div>

            <EditorPanel title="manifest.yaml" subtitle="Capability metadata, restrictions, routing hints, and env requirements." value={manifestRaw} onChange={setManifestRaw} onSave={saveManifest} saving={savingManifest} rows={18} />
            <EditorPanel title="instruction.md" subtitle="LLM prompt plus scheduling frontmatter." value={instructionRaw} onChange={setInstructionRaw} onSave={saveInstruction} saving={savingInstruction} rows={20} />
          </div>
        ) : null}
      </div>
    </div>
  )
}
