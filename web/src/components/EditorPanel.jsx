export default function EditorPanel({ title, subtitle, value, onChange, onSave, saving, rows = 18 }) {
  return (
    <div className="panel p-4">
      <div className="mb-3 flex items-start justify-between gap-4">
        <div>
          <div className="font-mono text-xs uppercase tracking-[0.16em] text-cyan">{title}</div>
          {subtitle ? <div className="mt-1 text-sm text-dim">{subtitle}</div> : null}
        </div>
        {onSave ? (
          <button className="btn btn-primary" onClick={onSave} disabled={saving}>
            {saving ? 'SAVING' : 'SAVE'}
          </button>
        ) : null}
      </div>
      <textarea className="textarea max-h-[calc(100vh-16rem)] overflow-auto font-mono text-xs" rows={rows} value={value} onChange={(e) => onChange(e.target.value)} spellCheck={false} />
    </div>
  )
}
