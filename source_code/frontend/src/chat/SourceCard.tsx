import type { Source } from '../lib/api'

export function ScoreBar({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(1, score)) * 100
  return (
    <span className="inline-flex items-center gap-1.5" title={`cosine similarity ${score.toFixed(3)}`}>
      <span className="h-1.5 w-14 overflow-hidden rounded-full bg-slate-200">
        <span className="block h-full rounded-full bg-blue-600" style={{ width: `${pct}%` }} />
      </span>
      <span className="tabular-nums">{score.toFixed(2)}</span>
    </span>
  )
}

export default function SourceCard({
  source: s,
  cited,
  expanded,
  onToggle,
}: {
  source: Source
  cited: boolean
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <div className={`rounded-lg border bg-white text-xs ${expanded ? 'border-blue-300' : 'border-slate-200'}`}>
      <button type="button" onClick={onToggle} className="flex w-full items-center gap-2 px-3 py-2 text-left">
        <span
          className={`inline-flex h-5 min-w-5 items-center justify-center rounded px-1 font-semibold ${
            cited ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-500'
          }`}
        >
          {s.n}
        </span>
        <span className="min-w-0 flex-1 truncate">
          <span className="font-medium text-slate-800">{s.heading || s.section || 'Untitled section'}</span>
          <span className="text-slate-500"> · {s.source}{s.page ? `, p.${s.page}` : ''}</span>
        </span>
        <ScoreBar score={s.score} />
      </button>
      {expanded && (
        <div className="border-t border-slate-100 px-3 py-2">
          <pre className="max-h-72 overflow-auto whitespace-pre-wrap font-sans text-[12px] leading-relaxed text-slate-700">{s.text}</pre>
          <p className="mt-1 text-[11px] text-slate-400">
            chunk {s.id.slice(0, 10)} · dense #{s.dense_rank ?? '–'} · keyword #{s.bm25_rank ?? '–'}
          </p>
        </div>
      )}
    </div>
  )
}
