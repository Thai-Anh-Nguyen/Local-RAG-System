import { useState } from 'react'
import SourceCard from '../chat/SourceCard'
import { api, type RetrieveResult } from '../lib/api'

export default function RetrievalTester() {
  const [question, setQuestion] = useState('')
  const [k, setK] = useState(5)
  const [mode, setMode] = useState<'hybrid' | 'dense'>('hybrid')
  const [result, setResult] = useState<RetrieveResult | null>(null)
  const [open, setOpen] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function run(e: React.FormEvent) {
    e.preventDefault()
    if (!question.trim()) return
    setLoading(true)
    setError('')
    try {
      setResult(await api.retrieve(question.trim(), k, mode))
      setOpen(null)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-slate-500">
        See exactly which chunks a question retrieves, before any LLM is involved. <b>Hybrid</b> fuses vector search with
        BM25 keyword search; <b>dense</b> is vector-only (the baseline).
      </p>
      <form onSubmit={run} className="flex flex-wrap items-end gap-2 text-sm">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. What are the prerequisites for CS311?"
          className="min-w-64 flex-1 rounded-md border border-slate-300 px-3 py-1.5"
        />
        <label className="flex items-center gap-1 text-xs text-slate-500">
          k
          <input
            type="number"
            min={1}
            max={20}
            value={k}
            onChange={(e) => setK(Math.min(20, Math.max(1, Number(e.target.value) || 1)))}
            className="w-14 rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-900"
          />
        </label>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as 'hybrid' | 'dense')}
          className="rounded-md border border-slate-300 bg-white px-2 py-1.5"
        >
          <option value="hybrid">hybrid</option>
          <option value="dense">dense</option>
        </select>
        <button disabled={loading} className="rounded-md bg-blue-700 px-4 py-1.5 font-medium text-white hover:bg-blue-800 disabled:opacity-50">
          {loading ? 'Searching…' : 'Retrieve'}
        </button>
      </form>

      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>}

      {result && (
        <>
          <div
            className={`rounded-lg px-3 py-2 text-xs ${result.would_refuse ? 'bg-amber-50 text-amber-900' : 'bg-emerald-50 text-emerald-900'}`}
          >
            Best relevance <b>{result.top_score.toFixed(3)}</b> vs threshold {result.threshold} →{' '}
            {result.would_refuse ? 'the chatbot would decline (LLM not called).' : 'the chatbot would answer from these chunks.'}
            <span className="ml-2 text-slate-500">mode: {result.mode}</span>
          </div>
          <div className="space-y-1.5">
            {result.hits.map((h) => (
              <SourceCard key={h.id} source={h} cited={false} expanded={open === h.n} onToggle={() => setOpen(open === h.n ? null : h.n)} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
