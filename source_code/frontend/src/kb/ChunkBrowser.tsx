import { useEffect, useState } from 'react'
import { api, type Chunk } from '../lib/api'

const PAGE_SIZE = 20

export default function ChunkBrowser({ source, refreshKey }: { source: string; refreshKey: number }) {
  const [page, setPage] = useState('')
  const [contains, setContains] = useState('')
  const [query, setQuery] = useState({ page: '', contains: '' })
  const [offset, setOffset] = useState(0)
  const [data, setData] = useState<{ total: number; items: Chunk[] } | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setOffset(0)
  }, [source, query])

  useEffect(() => {
    let cancelled = false
    api
      .chunks({
        source: source || undefined,
        page: query.page ? Number(query.page) : undefined,
        contains: query.contains || undefined,
        limit: PAGE_SIZE,
        offset,
      })
      .then((r) => !cancelled && (setData(r), setError('')))
      .catch((e) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [source, query, offset, refreshKey])

  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1
  const current = Math.floor(offset / PAGE_SIZE) + 1

  return (
    <div className="space-y-3">
      <form
        onSubmit={(e) => {
          e.preventDefault()
          setQuery({ page, contains })
        }}
        className="flex flex-wrap items-end gap-2 text-sm"
      >
        <label className="flex flex-col text-xs text-slate-500">
          Page
          <input
            value={page}
            onChange={(e) => setPage(e.target.value.replace(/\D/g, ''))}
            placeholder="any"
            className="w-20 rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-900"
          />
        </label>
        <label className="flex flex-1 flex-col text-xs text-slate-500">
          Text contains (case-sensitive)
          <input
            value={contains}
            onChange={(e) => setContains(e.target.value)}
            placeholder="e.g. CS311, GPA, Capstone"
            className="rounded-md border border-slate-300 px-2 py-1 text-sm text-slate-900"
          />
        </label>
        <button className="rounded-md bg-slate-800 px-3 py-1.5 text-white hover:bg-slate-900">Filter</button>
        {(query.page || query.contains) && (
          <button
            type="button"
            onClick={() => {
              setPage('')
              setContains('')
              setQuery({ page: '', contains: '' })
            }}
            className="px-2 py-1.5 text-slate-500 hover:text-slate-900"
          >
            Reset
          </button>
        )}
      </form>

      {error && <p className="rounded-md bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>}

      {data && (
        <>
          <p className="text-xs text-slate-500">
            {data.total} chunk{data.total === 1 ? '' : 's'}
            {source ? ` in ${source}` : ''} · showing {data.items.length ? offset + 1 : 0}–{offset + data.items.length}
          </p>
          <ol className="space-y-2">
            {data.items.map((c) => (
              <li key={c.id} className="rounded-lg border border-slate-200">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-slate-100 bg-slate-50 px-3 py-1.5 text-xs text-slate-600">
                  <span className="font-medium text-slate-800">p.{c.page} #{c.chunk_index}</span>
                  <span className="min-w-0 flex-1 truncate" title={c.section ? `${c.section} › ${c.heading}` : c.heading}>
                    {c.heading || '—'}
                  </span>
                  {!source && <span className="text-slate-400">{c.source}</span>}
                  <span className="tabular-nums">{c.char_len} chars</span>
                  {c.edited && <span className="rounded bg-amber-100 px-1.5 text-amber-800">edited</span>}
                  <code className="text-[10px] text-slate-400" title={c.id}>
                    {c.id.slice(0, 8)}
                  </code>
                </div>
                <pre className="max-h-64 overflow-auto whitespace-pre-wrap px-3 py-2 font-sans text-[12px] leading-relaxed text-slate-800">{c.text}</pre>
              </li>
            ))}
          </ol>
          {pages > 1 && (
            <div className="flex items-center justify-center gap-3 text-sm">
              <button
                disabled={current <= 1}
                onClick={() => setOffset(offset - PAGE_SIZE)}
                className="rounded-md border border-slate-300 px-3 py-1 disabled:opacity-40"
              >
                ← Prev
              </button>
              <span className="text-slate-500">
                {current} / {pages}
              </span>
              <button
                disabled={current >= pages}
                onClick={() => setOffset(offset + PAGE_SIZE)}
                className="rounded-md border border-slate-300 px-3 py-1 disabled:opacity-40"
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
