import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type DocumentInfo } from '../lib/api'
import ChunkBrowser from './ChunkBrowser'
import RetrievalTester from './RetrievalTester'

export default function KnowledgeBasePage() {
  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [selected, setSelected] = useState<string>('')
  const [view, setView] = useState<'chunks' | 'tester'>('chunks')
  const [status, setStatus] = useState<{ kind: 'info' | 'error'; text: string } | null>(null)
  const [uploading, setUploading] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)
  const fileRef = useRef<HTMLInputElement>(null)

  const refresh = useCallback(() => {
    api
      .documents()
      .then((r) => {
        setDocs(r.documents)
        setSelected((cur) => (cur && r.documents.some((d) => d.source === cur) ? cur : ''))
      })
      .catch((e) => setStatus({ kind: 'error', text: `Could not load documents: ${e.message}` }))
    setRefreshKey((k) => k + 1)
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  async function upload(file: File) {
    setUploading(true)
    setStatus({ kind: 'info', text: `Indexing ${file.name}… (PDF parsing can take ~30 s)` })
    try {
      const r = await api.uploadDocument(file)
      setStatus({
        kind: 'info',
        text: r.status === 'unchanged' ? `${r.source} is already indexed (unchanged).` : `${r.source}: ${r.status}, ${r.pages} pages → ${r.chunks} chunks.`,
      })
      refresh()
      setSelected(r.source)
    } catch (e) {
      setStatus({ kind: 'error', text: (e as Error).message })
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  async function remove(source: string) {
    if (!window.confirm(`Remove "${source}" and all its chunks from the knowledge base?`)) return
    try {
      const r = await api.deleteDocument(source)
      setStatus({ kind: 'info', text: `Removed ${source} (${r.deleted_chunks} chunks).` })
      refresh()
    } catch (e) {
      setStatus({ kind: 'error', text: (e as Error).message })
    }
  }

  const total = docs.reduce((n, d) => n + d.chunks, 0)

  return (
    <div className="mx-auto grid h-full max-w-6xl gap-4 overflow-hidden px-4 py-4 md:grid-cols-[260px_1fr]">
      <aside className="flex min-h-0 flex-col gap-3">
        <div className="rounded-xl border border-slate-200 bg-white p-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Documents</h2>
            <span className="text-xs text-slate-500">{total} chunks</span>
          </div>
          <ul className="mt-2 space-y-1 text-sm">
            <li>
              <button
                onClick={() => setSelected('')}
                className={`w-full rounded-md px-2 py-1.5 text-left ${selected === '' ? 'bg-blue-50 text-blue-800' : 'hover:bg-slate-50'}`}
              >
                All documents
              </button>
            </li>
            {docs.map((d) => (
              <li key={d.source} className="group flex items-center gap-1">
                <button
                  onClick={() => setSelected(d.source)}
                  className={`min-w-0 flex-1 rounded-md px-2 py-1.5 text-left ${selected === d.source ? 'bg-blue-50 text-blue-800' : 'hover:bg-slate-50'}`}
                >
                  <span className="block truncate">{d.source}</span>
                  <span className="text-xs text-slate-500">
                    {d.pages} pages · {d.chunks} chunks
                  </span>
                </button>
                <button
                  onClick={() => remove(d.source)}
                  title="Remove from knowledge base"
                  className="rounded p-1 text-slate-400 opacity-0 hover:bg-red-50 hover:text-red-600 group-hover:opacity-100"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.md,.txt"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
          />
          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            className="mt-3 w-full rounded-lg border border-dashed border-slate-300 px-3 py-2 text-sm text-slate-600 hover:border-blue-400 hover:text-blue-700 disabled:opacity-50"
          >
            {uploading ? 'Indexing…' : '+ Upload PDF / Markdown / text'}
          </button>
        </div>
        {status && (
          <p className={`rounded-lg px-3 py-2 text-xs ${status.kind === 'error' ? 'bg-red-50 text-red-700' : 'bg-blue-50 text-blue-800'}`}>
            {status.text}
          </p>
        )}
      </aside>

      <section className="flex min-h-0 flex-col rounded-xl border border-slate-200 bg-white">
        <div className="flex gap-1 border-b border-slate-200 px-3 pt-2 text-sm">
          {(
            [
              ['chunks', 'Chunks'],
              ['tester', 'Retrieval tester'],
            ] as const
          ).map(([v, label]) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`-mb-px border-b-2 px-3 py-2 ${view === v ? 'border-blue-700 font-medium text-blue-800' : 'border-transparent text-slate-600 hover:text-slate-900'}`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          {view === 'chunks' ? <ChunkBrowser source={selected} refreshKey={refreshKey} /> : <RetrievalTester />}
        </div>
      </section>
    </div>
  )
}
