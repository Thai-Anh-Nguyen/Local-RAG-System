import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Source } from '../lib/api'
import SourceCard from './SourceCard'

export type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  status: 'streaming' | 'done' | 'refused' | 'error'
  model?: string
  sources?: Source[]
  topScore?: number
  threshold?: number
  elapsedMs?: number
  error?: string
}

/** Turn "[2]" / "[1][3]" citation markers into links the renderer swaps for chips. */
const linkCitations = (text: string) => text.replace(/\[(\d{1,2})\](?!\()/g, '[$1](#cite-$1)')

export default function MessageView({ message: m }: { message: Message }) {
  const [open, setOpen] = useState<number | null>(null)

  if (m.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-blue-700 px-4 py-2 text-sm text-white">
          {m.content}
        </div>
      </div>
    )
  }

  const cited = new Set([...m.content.matchAll(/\[(\d{1,2})\]/g)].map((x) => Number(x[1])))

  return (
    <div className="space-y-2">
      <div
        className={`rounded-2xl rounded-bl-sm border px-4 py-3 text-sm ${
          m.status === 'refused' ? 'border-amber-200 bg-amber-50' : m.status === 'error' ? 'border-red-200 bg-red-50' : 'border-slate-200 bg-white'
        }`}
      >
        {m.content ? (
          <div className="prose prose-sm prose-slate max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({ href, children }) => {
                  const cite = href?.startsWith('#cite-') ? Number(href.slice(6)) : null
                  if (cite === null) return <a href={href} target="_blank" rel="noreferrer">{children}</a>
                  const known = m.sources?.some((s) => s.n === cite)
                  return (
                    <button
                      type="button"
                      onClick={() => setOpen(open === cite ? null : cite)}
                      title={known ? 'Show source' : 'No such source'}
                      className={`mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded px-1 align-text-top text-[11px] font-semibold no-underline ${
                        known ? 'bg-blue-100 text-blue-800 hover:bg-blue-200' : 'bg-slate-100 text-slate-400'
                      }`}
                    >
                      {cite}
                    </button>
                  )
                },
              }}
            >
              {linkCitations(m.content)}
            </ReactMarkdown>
          </div>
        ) : m.status === 'streaming' ? (
          <span className="inline-flex items-center gap-1 text-slate-400">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:120ms]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:240ms]" />
            <span className="ml-1 text-xs">{m.sources ? 'Writing answer…' : 'Searching the handbook…'}</span>
          </span>
        ) : null}
        {m.status === 'error' && <p className="mt-1 text-xs text-red-700">⚠ {m.error}</p>}
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-1 text-xs text-slate-500">
        {m.model && <span>{m.model}</span>}
        {m.topScore !== undefined && (
          <span title="Best cosine similarity between the question and a retrieved chunk. Below the threshold the question is declined without calling the LLM. This measures retrieval, not answer correctness.">
            retrieval relevance {m.topScore.toFixed(2)} / threshold {m.threshold?.toFixed(2)}
          </span>
        )}
        {m.elapsedMs !== undefined && <span>{(m.elapsedMs / 1000).toFixed(1)}s</span>}
      </div>

      {m.sources && m.sources.length > 0 && (
        <details open={open !== null} className="px-1">
          <summary className="cursor-pointer select-none text-xs font-medium text-slate-600 hover:text-slate-900">
            {m.status === 'refused' ? 'Closest chunks (not relevant enough)' : `Sources (${m.sources.length})`}
          </summary>
          <div className="mt-2 space-y-1.5">
            {m.sources.map((s) => (
              <SourceCard
                key={s.id}
                source={s}
                cited={cited.has(s.n)}
                expanded={open === s.n}
                onToggle={() => setOpen(open === s.n ? null : s.n)}
              />
            ))}
          </div>
        </details>
      )}
    </div>
  )
}
