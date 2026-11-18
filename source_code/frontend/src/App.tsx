import { useEffect, useState } from 'react'
import ChatPage from './chat/ChatPage'
import KnowledgeBasePage from './kb/KnowledgeBasePage'
import { api, type Health } from './lib/api'
import { load, save } from './lib/storage'

type Tab = 'chat' | 'kb'

const tabFromHash = (): Tab => (window.location.hash === '#/kb' ? 'kb' : 'chat')

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash)
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState(false)
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState<string>(() => load('rag.model', ''))

  useEffect(() => {
    const onHash = () => setTab(tabFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    const check = () =>
      api
        .health()
        .then((h) => {
          setHealth(h)
          setHealthError(false)
        })
        .catch(() => setHealthError(true))
    check()
    const id = setInterval(check, 15000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    api
      .models()
      .then((m) => {
        setModels(m.available)
        setModel((current) => (m.available.includes(current) ? current : m.default))
      })
      .catch(() => setModels([]))
  }, [health?.ollama.reachable])

  useEffect(() => {
    save('rag.model', model)
  }, [model])

  const go = (t: Tab) => {
    window.location.hash = t === 'kb' ? '#/kb' : '#/'
    setTab(t)
  }

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <div className="flex items-center gap-2">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <div>
              <h1 className="text-sm font-semibold leading-tight">Handbook Assistant</h1>
              <p className="text-xs text-slate-500">B.S. Computer Science · local RAG</p>
            </div>
          </div>
          <nav className="flex gap-1 rounded-lg bg-slate-100 p-1 text-sm">
            {(['chat', 'kb'] as const).map((t) => (
              <button
                key={t}
                onClick={() => go(t)}
                className={`rounded-md px-3 py-1 ${tab === t ? 'bg-white font-medium shadow-sm' : 'text-slate-600 hover:text-slate-900'}`}
              >
                {t === 'chat' ? 'Chat' : 'Knowledge base'}
              </button>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-4 text-sm">
            <label className="flex items-center gap-2">
              <span className="text-slate-500">Model</span>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={models.length === 0}
                className="rounded-md border border-slate-300 bg-white px-2 py-1"
              >
                {models.length === 0 && <option value="">unavailable</option>}
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>
            <HealthBadge health={health} error={healthError} />
          </div>
        </div>
      </header>
      <main className="min-h-0 flex-1">
        {tab === 'chat' ? <ChatPage model={model} /> : <KnowledgeBasePage />}
      </main>
    </div>
  )
}

function HealthBadge({ health, error }: { health: Health | null; error: boolean }) {
  const ok = health?.status === 'ok' && !error
  const label = error ? 'API offline' : !health ? 'Checking…' : ok ? 'Ready' : 'Degraded'
  const detail = health
    ? [
        `Ollama: ${health.ollama.reachable ? 'up' : 'down'} (${health.ollama.url})`,
        `LLM ${health.models.llm}: ${health.models.llm_available ? 'pulled' : 'missing'}`,
        `Embeddings ${health.models.embed}: ${health.models.embed_available ? 'pulled' : 'missing'}`,
        `Chroma: ${health.chroma.reachable ? `${health.chroma.chunks} chunks` : 'down'}`,
      ].join('\n')
    : 'Backend not reachable'
  return (
    <span title={detail} className="flex cursor-help items-center gap-1.5 text-slate-600">
      <span className={`h-2.5 w-2.5 rounded-full ${ok ? 'bg-emerald-500' : error ? 'bg-red-500' : 'bg-amber-500'}`} />
      {label}
    </span>
  )
}
