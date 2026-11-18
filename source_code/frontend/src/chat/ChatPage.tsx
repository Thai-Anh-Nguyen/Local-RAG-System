import { useEffect, useRef, useState } from 'react'
import { streamChat, type Source, type Turn } from '../lib/api'
import { load, save } from '../lib/storage'
import MessageView, { type Message } from './MessageView'

const STORAGE_KEY = 'rag.chat.v1'
const HISTORY_PAIRS = 3

const SUGGESTIONS = [
  'How many credits are required for the CS major?',
  'What are the prerequisites for CS311?',
  'What is required to enroll in the Capstone?',
  'Which courses make up the CS minor?',
]

/** Last N completed question/answer pairs, sent so follow-up questions have context. */
function historyFrom(messages: Message[]): Turn[] {
  const turns: Turn[] = []
  for (let i = 0; i + 1 < messages.length; i++) {
    const q = messages[i]
    const a = messages[i + 1]
    if (q.role === 'user' && a.role === 'assistant' && a.status === 'done') {
      turns.push({ role: 'user', content: q.content }, { role: 'assistant', content: a.content })
    }
  }
  return turns.slice(-2 * HISTORY_PAIRS)
}

export default function ChatPage({ model }: { model: string }) {
  const [messages, setMessages] = useState<Message[]>(() =>
    load<Message[]>(STORAGE_KEY, []).map((m) => (m.status === 'streaming' ? { ...m, status: 'error', error: 'Interrupted' } : m)),
  )
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Effects use block bodies: newer Chromium returns a Promise from scrollIntoView(), and an
  // arrow that returns it would hand React a non-function "cleanup" and crash the app.
  useEffect(() => {
    save(STORAGE_KEY, messages)
  }, [messages])
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const updateLast = (patch: (m: Message) => Message) =>
    setMessages((ms) => [...ms.slice(0, -1), patch(ms[ms.length - 1])])

  async function ask(question: string) {
    question = question.trim()
    if (!question || busy) return
    const history = historyFrom(messages)
    const now = Date.now()
    setMessages((ms) => [
      ...ms,
      { id: `u${now}`, role: 'user', content: question, status: 'done' },
      { id: `a${now}`, role: 'assistant', content: '', status: 'streaming', model },
    ])
    setInput('')
    setBusy(true)
    const controller = new AbortController()
    abortRef.current = controller
    try {
      await streamChat(
        { question, history, model: model || undefined },
        (e) => {
          switch (e.event) {
            case 'sources':
              updateLast((m) => ({ ...m, sources: e.data.sources, topScore: e.data.top_score, threshold: e.data.threshold, model: e.data.model }))
              break
            case 'token':
              updateLast((m) => ({ ...m, content: m.content + e.data.text }))
              break
            case 'done':
              updateLast((m) => ({ ...m, status: 'done', elapsedMs: e.data.elapsed_ms }))
              break
            case 'refusal':
              updateLast((m) => ({
                ...m,
                status: 'refused',
                content: e.data.message,
                sources: e.data.sources as Source[],
                topScore: e.data.top_score,
                threshold: e.data.threshold,
              }))
              break
            case 'error':
              updateLast((m) => ({ ...m, status: 'error', error: e.data.message }))
              break
          }
        },
        controller.signal,
      )
      updateLast((m) => (m.status === 'streaming' ? { ...m, status: 'done' } : m))
    } catch (err) {
      const aborted = err instanceof DOMException && err.name === 'AbortError'
      updateLast((m) => ({ ...m, status: aborted && m.content ? 'done' : 'error', error: aborted ? 'Stopped' : String(err) }))
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col px-4">
      <div className="flex-1 overflow-y-auto py-6">
        {messages.length === 0 ? (
          <div className="mt-16 text-center">
            <h2 className="text-lg font-semibold">Ask about the major handbook</h2>
            <p className="mt-1 text-sm text-slate-500">
              Answers come only from the handbook, with numbered citations. Off-topic questions are declined.
            </p>
            <div className="mt-6 grid gap-2 sm:grid-cols-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => ask(s)}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-sm hover:border-blue-300 hover:bg-blue-50"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {messages.map((m) => (
              <MessageView key={m.id} message={m} />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          ask(input)
        }}
        className="sticky bottom-0 border-t border-slate-200 bg-slate-50 py-3"
      >
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                ask(input)
              }
            }}
            rows={1}
            placeholder="Ask a question about the handbook…  (Enter to send, Shift+Enter for a new line)"
            className="max-h-40 min-h-[42px] flex-1 resize-y rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />
          {busy ? (
            <button
              type="button"
              onClick={() => abortRef.current?.abort()}
              className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm hover:bg-slate-100"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="rounded-lg bg-blue-700 px-4 py-2 text-sm font-medium text-white hover:bg-blue-800 disabled:opacity-40"
            >
              Send
            </button>
          )}
        </div>
        <div className="mt-1.5 flex justify-between text-xs text-slate-500">
          <span>History is kept in this browser; the last {HISTORY_PAIRS} exchanges are sent for follow-ups.</span>
          {messages.length > 0 && !busy && (
            <button type="button" onClick={() => setMessages([])} className="hover:text-slate-800 hover:underline">
              Clear conversation
            </button>
          )}
        </div>
      </form>
    </div>
  )
}
