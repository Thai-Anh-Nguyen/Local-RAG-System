// Typed client for the FastAPI backend (same-origin /api; proxied by Vite in dev, nginx in Docker).

export type Source = {
  n: number
  id: string
  source: string
  page: number | null
  chunk_index: number | null
  section: string
  heading: string
  char_len: number
  edited: boolean
  text: string
  score: number
  dense_rank: number | null
  bm25_rank: number | null
  rrf: number | null
}

export type Chunk = Omit<Source, 'n' | 'score' | 'dense_rank' | 'bm25_rank' | 'rrf'>

export type DocumentInfo = { source: string; chunks: number; pages: number; ingested_at: number | null }

export type Health = {
  status: 'ok' | 'degraded'
  ollama: { reachable: boolean; url: string }
  chroma: { reachable: boolean; collection: string; chunks: number | null }
  models: { llm: string; llm_available: boolean; embed: string; embed_available: boolean }
}

export type RetrieveResult = {
  question: string
  mode: 'hybrid' | 'dense'
  threshold: number
  top_score: number
  would_refuse: boolean
  hits: Source[]
}

export type Turn = { role: 'user' | 'assistant'; content: string }

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* not JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => fetch('/api/health').then(json<Health>),
  models: () => fetch('/api/models').then(json<{ default: string; available: string[] }>),
  documents: () => fetch('/api/documents').then(json<{ documents: DocumentInfo[] }>),
  uploadDocument: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return fetch('/api/documents', { method: 'POST', body: form }).then(
      json<{ source: string; status: string; chunks: number; pages: number }>,
    )
  },
  deleteDocument: (source: string) =>
    fetch(`/api/documents/${encodeURIComponent(source)}`, { method: 'DELETE' }).then(json<{ deleted_chunks: number }>),
  chunks: (params: { source?: string; page?: number; contains?: string; limit?: number; offset?: number }) => {
    const q = new URLSearchParams()
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') q.set(k, String(v))
    return fetch(`/api/chunks?${q}`).then(json<{ total: number; limit: number; offset: number; items: Chunk[] }>)
  },
  retrieve: (question: string, k?: number, mode?: 'hybrid' | 'dense') =>
    fetch('/api/retrieve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, k, mode }),
    }).then(json<RetrieveResult>),
}

// ---- streaming chat (SSE over POST, parsed manually since EventSource is GET-only) ----
export type ChatEvent =
  | { event: 'sources'; data: { model: string; mode: string; top_score: number; threshold: number; sources: Source[] } }
  | { event: 'token'; data: { text: string } }
  | { event: 'done'; data: { model: string; elapsed_ms: number } }
  | { event: 'refusal'; data: { message: string; top_score: number; threshold: number; sources: Source[] } }
  | { event: 'error'; data: { message: string } }

export async function streamChat(
  body: { question: string; history: Turn[]; model?: string },
  onEvent: (e: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    await json(res)
    return
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let event = ''
      const data: string[] = []
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
      }
      if (event && data.length) onEvent({ event, data: JSON.parse(data.join('\n')) } as ChatEvent)
    }
  }
}
