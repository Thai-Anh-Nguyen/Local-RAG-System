"""POST /api/chat — retrieval + grounded generation, streamed as Server-Sent Events.

Event sequence:
    sources  {model, mode, top_score, threshold, sources:[...]}
    token    {text}            (repeated)
    done     {model, elapsed_ms}
or a single
    refusal  {message, top_score, threshold, sources:[...]}   (retrieval below threshold, LLM not called)
or
    error    {message}
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from rag.config import Settings
from rag.generation import ChatModel, build_messages, make_llm, stream_answer
from rag.retrieval import Turn, retrieval_query, retrieve, top_score
from rag.store import KnowledgeStore

from .deps import settings_dep, store_dep
from .schemas import ChatRequest

router = APIRouter(prefix="/api")
log = logging.getLogger("rag.api")

SNIPPET_CHARS = 1200
REFUSAL = ("I couldn't find anything relevant to that in the handbook, so I won't guess. "
           "Try rephrasing, or ask about the CS program's requirements, courses, outcomes or policies.")


def llm_factory_dep() -> Callable[[Settings, str | None], ChatModel]:
    return make_llm


def _event(name: str, data: dict) -> dict:
    return {"event": name, "data": json.dumps(data, ensure_ascii=False)}


@router.post("/chat")
async def chat(
    req: ChatRequest,
    settings: Settings = Depends(settings_dep),
    store: KnowledgeStore = Depends(store_dep),
    llm_factory=Depends(llm_factory_dep),
):
    model = req.model or settings.llm_model
    if model not in settings.allowed_models:
        raise HTTPException(400, f"Model '{model}' is not allowed. Allowed: {settings.allowed_models}")
    history = [Turn(t.role, t.content) for t in req.history]

    async def events():
        started = time.perf_counter()
        try:
            query = retrieval_query(req.question, history)
            hits = await run_in_threadpool(retrieve, store, query, settings.top_k, settings.retrieval_mode)
            best = top_score(hits)
            sources = [h.as_dict(SNIPPET_CHARS) for h in hits]
            if best < settings.relevance_threshold:
                yield _event("refusal", {"message": REFUSAL, "top_score": round(best, 4),
                                         "threshold": settings.relevance_threshold, "sources": sources})
                return
            yield _event("sources", {"model": model, "mode": settings.retrieval_mode, "top_score": round(best, 4),
                                     "threshold": settings.relevance_threshold, "sources": sources})
            messages = build_messages(req.question, hits, history, settings.history_turns)
            async for text in stream_answer(llm_factory(settings, model), messages):
                yield _event("token", {"text": text})
            yield _event("done", {"model": model, "elapsed_ms": round((time.perf_counter() - started) * 1000)})
        except Exception as e:  # surface failures (Ollama down, model missing) to the UI
            log.exception("chat failed")
            yield _event("error", {"message": f"{type(e).__name__}: {e}"})

    return EventSourceResponse(events(), ping=15)
