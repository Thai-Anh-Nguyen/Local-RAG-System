"""FastAPI app. Run (from source_code/):  uv run uvicorn api.main:app --reload --port 8080"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.config import get_settings
from rag.ingest import collect_files, ingest_file
from rag.store import get_store

from .routes_chat import router as chat_router
from .routes_kb import router as kb_router

log = logging.getLogger("rag.api")

# Shared with /api/health so the UI can show "indexing…" / failures on first start.
INGEST_STATE: dict = {"state": "pending", "error": None}

RETRY_SECONDS = 10
MAX_ATTEMPTS = 60  # ~10 minutes: covers Ollama/Chroma still starting or models still pulling


def auto_ingest() -> None:
    """Index the bundled handbook(s) when the collection is empty (first run)."""
    settings = get_settings()
    store = get_store()
    if store.count() > 0:
        return
    files = collect_files([settings.data_path]) if settings.data_path.exists() else []
    for f in files:
        r = ingest_file(f, store, settings)
        log.warning("auto-ingest: %s %s (%d chunks)", r.status, r.source, r.chunks)


def auto_ingest_with_retry() -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        INGEST_STATE.update(state="running", error=None)
        try:
            auto_ingest()
            INGEST_STATE.update(state="done", error=None)
            return
        except Exception as e:  # Ollama/Chroma not ready yet: keep serving, retry in the background
            INGEST_STATE.update(state="retrying", error=f"{type(e).__name__}: {e}")
            log.warning("auto-ingest attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, e)
            time.sleep(RETRY_SECONDS)
    INGEST_STATE["state"] = "failed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=auto_ingest_with_retry, name="auto-ingest", daemon=True).start()
    yield


def create_app(*, run_lifespan: bool = True) -> FastAPI:
    app = FastAPI(title="Local RAG — CS Handbook", lifespan=lifespan if run_lifespan else None)
    app.include_router(kb_router)
    app.include_router(chat_router)
    return app


app = create_app()
