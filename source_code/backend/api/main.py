"""FastAPI app. Run (from source_code/):  uv run uvicorn api.main:app --reload --port 8080"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.config import get_settings
from rag.ingest import collect_files, ingest_file
from rag.store import get_store

from .routes_kb import router as kb_router

log = logging.getLogger("rag.api")


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        auto_ingest()
    except Exception as e:  # keep serving; /api/health shows what is down
        log.error("auto-ingest skipped: %s", e)
    yield


def create_app(*, run_lifespan: bool = True) -> FastAPI:
    app = FastAPI(title="Local RAG — CS Handbook", lifespan=lifespan if run_lifespan else None)
    app.include_router(kb_router)
    return app


app = create_app()
