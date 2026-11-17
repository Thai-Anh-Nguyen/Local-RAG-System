"""Knowledge-base routes: health, models, retrieval inspector, documents, chunks."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from rag.config import Settings
from rag.ingest import ingest_file
from rag.parsing import SUPPORTED_SUFFIXES
from rag.retrieval import retrieve
from rag.store import KnowledgeStore

from .deps import settings_dep, store_dep
from .schemas import RetrieveRequest

router = APIRouter(prefix="/api")


async def ollama_models(settings: Settings) -> list[str] | None:
    """Names of models pulled in Ollama, or None if Ollama is unreachable."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_url}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
    except (httpx.HTTPError, ValueError):
        return None


def _has(models: list[str], name: str) -> bool:
    return name in models or f"{name}:latest" in models


@router.get("/health")
async def health(settings: Settings = Depends(settings_dep), store: KnowledgeStore = Depends(store_dep)):
    models = await ollama_models(settings)
    try:
        chunks = await run_in_threadpool(store.count)
        chroma_ok = True
    except Exception:
        chunks, chroma_ok = None, False
    ok = bool(chroma_ok and models is not None and _has(models, settings.embed_model)
              and _has(models, settings.llm_model))
    return {
        "status": "ok" if ok else "degraded",
        "ollama": {"reachable": models is not None, "url": settings.ollama_url},
        "chroma": {"reachable": chroma_ok, "collection": settings.collection, "chunks": chunks},
        "models": {
            "llm": settings.llm_model,
            "llm_available": models is not None and _has(models, settings.llm_model),
            "embed": settings.embed_model,
            "embed_available": models is not None and _has(models, settings.embed_model),
        },
    }


@router.get("/models")
async def models(settings: Settings = Depends(settings_dep)):
    pulled = await ollama_models(settings) or []
    return {
        "default": settings.llm_model,
        "available": [m for m in settings.allowed_models if _has(pulled, m)],
    }


@router.post("/retrieve")
async def retrieve_route(
    req: RetrieveRequest,
    settings: Settings = Depends(settings_dep),
    store: KnowledgeStore = Depends(store_dep),
):
    hits = await run_in_threadpool(retrieve, store, req.question, req.k or settings.top_k)
    top = hits[0].score if hits else 0.0
    return {
        "question": req.question,
        "threshold": settings.relevance_threshold,
        "top_score": round(top, 4),
        "would_refuse": top < settings.relevance_threshold,
        "hits": [h.as_dict() for h in hits],
    }


# ---- documents ---------------------------------------------------------------------------
@router.get("/documents")
async def list_documents(store: KnowledgeStore = Depends(store_dep)):
    return {"documents": await run_in_threadpool(store.list_documents)}


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


@router.post("/documents", status_code=201)
async def upload_document(
    file: UploadFile,
    settings: Settings = Depends(settings_dep),
    store: KnowledgeStore = Depends(store_dep),
):
    name = _SAFE_NAME.sub("_", Path(file.filename or "").name).strip()
    if not name or Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_SUFFIXES))}")

    # Keep uploads next to the handbooks so they are re-ingested if the collection is rebuilt.
    upload_dir = settings.data_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=upload_dir, suffix=".part", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
    tmp_path = Path(tmp.name)
    try:
        result = await run_in_threadpool(ingest_file, tmp_path, store, settings, source=name)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not ingest {name}: {e}") from e
    tmp_path.replace(upload_dir / name)
    return result.__dict__


@router.delete("/documents/{source}")
async def delete_document(
    source: str,
    settings: Settings = Depends(settings_dep),
    store: KnowledgeStore = Depends(store_dep),
):
    removed = await run_in_threadpool(store.delete_source, source)
    if not removed:
        raise HTTPException(404, f"No chunks for source '{source}'")
    (settings.data_path / "uploads" / Path(source).name).unlink(missing_ok=True)
    return {"source": source, "deleted_chunks": removed}


# ---- chunks ------------------------------------------------------------------------------
@router.get("/chunks")
async def list_chunks(
    source: str | None = None,
    page: int | None = Query(default=None, ge=1),
    contains: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    store: KnowledgeStore = Depends(store_dep),
):
    total, chunks = await run_in_threadpool(store.get_chunks, source, page, contains, limit, offset)
    return {"total": total, "limit": limit, "offset": offset, "items": [c.as_dict() for c in chunks]}


@router.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: str, store: KnowledgeStore = Depends(store_dep)):
    chunk = await run_in_threadpool(store.get_chunk, chunk_id)
    if chunk is None:
        raise HTTPException(404, "Chunk not found")
    return chunk.as_dict()
