"""Retrieval: build the search query from the conversation and fetch scored chunks.

Two modes (RAG_RETRIEVAL_MODE):
- dense:  cosine similarity on nomic-embed-text vectors (baseline behaviour).
- hybrid: dense + BM25 keyword search, merged with Reciprocal Rank Fusion (RRF).
  BM25 rescues exact strings the embedding model blurs, such as course codes ("CS311")
  and course names ("Operating Systems").

In both modes every hit reports its cosine similarity, which drives the refusal threshold.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from .store import Chunk, KnowledgeStore

RRF_K = 60  # standard RRF constant
_TOKEN = re.compile(r"[a-z0-9]+")
_CODE_SPACE = re.compile(r"\b([a-z]{2,5})\s+(\d{3}|\dxx)\b")  # "cs 311" -> "cs311"
_STOP = set(
    "a an and are as at be by can do does for from how i in is it its me my of on or the to "
    "what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    text = _CODE_SPACE.sub(r"\1\2", text.lower())
    return [t for t in _TOKEN.findall(text) if t not in _STOP]


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class Hit:
    n: int  # citation number, 1-based
    chunk: Chunk
    score: float  # cosine similarity
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf: float | None = None

    def as_dict(self, snippet_chars: int | None = None) -> dict[str, Any]:
        d = self.chunk.as_dict()
        if snippet_chars is not None:
            d["text"] = d["text"][:snippet_chars]
        return {
            "n": self.n,
            "score": round(self.score, 4),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rrf": round(self.rrf, 5) if self.rrf is not None else None,
            **d,
        }


def retrieval_query(question: str, history: list[Turn]) -> str:
    """Current question plus the previous user question, so follow-ups like
    "what about its prerequisites?" still find the right section."""
    previous = next((t.content for t in reversed(history) if t.role == "user"), None)
    return f"{previous}\n{question}" if previous else question


class _BM25Index:
    """BM25 over all chunks, rebuilt when the collection changes (count-based check)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._count = -1
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def search(self, store: KnowledgeStore, query: str, n: int) -> list[Chunk]:
        with self._lock:
            count = store.count()
            if count != self._count or store.version != getattr(self, "_version", None):
                self._chunks = store.all_chunks()
                self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks]) if self._chunks else None
                self._count, self._version = count, store.version
            if not self._bm25:
                return []
            q = tokenize(query)
            if not q:
                return []
            scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._chunks[i] for i in ranked[:n] if scores[i] > 0]


_bm25 = _BM25Index()


def retrieve(store: KnowledgeStore, query: str, k: int, mode: str = "hybrid") -> list[Hit]:
    vector = store.embed_query(query)
    if mode == "dense":
        return [Hit(i + 1, c, s, dense_rank=i + 1) for i, (c, s) in enumerate(store.search_by_vector(vector, k))]

    pool = max(4 * k, 20)  # candidates taken from each ranker before fusion
    dense = store.search_by_vector(vector, min(pool, store.count()))
    keyword = _bm25.search(store, query, pool)

    fused: dict[str, dict[str, Any]] = {}
    for rank, (chunk, cos) in enumerate(dense, 1):
        fused[chunk.id] = {"chunk": chunk, "cos": cos, "dense_rank": rank, "bm25_rank": None,
                           "rrf": 1 / (RRF_K + rank)}
    for rank, chunk in enumerate(keyword, 1):
        entry = fused.setdefault(chunk.id, {"chunk": chunk, "cos": None, "dense_rank": None, "rrf": 0.0})
        entry["bm25_rank"] = rank
        entry["rrf"] += 1 / (RRF_K + rank)

    top = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:k]
    missing = [e["chunk"].id for e in top if e["cos"] is None]
    if missing:  # keyword-only hits outside the dense candidate pool
        cos = store.cosine_for(vector, missing)
        for e in top:
            if e["cos"] is None:
                e["cos"] = cos.get(e["chunk"].id, 0.0)
    return [
        Hit(i + 1, e["chunk"], e["cos"], dense_rank=e["dense_rank"], bm25_rank=e["bm25_rank"], rrf=e["rrf"])
        for i, e in enumerate(top)
    ]


def top_score(hits: list[Hit]) -> float:
    """Best cosine similarity among the hits (drives the refusal threshold)."""
    return max((h.score for h in hits), default=0.0)
