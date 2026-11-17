"""Retrieval: build the search query from the conversation and fetch scored chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .store import Chunk, KnowledgeStore


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class Hit:
    n: int  # citation number, 1-based
    chunk: Chunk
    score: float  # cosine similarity

    def as_dict(self, snippet_chars: int | None = None) -> dict[str, Any]:
        d = self.chunk.as_dict()
        if snippet_chars is not None:
            d["text"] = d["text"][:snippet_chars]
        return {"n": self.n, "score": round(self.score, 4), **d}


def retrieval_query(question: str, history: list[Turn]) -> str:
    """Current question plus the previous user question, so follow-ups like
    "what about its prerequisites?" still find the right section."""
    previous = next((t.content for t in reversed(history) if t.role == "user"), None)
    return f"{previous}\n{question}" if previous else question


def retrieve(store: KnowledgeStore, query: str, k: int) -> list[Hit]:
    return [Hit(i + 1, c, s) for i, (c, s) in enumerate(store.search(query, k))]
