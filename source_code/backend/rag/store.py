"""Chroma vector store access: the collection uses cosine distance, and all reads/writes go through here."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings
from .embeddings import make_embeddings


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        m = self.metadata
        return {
            "id": self.id,
            "source": m.get("source", ""),
            "page": m.get("page"),
            "chunk_index": m.get("chunk_index"),
            "section": m.get("section", ""),
            "heading": m.get("heading", ""),
            "char_len": m.get("char_len", len(self.text)),
            "edited": bool(m.get("edited", False)),
            "text": self.text,
        }


class KnowledgeStore:
    def __init__(self, client: chromadb.ClientAPI, embeddings: Embeddings, collection: str):
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection
        self.vs = Chroma(
            client=client,
            collection_name=collection,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self.vs._collection

    # ---- writes ----------------------------------------------------------------
    def add_chunks(self, docs: list[Document]) -> None:
        if not docs:
            return
        now = time.time()
        for d in docs:
            d.metadata["ingested_at"] = now
        self.vs.add_documents(docs, ids=[d.metadata["id"] for d in docs])

    def delete_source(self, source: str) -> int:
        ids = self.collection.get(where={"source": source}, include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.__init__(self.client, self.embeddings, self.collection_name)

    # ---- reads -----------------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def source_hash(self, source: str) -> str | None:
        got = self.collection.get(where={"source": source}, limit=1, include=["metadatas"])
        return got["metadatas"][0].get("file_hash") if got["ids"] else None

    def list_documents(self) -> list[dict[str, Any]]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        docs: dict[str, dict[str, Any]] = {}
        for m in metas:
            d = docs.setdefault(
                m["source"],
                {"source": m["source"], "chunks": 0, "pages": set(), "ingested_at": m.get("ingested_at")},
            )
            d["chunks"] += 1
            d["pages"].add(m.get("page"))
        return [
            {**d, "pages": len(d["pages"])} for d in sorted(docs.values(), key=lambda d: d["source"])
        ]

    def get_chunks(
        self,
        source: str | None = None,
        page: int | None = None,
        contains: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[Chunk]]:
        filters = [{"source": source}] if source else []
        if page is not None:
            filters.append({"page": page})
        where = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)
        got = self.collection.get(
            where=where,
            where_document={"$contains": contains} if contains else None,
            include=["documents", "metadatas"],
        )
        chunks = [Chunk(i, t, m) for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])]
        # Chroma does not guarantee order; the knowledge base is small, so sort in memory.
        chunks.sort(key=lambda c: (c.metadata.get("source", ""), c.metadata.get("page") or 0,
                                   c.metadata.get("chunk_index") or 0))
        return len(chunks), chunks[offset : offset + limit]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        got = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not got["ids"]:
            return None
        return Chunk(got["ids"][0], got["documents"][0], got["metadatas"][0])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Top-k chunks with cosine similarity (1 - cosine distance)."""
        results = self.vs.similarity_search_with_score(query, k=k)
        return [
            (Chunk(doc.id or doc.metadata.get("id", ""), doc.page_content, doc.metadata), 1.0 - dist)
            for doc, dist in results
        ]


def make_store(settings: Settings) -> KnowledgeStore:
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    embeddings = make_embeddings(settings.embed_model, settings.ollama_url, settings.embed_prefixes)
    return KnowledgeStore(client, embeddings, settings.collection)


@lru_cache
def get_store() -> KnowledgeStore:
    return make_store(get_settings())
