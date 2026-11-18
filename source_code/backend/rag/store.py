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
        self.version = getattr(self, "version", 0) + 1  # bumped on every write (cache invalidation)
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
    def add_chunks(self, docs: list[Document], embeddings: list[list[float]] | None = None) -> None:
        if not docs:
            return
        now = time.time()
        for d in docs:
            d.metadata["ingested_at"] = now
        if embeddings is None:
            embeddings = self.embeddings.embed_documents([d.page_content for d in docs])
        self.collection.upsert(
            ids=[d.metadata["id"] for d in docs],
            documents=[d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
            embeddings=embeddings,
        )
        self.version += 1

    def replace_source(self, source: str, docs: list[Document]) -> None:
        """Embed first, then swap: if embedding fails (e.g. Ollama down) the old chunks stay."""
        embeddings = self.embeddings.embed_documents([d.page_content for d in docs]) if docs else []
        self.delete_source(source)
        self.add_chunks(docs, embeddings)

    def delete_source(self, source: str) -> int:
        ids = self.collection.get(where={"source": source}, include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
            self.version += 1
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

    def all_chunks(self) -> list[Chunk]:
        got = self.collection.get(include=["documents", "metadatas"])
        return [Chunk(i, t, m) for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])]

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)

    def search_by_vector(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        """Top-k chunks with cosine similarity (1 - cosine distance)."""
        if k <= 0:
            return []
        got = self.collection.query(
            query_embeddings=[vector], n_results=k, include=["documents", "metadatas", "distances"]
        )
        return [
            (Chunk(i, t, m), 1.0 - d)
            for i, t, m, d in zip(got["ids"][0], got["documents"][0], got["metadatas"][0], got["distances"][0])
        ]

    def cosine_for(self, vector: list[float], ids: list[str]) -> dict[str, float]:
        """Cosine similarity between `vector` and the stored embeddings of `ids`."""
        if not ids:
            return {}
        got = self.collection.get(ids=ids, include=["embeddings"])
        qn = sum(x * x for x in vector) ** 0.5 or 1.0
        out = {}
        for cid, emb in zip(got["ids"], got["embeddings"]):
            en = sum(float(x) * float(x) for x in emb) ** 0.5 or 1.0
            out[cid] = sum(float(a) * float(b) for a, b in zip(vector, emb)) / (qn * en)
        return out

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Dense-only top-k (cosine similarity)."""
        return self.search_by_vector(self.embed_query(query), k)


def make_store(settings: Settings) -> KnowledgeStore:
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    embeddings = make_embeddings(settings.embed_model, settings.ollama_url, settings.embed_prefixes)
    return KnowledgeStore(client, embeddings, settings.collection)


@lru_cache
def get_store() -> KnowledgeStore:
    return make_store(get_settings())
