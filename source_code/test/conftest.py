"""Shared fixtures: an in-memory Chroma store with deterministic fake embeddings, a fake LLM,
and a FastAPI TestClient wired to them. No Ollama or Chroma server needed."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid

import chromadb
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessageChunk

from api.deps import settings_dep, store_dep
from api.main import create_app
from api.routes_chat import llm_factory_dep
from rag.config import Settings
from rag.store import KnowledgeStore

DIM = 256


class HashEmbeddings(Embeddings):
    """Bag-of-words hashed into a fixed vector: shared words -> high cosine, unrelated -> ~0."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def __init__(self, reply="The major requires 56 credits [1].", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    async def astream(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise ConnectionError("Ollama is down")
        for word in self.reply.split(" "):
            yield AIMessageChunk(content=word + " ")


CHUNKS = [
    ("Major Requirements (56 credits)", 7, "Computer Science major requirements total 56 credits of courses."),
    ("CS311: Advanced Deep Learning", 18, "Prerequisites: CS313. Advanced deep learning architectures."),
    ("CS211: Operating Systems", 16, "Operating systems course topics: processes, threads, scheduling, memory, file systems."),
    ("Capstone", 9, "Capstone enrollment needs a minimum major GPA of 3.5 and an approved application."),
]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        ollama_url="http://127.0.0.1:9",  # nothing listens here
        data_dir=tmp_path / "data",
        relevance_threshold=0.3,
        top_k=3,
    )


@pytest.fixture
def store() -> KnowledgeStore:
    client = chromadb.EphemeralClient()
    s = KnowledgeStore(client, HashEmbeddings(), f"test_{uuid.uuid4().hex[:8]}")
    docs = [
        Document(
            page_content=f"[{heading}]\n{text}",
            metadata={"id": f"c{i}", "source": "hb.pdf", "file_hash": "h", "page": page, "chunk_index": 0,
                      "section": "", "heading": heading, "char_len": len(text), "edited": False},
        )
        for i, (heading, page, text) in enumerate(CHUNKS)
    ]
    s.add_chunks(docs)
    return s


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(settings, store, llm):
    app = create_app(run_lifespan=False)
    app.dependency_overrides[settings_dep] = lambda: settings
    app.dependency_overrides[store_dep] = lambda: store
    app.dependency_overrides[llm_factory_dep] = lambda: (lambda _settings, _model: llm)
    with TestClient(app) as c:
        yield c


def read_sse(response) -> list[tuple[str, dict]]:
    """Parse an SSE response body into (event, data) pairs (ping comments ignored)."""
    events, name, data = [], None, []
    for line in response.iter_lines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif line == "" and name:
            events.append((name, json.loads("\n".join(data))))
            name, data = None, []
    if name:
        events.append((name, json.loads("\n".join(data))))
    return events
