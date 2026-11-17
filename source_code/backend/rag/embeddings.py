"""Ollama embeddings with nomic-embed-text's task prefixes.

nomic-embed-text was trained with "search_document: " on indexed text and "search_query: " on
queries. LangChain's OllamaEmbeddings sends raw text, which is what the baseline does.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class PrefixedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, use_prefixes: bool = True):
        self.inner = inner
        self.use_prefixes = use_prefixes

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.use_prefixes:
            texts = [DOC_PREFIX + t for t in texts]
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(QUERY_PREFIX + text if self.use_prefixes else text)


def make_embeddings(model: str, base_url: str, use_prefixes: bool) -> PrefixedEmbeddings:
    return PrefixedEmbeddings(OllamaEmbeddings(model=model, base_url=base_url), use_prefixes)
