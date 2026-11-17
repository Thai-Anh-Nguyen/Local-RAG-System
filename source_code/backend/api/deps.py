"""FastAPI dependencies; tests override these with fakes."""

from rag.config import Settings, get_settings
from rag.store import KnowledgeStore, get_store


def settings_dep() -> Settings:
    return get_settings()


def store_dep() -> KnowledgeStore:
    return get_store()
