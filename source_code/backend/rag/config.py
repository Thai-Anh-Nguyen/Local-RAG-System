"""Runtime settings, read from environment variables (prefix RAG_) and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# source_code/ — so relative paths in .env resolve the same from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    # Ollama
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "llama3.1:8b"
    llm_allowlist: str = "llama3.1:8b,qwen2.5:0.5b"
    embed_model: str = "nomic-embed-text"
    embed_prefixes: bool = True
    temperature: float = 0.1
    num_ctx: int = 8192

    # Chroma
    chroma_host: str = "127.0.0.1"
    chroma_port: int = 8000
    collection: str = "cs_handbook"

    # Ingestion
    pdf_parser: Literal["pymupdf", "pypdf"] = "pymupdf"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    data_dir: Path = Path("data/handbooks")

    # Retrieval / answering
    top_k: int = 5
    relevance_threshold: float = 0.5
    history_turns: int = 3

    @property
    def allowed_models(self) -> list[str]:
        return [m.strip() for m in self.llm_allowlist.split(",") if m.strip()]

    @property
    def data_path(self) -> Path:
        return self.data_dir if self.data_dir.is_absolute() else PROJECT_ROOT / self.data_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
