"""Ingest files into the knowledge base.

Usage (from source_code/):
    uv run python -m rag.ingest data/handbooks                 # a folder (recursive)
    uv run python -m rag.ingest path/to/file.pdf notes.md      # specific files
    uv run python -m rag.ingest data/handbooks --reset         # wipe the collection first
    uv run python -m rag.ingest file.pdf --parser pypdf        # override RAG_PDF_PARSER

Re-ingesting an unchanged file is a no-op; a changed file replaces its old chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

from .chunking import CHUNKER_VERSION, chunk_pages
from .config import Settings, get_settings
from .parsing import SUPPORTED_SUFFIXES, parse_file
from .store import KnowledgeStore, get_store


@dataclass
class IngestResult:
    source: str
    status: str  # "indexed" | "replaced" | "unchanged"
    chunks: int
    pages: int


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES)
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Unsupported file type: {p} (supported: {sorted(SUPPORTED_SUFFIXES)})")
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


def ingest_file(
    path: Path,
    store: KnowledgeStore,
    settings: Settings,
    *,
    source: str | None = None,
    parser: str | None = None,
) -> IngestResult:
    source = source or path.name
    parser = parser or settings.pdf_parser
    # Fingerprint = file content + everything that shapes the chunks, so changing the parser,
    # chunk size/overlap or the chunker itself re-indexes the file instead of skipping it.
    recipe = f"{parser}:{settings.chunk_size}:{settings.chunk_overlap}:{CHUNKER_VERSION}"
    digest = file_hash(path.read_bytes() + recipe.encode())
    existing = store.source_hash(source)
    if existing == digest:
        return IngestResult(source, "unchanged", 0, 0)

    pages = parse_file(path, parser)
    docs = chunk_pages(
        pages,
        source=source,
        file_hash=digest,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    store.replace_source(source, docs)
    return IngestResult(source, "replaced" if existing else "indexed", len(docs), len(pages))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rag.ingest", description="Index .pdf/.md/.txt files")
    ap.add_argument("paths", nargs="+", type=Path, help="files and/or folders")
    ap.add_argument("--reset", action="store_true", help="delete the whole collection first")
    ap.add_argument("--parser", choices=["pymupdf", "pypdf"], help="PDF parser (default: RAG_PDF_PARSER)")
    args = ap.parse_args(argv)

    settings = get_settings()
    try:
        files = collect_files(args.paths)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not files:
        print("No supported files found.", file=sys.stderr)
        return 1

    store = get_store()
    if args.reset:
        store.reset()
        print(f"Collection '{settings.collection}' reset.")
    for f in files:
        r = ingest_file(f, store, settings, parser=args.parser)
        print(f"  {r.status:<9} {r.source}  ({r.pages} pages, {r.chunks} chunks)")
    print(f"Collection '{settings.collection}' now holds {store.count()} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
