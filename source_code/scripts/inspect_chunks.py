"""Show how a file is parsed and chunked, offline (no Ollama or Chroma needed).

Usage (from source_code/):
    uv run python scripts/inspect_chunks.py data/handbooks/CS_MajorHandbook.pdf
    uv run python scripts/inspect_chunks.py FILE --raw-page 7          # also print page 7 as parsed
    uv run python scripts/inspect_chunks.py FILE --grep CS311 --show 3  # chunks containing "CS311"
    uv run python scripts/inspect_chunks.py FILE --parser pypdf        # compare with the baseline parser

This runs exactly the parse + chunk steps that `python -m rag.ingest` uses before embedding.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from rag.chunking import chunk_pages
from rag.config import get_settings
from rag.ingest import file_hash
from rag.parsing import parse_file

BOLD, DIM, CYAN, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[0m"


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--parser", choices=["pymupdf", "pypdf"], default=s.pdf_parser)
    ap.add_argument("--chunk-size", type=int, default=s.chunk_size)
    ap.add_argument("--chunk-overlap", type=int, default=s.chunk_overlap)
    ap.add_argument("--show", type=int, default=5, help="how many chunks to print (default 5)")
    ap.add_argument("--grep", help="only print chunks containing this text (case-insensitive)")
    ap.add_argument("--page", type=int, help="only print chunks from this page")
    ap.add_argument("--raw-page", type=int, help="print this page's parsed text before chunking")
    args = ap.parse_args()

    if not args.file.is_file():
        print(f"error: {args.file} not found", file=sys.stderr)
        return 2

    # 1. Parse
    t0 = time.perf_counter()
    pages = parse_file(args.file, args.parser)
    t_parse = time.perf_counter() - t0
    chars = sum(len(p.text) for p in pages)
    print(f"{BOLD}1. Parse{RESET}  {args.file.name} with {args.parser}: "
          f"{len(pages)} non-empty pages, {chars:,} characters ({t_parse:.1f}s)")
    if args.raw_page:
        page = next((p for p in pages if p.page == args.raw_page), None)
        print(f"{DIM}--- page {args.raw_page} as parsed ---{RESET}")
        print(page.text if page else "(no such page)")
        print(f"{DIM}--- end of page {args.raw_page} ---{RESET}")

    # 2. Chunk
    docs = chunk_pages(pages, source=args.file.name, file_hash=file_hash(args.file.read_bytes()),
                       chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    lens = [len(d.page_content) for d in docs]
    sections = {d.metadata["section"] for d in docs}
    print(f"{BOLD}2. Chunk{RESET}  size={args.chunk_size}, overlap={args.chunk_overlap}, heading-aware: "
          f"{len(docs)} chunks, {len(sections)} sections, "
          f"length min/median/max = {min(lens)}/{int(statistics.median(lens))}/{max(lens)} chars")

    # 3. Show
    picked = [d for d in docs
              if (args.page is None or d.metadata["page"] == args.page)
              and (args.grep is None or args.grep.lower() in d.page_content.lower())]
    what = " matching" if (args.grep or args.page) else ""
    print(f"{BOLD}3. Chunks{RESET}  showing {min(args.show, len(picked))} of {len(picked)}{what}\n")
    for d in picked[: args.show]:
        m = d.metadata
        print(f"{CYAN}#{m['chunk_index']}  p.{m['page']}  {m['char_len']} chars  id={m['id'][:12]}{RESET}")
        print(f"{DIM}section: {m['section']}{RESET}")
        print(d.page_content.strip())
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
