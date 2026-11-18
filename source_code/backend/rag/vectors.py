"""Look inside the Chroma vector database: collection settings, stored embeddings, vector search.

Usage (Docker stack, from the repo root):
    docker compose -f docker/compose.yml exec api python -m rag.vectors
    docker compose -f docker/compose.yml exec api python -m rag.vectors --contains Capstone --limit 2
    docker compose -f docker/compose.yml exec api python -m rag.vectors --query "minimum GPA for the capstone"
Dev mode (from source_code/): uv run python -m rag.vectors ...

Unlike GET /api/chunks (text + metadata only), this prints the embedding vectors themselves and
runs a nearest-neighbour query directly against Chroma's HNSW index, with no BM25, gate, or LLM.
"""

from __future__ import annotations

import argparse
import math
import sys

from .config import get_settings
from .store import get_store

BOLD, DIM, CYAN, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[0m"


def fmt_vec(v, n: int) -> str:
    head = ", ".join(f"{float(x):+.4f}" for x in v[:n])
    return f"[{head}, … ]  ({len(v)} dims, L2 norm {math.sqrt(sum(float(x) ** 2 for x in v)):.3f})"


def one_line(text: str, width: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rag.vectors", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contains", help="only chunks whose text contains this (case-sensitive, Chroma filter)")
    ap.add_argument("--limit", type=int, default=2, help="stored chunks to print (default 2)")
    ap.add_argument("--dims", type=int, default=8, help="vector values to print per chunk (default 8)")
    ap.add_argument("--query", help="embed this question and run a nearest-neighbour search")
    ap.add_argument("--k", type=int, default=3, help="neighbours for --query (default 3)")
    args = ap.parse_args(argv)

    s = get_settings()
    store = get_store()
    col = store.collection

    # 1. The database itself
    print(f"{BOLD}Chroma vector database{RESET}  {s.chroma_host}:{s.chroma_port}")
    print(f"  collection : {col.name}  id={col.id}")
    print(f"  index      : HNSW, distance = {(col.metadata or {}).get('hnsw:space', 'l2')}")
    print(f"  vectors    : {col.count()} chunks, embedded with {s.embed_model} "
          f"(prefixes {'on' if s.embed_prefixes else 'off'})")

    # 2. Stored records: id + text + metadata + embedding
    where_doc = {"$contains": args.contains} if args.contains else None
    got = col.get(where_document=where_doc, limit=args.limit,
                  include=["embeddings", "documents", "metadatas"])
    label = f' containing "{args.contains}"' if args.contains else ""
    print(f"\n{BOLD}Stored records{RESET}{label} (showing {len(got['ids'])})")
    for cid, doc, meta, emb in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]):
        print(f"\n  {CYAN}id {cid[:16]}…  {meta.get('source')} p.{meta.get('page')}  "
              f"{meta.get('heading') or meta.get('section') or '(no heading)'}{RESET}")
        print(f"  text      : {one_line(doc)}")
        print(f"  embedding : {fmt_vec(emb, args.dims)}")

    # 3. Vector search straight against the index
    if args.query:
        qv = store.embed_query(args.query)
        print(f"\n{BOLD}Vector search{RESET}  \"{args.query}\"")
        print(f"  query embedding : {fmt_vec(qv, args.dims)}")
        res = col.query(query_embeddings=[qv], n_results=args.k,
                        include=["documents", "metadatas", "distances"])
        for rank, (doc, meta, dist) in enumerate(
                zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1):
            print(f"  {rank}. cosine distance {dist:.4f} → similarity {1 - dist:.4f}  "
                  f"{CYAN}p.{meta.get('page')} {meta.get('heading') or meta.get('section')}{RESET}")
            print(f"     {DIM}{one_line(doc, 100)}{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
