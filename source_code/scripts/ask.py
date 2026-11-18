"""Ask the handbook chatbot from the terminal (streams the answer, then lists the sources).

Usage (from source_code/), against a running stack:
    uv run python scripts/ask.py "What are the prerequisites for CS311?"
    uv run python scripts/ask.py "..." --model qwen2.5:0.5b        # compare models
    uv run python scripts/ask.py                                    # interactive chat with history
    uv run python scripts/ask.py --url http://localhost:8080 "..."  # dev-mode API instead of Docker

Uses the same POST /api/chat endpoint as the web UI. In interactive mode the last 3 turns are
sent as history, so follow-ups like "and the minor?" work. Type "exit" or Ctrl-D to quit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

BOLD, DIM, CYAN, YELLOW, RED, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"


def ask(client: httpx.Client, question: str, history: list[dict], model: str | None, show_sources: bool) -> str:
    body = {"question": question, "history": history, **({"model": model} if model else {})}
    answer, sources, name = "", [], None
    t0 = time.perf_counter()
    with client.stream("POST", "/api/chat", json=body, timeout=600) as r:
        if r.status_code != 200:
            r.read()
            print(f"{RED}HTTP {r.status_code}: {r.text}{RESET}")
            return ""
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line[6:].strip()
                continue
            if not (line.startswith("data:") and name):
                continue
            data = json.loads(line[5:].strip())
            if name == "sources":
                sources = data["sources"]
                print(f"{DIM}[model {data['model']} · {data['mode']} retrieval · "
                      f"relevance {data['top_score']:.3f} ≥ threshold {data['threshold']}]{RESET}")
                print(f"{BOLD}A:{RESET} ", end="", flush=True)
            elif name == "token":
                answer += data["text"]
                print(data["text"], end="", flush=True)
            elif name == "done":
                print(f"\n{DIM}({data['elapsed_ms'] / 1000:.1f}s){RESET}")
            elif name == "refusal":
                print(f"{YELLOW}[refused before the LLM: relevance {data['top_score']:.3f} < "
                      f"threshold {data['threshold']}]{RESET}\n{BOLD}A:{RESET} {data['message']}")
                answer = data["message"]
            elif name == "error":
                print(f"\n{RED}error: {data['message']}{RESET}")
            name = None
    if show_sources and sources:
        print(f"{BOLD}Sources{RESET}")
        for s in sources:
            print(f"  {CYAN}[{s['n']}]{RESET} {s['source']} p.{s['page']} — {s['heading'] or s['section']}"
                  f"  {DIM}(score {s['score']:.3f}){RESET}")
    print(f"{DIM}total {time.perf_counter() - t0:.1f}s{RESET}\n")
    return answer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*", help="omit for interactive mode")
    ap.add_argument("--url", default="http://localhost:3000", help="Docker UI (default) or dev API :8080")
    ap.add_argument("--model", help="e.g. llama3.1:8b or qwen2.5:0.5b (default: server default)")
    ap.add_argument("--no-sources", action="store_true", help="don't list the retrieved sources")
    args = ap.parse_args()
    show = not args.no_sources

    with httpx.Client(base_url=args.url, timeout=30) as client:
        try:
            client.get("/api/health").raise_for_status()
        except httpx.HTTPError as e:
            print(f"{RED}Cannot reach the API at {args.url}: {e}{RESET}", file=sys.stderr)
            return 2

        if args.question:
            ask(client, " ".join(args.question), [], args.model, show)
            return 0

        history: list[dict] = []
        print(f"{DIM}Handbook chatbot at {args.url}. Type 'exit' to quit.{RESET}")
        while True:
            try:
                q = input(f"{BOLD}Q:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if q.lower() in {"exit", "quit"}:
                return 0
            if not q:
                continue
            a = ask(client, q, history[-6:], args.model, show)
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


if __name__ == "__main__":
    sys.exit(main())
