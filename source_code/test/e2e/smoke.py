"""End-to-end smoke test against a running stack (Docker or dev mode).

    uv run python test/e2e/smoke.py                          # Docker stack via nginx (http://localhost:3000)
    uv run python test/e2e/smoke.py --url http://localhost:8080   # dev-mode API directly

Checks: health is ok, handbook questions get streamed answers with valid [n] citations,
an off-topic question is refused without generation. Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

import httpx

ANSWERABLE = [
    "How many credits are required for the Computer Science major?",
    "What are the prerequisites for CS311?",
    "What is the minimum major GPA to enroll in the Capstone?",
    "What courses are required for the CS minor?",
    "What does the Operating Systems course cover?",
]
OFF_TOPIC = "Who won the 2022 FIFA World Cup?"


def chat(client: httpx.Client, question: str, model: str | None) -> list[tuple[str, dict]]:
    events, name = [], None
    body = {"question": question, "history": [], **({"model": model} if model else {})}
    with client.stream("POST", "/api/chat", json=body, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:") and name:
                events.append((name, json.loads(line[5:].strip())))
                name = None
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3000")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    failures = 0

    def check(ok: bool, label: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    with httpx.Client(base_url=args.url, timeout=30) as client:
        health = client.get("/api/health").json()
        check(health["status"] == "ok", f"health ok ({json.dumps(health['models'])}, chunks={health['chroma']['chunks']})")

        for q in ANSWERABLE:
            t0 = time.perf_counter()
            events = chat(client, q, args.model)
            names = [e for e, _ in events]
            answer = "".join(d["text"] for e, d in events if e == "token")
            sources = events[0][1].get("sources", []) if events else []
            cites = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
            valid = cites and cites <= {s["n"] for s in sources}
            print(f"\nQ: {q}  ({time.perf_counter() - t0:.1f}s)\nA: {answer.strip()[:400]}")
            check(names[:1] == ["sources"] and names[-1:] == ["done"], "sources -> tokens -> done")
            check(bool(valid), f"answer cites existing sources {sorted(cites)}")

        events = chat(client, OFF_TOPIC, args.model)
        print(f"\nQ: {OFF_TOPIC}")
        check([e for e, _ in events] == ["refusal"],
              f"off-topic refused (top_score={events[0][1].get('top_score')})")

    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
