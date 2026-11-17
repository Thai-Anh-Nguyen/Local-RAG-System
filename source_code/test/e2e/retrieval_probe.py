"""Retrieval probe against a running API: shows the top-k chunks and scores per question.

    uv run python test/e2e/retrieval_probe.py [--url http://localhost:8080] [--k 5]

`expect` is a substring that should appear in a relevant chunk (text or heading); the probe
reports the rank of the first chunk containing it, or "miss". Off-topic questions have expect=None
and are used to see how low scores go when nothing is relevant (refusal-threshold calibration).
"""

from __future__ import annotations

import argparse

import httpx

PROBES: list[tuple[str, str | None]] = [
    ("How many credits are required for the Computer Science major?", "56 credits"),
    ("What are the prerequisites for CS311?", "CS311"),
    ("What is the minimum major GPA to enroll in the Capstone?", "3.5"),
    ("What courses are required for the CS minor?", "Minor Requirements"),
    ("What are the Program Educational Objectives?", "Program Educational Objectives"),
    ("What does the Operating Systems course cover?", "CS211"),
    ("Which courses are offered once a year?", "Once a"),
    ("What careers can CS graduates pursue?", "Career"),
    ("How are students assessed in the program?", "Assessment"),
    ("What is the major code of the B.S. Computer Science?", "7480101"),
    # off-topic
    ("What is the capital of France?", None),
    ("How do I bake sourdough bread?", None),
    ("Who won the 2022 FIFA World Cup?", None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    with httpx.Client(base_url=args.url, timeout=60) as client:
        for question, expect in PROBES:
            r = client.post("/api/retrieve", json={"question": question, "k": args.k}).json()
            rank = None
            if expect:
                for h in r["hits"]:
                    if expect.lower() in (h["text"] + h["heading"]).lower():
                        rank = h["n"]
                        break
            verdict = "off-topic" if expect is None else (f"hit@{rank}" if rank else "MISS")
            print(f"\n[{verdict}] top={r['top_score']:.3f}  {question}")
            for h in r["hits"]:
                print(f"   {h['n']}. {h['score']:.3f}  p{h['page']:<2} {h['heading'][:70]}")


if __name__ == "__main__":
    main()
