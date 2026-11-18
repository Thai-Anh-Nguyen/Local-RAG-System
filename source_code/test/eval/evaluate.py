"""Evaluation: baseline vs. improved pipeline on a fixed handbook question set.

    # needs an Ollama with nomic-embed-text, llama3.1:8b, qwen2.5:0.5b and a Chroma server
    uv run python test/eval/evaluate.py --ollama http://127.0.0.1:11435 --chroma-port 8000

Part 1, retrieval ablation (hit@5 / MRR, no LLM), each step adds one change:
  A  baseline     pypdf text, plain 1000/200 splitter, no nomic prefixes, L2 distance (as run_txt.py)
  B  +structure   PyMuPDF markdown, heading-aware chunks with breadcrumbs, cosine
  C  +prefixes    nomic search_document:/search_query: prefixes
  D  +hybrid      C + BM25 keyword search fused with RRF            (= the shipped system)

Part 2, answers on all questions from three generation setups:
  baseline-qwen0.5b   A retrieval, baseline gate/prompt (general-knowledge fallback), temp 0.7, qwen2.5:0.5b
  final-qwen0.5b      D retrieval, refusal threshold, grounded cited prompt, temp 0.1, qwen2.5:0.5b
  final-llama8b       same as above with llama3.1:8b (the default)

Writes report/eval/{retrieval.md, answers.md, summary.md, results.json}. The "auto" columns are
string-match heuristics; the grade columns in answers.md are for manual grading.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chunking import chunk_pages
from rag.config import Settings
from rag.embeddings import PrefixedEmbeddings
from rag.generation import NOT_FOUND, build_messages
from rag.parsing import parse_file
from rag.retrieval import Turn, retrieval_query, retrieve, top_score
from rag.store import KnowledgeStore

ROOT = Path(__file__).resolve().parents[2]  # source_code/
HANDBOOK = ROOT / "data/handbooks/CS_MajorHandbook.pdf"
QUESTIONS = Path(__file__).with_name("questions.json")
OUT = ROOT.parent / "report/eval"
K = 5

# Verbatim from final-project/sample_code/run_txt.py
BASELINE_PROMPT = """You are a helpful AI assistant that answers questions based on provided documents.

CONTEXT FROM DOCUMENTS:
{context}
---
QUESTION: {question}

INSTRUCTIONS:
1. If the context contains relevant information, answer based on that.
2. If the context is insufficient, use your general knowledge.
3. Be concise - 2-4 sentences unless more detail is needed.

ANSWER:"""
BASELINE_THRESHOLD = 0.3
BASELINE_FALLBACK = "No sufficiently relevant documents found. Use your general knowledge."


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def is_hit(text: str, evidence: str | None) -> bool:
    return bool(evidence) and squash(evidence) in squash(text)


# ---- index builders -------------------------------------------------------------------------
class BaselineIndex:
    """run_txt.py's pipeline, extended only as far as reading the PDF with pypdf."""

    def __init__(self, client, emb, name="eval_A_baseline"):
        try:
            client.delete_collection(name)
        except Exception:
            pass
        self.col = client.create_collection(name)  # Chroma default space: L2
        self.emb = emb
        pages = parse_file(HANDBOOK, "pypdf")
        docs = [Document(p.text, metadata={"source": HANDBOOK.name, "page": p.page}) for p in pages]
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        self.col.add(
            ids=[f"a{i}" for i in range(len(chunks))],
            documents=[c.page_content for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=emb.embed_documents([c.page_content for c in chunks]),
        )
        self.n = len(chunks)

    def search(self, query: str, k: int = K):
        r = self.col.query(query_embeddings=[self.emb.embed_query(query)], n_results=k,
                           include=["documents", "metadatas", "distances"])
        return [
            {"text": t, "page": m["page"], "distance": d, "score": 1 / (1 + d)}
            for t, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]


def structured_store(client, emb, name) -> KnowledgeStore:
    try:
        client.delete_collection(name)
    except Exception:
        pass
    store = KnowledgeStore(client, emb, name)
    pages = parse_file(HANDBOOK, "pymupdf")
    store.add_chunks(chunk_pages(pages, source=HANDBOOK.name, file_hash="eval", chunk_size=1000, chunk_overlap=200))
    return store


# ---- part 1: retrieval ----------------------------------------------------------------------
def rank_of(hits_text: list[str], evidence: str | None) -> int | None:
    return next((i + 1 for i, t in enumerate(hits_text) if is_hit(t, evidence)), None)


def retrieval_ablation(questions, baseline: BaselineIndex, store_b, store_c):
    configs = {
        "A baseline (pypdf, plain chunks, no prefix, L2)": lambda q: [h["text"] for h in baseline.search(q)],
        "B + PyMuPDF heading-aware chunks (dense)": lambda q: [h.chunk.text for h in retrieve(store_b, q, K, "dense")],
        "C + nomic task prefixes (dense)": lambda q: [h.chunk.text for h in retrieve(store_c, q, K, "dense")],
        "D + hybrid BM25/RRF (shipped)": lambda q: [h.chunk.text for h in retrieve(store_c, q, K, "hybrid")],
    }
    answerable = [q for q in questions if q.get("answerable", True)]
    rows, per_question = [], {}
    for name, fn in configs.items():
        ranks = []
        for q in answerable:
            query = retrieval_query(q["question"], [Turn(**t) for t in q.get("history", [])])
            r = rank_of(fn(query), q["evidence"])
            ranks.append(r)
            per_question.setdefault(q["id"], {})[name[0]] = r
        hits = sum(r is not None for r in ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        rows.append({"config": name, "hit@1": sum(r == 1 for r in ranks), "hit@5": hits,
                     "n": len(ranks), "mrr": round(mrr, 3)})
    return rows, per_question


def score_distribution(questions, baseline: BaselineIndex, store_c):
    """Top scores for answerable vs. unanswerable questions, for both gates."""
    out = {"baseline_avg_inv_l2": {"answerable": [], "unanswerable": []},
           "final_best_cosine": {"answerable": [], "unanswerable": []}}
    for q in questions:
        key = "answerable" if q.get("answerable", True) else "unanswerable"
        hits = baseline.search(q["question"])
        out["baseline_avg_inv_l2"][key].append(sum(h["score"] for h in hits) / len(hits))
        out["final_best_cosine"][key].append(top_score(retrieve(store_c, q["question"], K, "hybrid")))
    return out


# ---- part 2: generation ---------------------------------------------------------------------
def run_baseline(q, baseline: BaselineIndex, llm: OllamaLLM):
    t0 = time.perf_counter()
    hits = baseline.search(q["question"])
    avg = sum(h["score"] for h in hits) / len(hits)
    fallback = avg < BASELINE_THRESHOLD
    context = BASELINE_FALLBACK if fallback else "\n\n---\n\n".join(f"[{HANDBOOK.name}]\n{h['text']}" for h in hits)
    answer = llm.invoke(BASELINE_PROMPT.replace("{context}", context).replace("{question}", q["question"])).strip()
    return {"answer": answer, "refused": False, "gate_score": round(avg, 4), "fallback": fallback,
            "latency_s": round(time.perf_counter() - t0, 2), "pages": [h["page"] for h in hits]}


def run_final(q, store: KnowledgeStore, settings: Settings, model: str):
    t0 = time.perf_counter()
    history = [Turn(**t) for t in q.get("history", [])]
    hits = retrieve(store, retrieval_query(q["question"], history), settings.top_k, "hybrid")
    best = top_score(hits)
    if best < settings.relevance_threshold:
        return {"answer": "(refused before LLM)", "refused": True, "gate_score": round(best, 4),
                "latency_s": round(time.perf_counter() - t0, 2), "pages": [h.chunk.metadata["page"] for h in hits]}
    llm = ChatOllama(model=model, base_url=settings.ollama_url, temperature=settings.temperature,
                     num_ctx=settings.num_ctx)
    answer = llm.invoke(build_messages(q["question"], hits, history, settings.history_turns)).content.strip()
    cites = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return {"answer": answer, "refused": False, "gate_score": round(best, 4),
            "latency_s": round(time.perf_counter() - t0, 2), "pages": [h.chunk.metadata["page"] for h in hits],
            "citations": sorted(cites), "citations_valid": bool(cites) and cites <= {h.n for h in hits}}


def auto_check(q, r) -> str:
    """Heuristic only: answerable -> all key facts present; unanswerable -> declined."""
    if not q.get("answerable", True):
        declined = r["refused"] or NOT_FOUND.lower()[:20] in r["answer"].lower() or "not" in r["answer"].lower()[:80]
        return "declined" if declined else "ANSWERED"
    if r["refused"]:
        return "REFUSED"
    found = [f for f in q["facts"] if squash(f) in squash(r["answer"])]
    return "facts ok" if len(found) == len(q["facts"]) else f"facts {len(found)}/{len(q['facts'])}"


# ---- reporting ------------------------------------------------------------------------------
def md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", "<br>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default="http://127.0.0.1:11435")
    ap.add_argument("--chroma-host", default="127.0.0.1")
    ap.add_argument("--chroma-port", type=int, default=8000)
    ap.add_argument("--only", nargs="*", help="question ids to run (default: all)")
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text())
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
    settings = Settings(_env_file=None, ollama_url=args.ollama)
    client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
    raw = OllamaEmbeddings(model="nomic-embed-text", base_url=args.ollama)

    print("Indexing A (baseline)…")
    baseline = BaselineIndex(client, PrefixedEmbeddings(raw, use_prefixes=False))
    print("Indexing B (structure, no prefixes)…")
    store_b = structured_store(client, PrefixedEmbeddings(raw, use_prefixes=False), "eval_B_structure")
    print("Indexing C/D (structure + prefixes)…")
    store_c = structured_store(client, PrefixedEmbeddings(raw, use_prefixes=True), "eval_C_prefixes")
    print(f"  chunks: A={baseline.n}  B={store_b.count()}  C={store_c.count()}")

    retrieval_rows, per_q = retrieval_ablation(questions, baseline, store_b, store_c)
    dist = score_distribution(questions, baseline, store_c)

    results = []
    if not args.skip_generation:
        base_llm = OllamaLLM(model="qwen2.5:0.5b", base_url=args.ollama, temperature=0.7)
        for q in questions:
            print(f"  {q['id']}: {q['question'][:60]}")
            row = {"id": q["id"], "question": q["question"], "category": q["category"], "expected": q["expected"]}
            row["baseline-qwen0.5b"] = run_baseline(q, baseline, base_llm)
            row["final-qwen0.5b"] = run_final(q, store_c, settings, "qwen2.5:0.5b")
            row["final-llama8b"] = run_final(q, store_c, settings, "llama3.1:8b")
            for setup in ("baseline-qwen0.5b", "final-qwen0.5b", "final-llama8b"):
                row[setup]["auto"] = auto_check(q, row[setup])
            results.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"retrieval": retrieval_rows, "retrieval_per_question": per_q, "scores": dist, "answers": results},
        indent=2, ensure_ascii=False))

    # retrieval.md
    lines = ["# Retrieval ablation", "",
             f"{sum(q.get('answerable', True) for q in questions)} answerable questions; a hit = a retrieved chunk "
             "contains the evidence string (whitespace-insensitive). k = 5.", "",
             "| Config | hit@1 | hit@5 | MRR |", "|---|---|---|---|"]
    lines += [f"| {r['config']} | {r['hit@1']}/{r['n']} | {r['hit@5']}/{r['n']} | {r['mrr']} |" for r in retrieval_rows]
    lines += ["", "## Rank of first evidence chunk per question (– = miss)", "",
              "| Q | A | B | C | D |", "|---|---|---|---|---|"]
    for qid, r in per_q.items():
        lines.append(f"| {qid} | " + " | ".join(str(r.get(c) or "–") for c in "ABCD") + " |")

    def stats(xs):
        return f"min {min(xs):.3f} · median {statistics.median(xs):.3f} · max {max(xs):.3f}" if xs else "–"

    lines += ["", "## Relevance-gate scores", "",
              "| Gate | answerable questions | unanswerable / off-topic |", "|---|---|---|",
              f"| Baseline: mean 1/(1+L2) over top-5 (threshold {BASELINE_THRESHOLD}) | "
              f"{stats(dist['baseline_avg_inv_l2']['answerable'])} | {stats(dist['baseline_avg_inv_l2']['unanswerable'])} |",
              f"| Final: best cosine among hybrid top-5 (threshold {settings.relevance_threshold}) | "
              f"{stats(dist['final_best_cosine']['answerable'])} | {stats(dist['final_best_cosine']['unanswerable'])} |"]
    (OUT / "retrieval.md").write_text("\n".join(lines) + "\n")

    if results:
        setups = ("baseline-qwen0.5b", "final-qwen0.5b", "final-llama8b")
        a = ["# Answers (for manual grading)", "",
             "Grade each answer: **correct / incomplete / incorrect / ambiguous** (unanswerable questions: "
             "correct = declined). `auto` is a string-match heuristic, not a grade.", ""]
        for row in results:
            a += [f"## {row['id']} ({row['category']}): {row['question']}", "", f"**Expected:** {row['expected']}", "",
                  "| Setup | Answer | auto | gate score | latency | grade |", "|---|---|---|---|---|---|"]
            for s in setups:
                r = row[s]
                extra = " (fallback: general knowledge)" if r.get("fallback") else ""
                a.append(f"| {s} | {md_escape(r['answer'])}{extra} | {r['auto']} | {r['gate_score']} | "
                         f"{r['latency_s']}s |  |")
            a.append("")
        (OUT / "answers.md").write_text("\n".join(a))

        s = ["# Evaluation summary (automatic heuristics)", "",
             "| Setup | facts ok (answerable) | refused/declined (unanswerable) | wrongly refused (answerable) | "
             "citations valid | median latency |", "|---|---|---|---|---|---|"]
        answerable_ids = {q["id"] for q in questions if q.get("answerable", True)}
        ans = [r for r in results if r["id"] in answerable_ids]
        una = [r for r in results if r["id"] not in answerable_ids]
        for st in setups:
            ok = sum(r[st]["auto"] == "facts ok" for r in ans)
            dec = sum(r[st]["auto"] == "declined" for r in una)
            wrong_ref = sum(r[st]["refused"] for r in ans)
            cit = [r[st].get("citations_valid") for r in ans if not r[st]["refused"] and "citations_valid" in r[st]]
            cit_s = f"{sum(bool(c) for c in cit)}/{len(cit)}" if cit else "n/a (baseline has none)"
            lat = statistics.median(r[st]["latency_s"] for r in results)
            s.append(f"| {st} | {ok}/{len(ans)} | {dec}/{len(una)} | {wrong_ref}/{len(ans)} | {cit_s} | {lat:.1f}s |")
        fb = sum(r["baseline-qwen0.5b"]["fallback"] for r in results)
        s += ["", f"Baseline relevance gate fired (fallback to general knowledge) on {fb}/{len(results)} questions."]
        (OUT / "summary.md").write_text("\n".join(s) + "\n")

    print(f"\nWrote {OUT}/retrieval.md" + (", answers.md, summary.md" if results else "") + ", results.json")
    print((OUT / "retrieval.md").read_text())
    if results:
        print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
