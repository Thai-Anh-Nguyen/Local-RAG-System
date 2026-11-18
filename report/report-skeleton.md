# Building and Improving a Local RAG System: CS Major Handbook Assistant

*CS311 Advanced Deep Learning, Final Project · Student ID 240039*

> **Skeleton.** Every section lists what to cover, the facts and numbers already measured (from `report/eval/`, `report/design-decisions.md` and the code), and figures or screenshots to add. Write the prose in your own words. The target is at least 1,500 words; suggested lengths are given per section.

---

## 1. Introduction (~150 words)
- Problem: LLMs hallucinate and don't know institution-specific documents. RAG grounds answers in retrieved text, and running it locally keeps data private, costs nothing per query and works offline.
- Goal: extend the course baseline (`run_txt.py`, a .txt-only CLI) into a deployable chatbot for the **B.S. Computer Science major handbook** (19 pages, about 5.5k words).
- Contributions, one line each: PDF ingestion (Task 1), handbook chatbot with evaluation (Task 2), Docker deployment with GPU (Task 3), web UI (Task 4). Improvements beyond the baseline: heading-aware chunking, nomic prefixes, hybrid retrieval, grounded and cited prompting, a refusal gate.

## 2. System architecture (~250 words)
- Figure: the mermaid diagram from the README (export it as an image).
- Components and choices, with one reason each:
  - **Parsing:** PyMuPDF via `pymupdf4llm` (markdown with headings and tables) as the default; pypdf can be switched on.
  - **Chunking:** split at headings, then `RecursiveCharacterTextSplitter` 1000/200. Each chunk carries a `[Section > Heading]` breadcrumb plus page and heading metadata. Chunk IDs are stable SHA-1 hashes, so re-ingesting is idempotent.
  - **Embeddings:** `nomic-embed-text` (137M parameters, 768 dimensions) through Ollama, with `search_document:` / `search_query:` prefixes.
  - **Vector store:** ChromaDB server, cosine space.
  - **Retrieval:** hybrid. Dense top-20 and BM25 top-20 are fused with Reciprocal Rank Fusion (k = 60), and the top 5 go to the LLM.
  - **Gate:** if the best cosine is below 0.62, refuse without calling the LLM.
  - **Generation:** `llama3.1:8b` (or `qwen2.5:0.5b`), temperature 0.1, `num_ctx` 8192, streamed over SSE.
  - **API:** FastAPI, stateless; history comes from the client.
  - **UI:** React + Vite + Tailwind, served by nginx.
- Monorepo layout: `source_code/backend/rag` has no FastAPI imports, so the CLI, API and evaluation share one core.

## 3. Task 1: PDF support (~150 words)
- What changed: the loader dispatches by suffix (`.pdf`, `.md`, `.txt`), with a CLI (`python -m rag.ingest <files|dirs> [--reset] [--parser]`) and UI upload.
- **Observation (PDF quality):** the handbook PDF (exported from macOS Quartz) has broken ligature glyphs. "tt" and "ti" come out as `0`, `2`, `8`, `;` or `'` ("A0ributes", "Descrip8on", "Educa;onal", "Founda'onal"). These were fixed with targeted regexes that leave course codes alone (`CS2xx`, `MATH103`) and are covered by unit tests. Spaces are also lost inside table cells ("Once ayear", "DeepLearning"); that is not fixed and is listed as a limitation.
- Page 10 is an image (the curriculum diagram), so it yields no text. There's no OCR (FAQ 5).
- PyMuPDF vs pypdf: PyMuPDF keeps `#` headings and markdown tables; pypdf gives flat text. Chunk counts: <!-- fill from results.json: A vs C -->.
- Screenshot: the Knowledge base page showing the chunks of a page.

## 4. Task 2: Handbook chatbot and evaluation (~400 words, the core of the report)
### 4.1 Test set
- 30 questions (`source_code/test/eval/questions.json`):
  - 10 on requirements and policies
  - 8 on courses
  - 6 on outcomes and other topics
  - 1 tricky question (the handbook's own "Applied Mathematics" typo)
  - 2 in-domain questions the handbook can't answer, and 1 off-topic question
  - 1 in Vietnamese and 1 follow-up
- Each has an expected answer, key facts and an evidence string used for automatic hit@k.

### 4.2 Retrieval ablation (from `report/eval/retrieval.md`)
- Table: A baseline → B structure → C prefixes → D hybrid, with hit@1, hit@5 and MRR. <!-- paste -->
- Discuss which step fixed which questions, using the per-question rank table. Expected story, to be checked against the numbers:
  - Structure fixed course-description attribution: the bold "CS307" heading bug was found with the chunk inspector.
  - Prefixes improved the ranking of natural-language questions.
  - BM25 rescued exact codes and names ("CS311", "Operating Systems") that the dense model blurred. The retrieval check before the fix: "prerequisites for CS311" missed CS311 entirely, and "what does Operating Systems cover" ranked CS210 Networks first.

### 4.3 Relevance gate
- Baseline gate: mean of `1/(1+L2)` over the top 5 vs 0.3. Report the measured range <!-- from retrieval.md -->. **Claim to verify with the numbers:** because nomic vectors are about unit length (L2 ≤ 2, so the score is ≥ 0.33), the gate can never fire, and the baseline answers everything, falling back to general knowledge only in theory.
- Final gate: the best cosine among the hybrid top 5. Calibrated on the probe: off-topic questions peaked at about 0.59 and the weakest on-topic question scored about 0.66, giving a threshold of 0.62. Report the measured distributions and any wrongly refused questions.

### 4.4 Answer quality: baseline vs final, small vs large model
- Table from `report/eval/summary.md`. Then add your **manual grades** from `report/eval/answers.md` (correct / incomplete / incorrect / ambiguous per setup).
- Required by the brief: examples of **correct, incorrect, incomplete and ambiguous** answers. Pick 1–2 each from `answers.md` and explain *why* (retrieval miss, table flattening, small model ignoring instructions, source ambiguity).
- The T1 "Applied Mathematics" typo is a natural ambiguity case: the source document itself is wrong.
- The V1 Vietnamese question: was the answer language right, and did retrieval still work?
- The F1 follow-up: did history-aware retrieval find CS313?
- Latency: median per setup, on the GPU.

## 5. Task 3: Docker deployment (~200 words)
- Services: `frontend` (nginx), `api`, `chroma`, `ollama` and the one-shot `ollama-init`. Volumes for models and vectors; a bind mount for documents and uploads.
- GPU: NVIDIA Container Toolkit with **CDI** (`devices: [nvidia.com/gpu=all]`) on Fedora 44 with SELinux enforcing. `--gpus all` and `driver: nvidia` fail without the legacy runtime. Verified: 100% GPU offload on the RTX 5050 (Blackwell, compute capability 12.0) with the `cuda_v13` libraries. Details are in `docs/gpu-setup-notes.md`.
- Variants: `compose.cpu.yml` (no GPU) and `compose.dev.yml` (Chroma only, with host Ollama and hot reload).
- Design decision: the spec doesn't require the LLM inside Docker, but full containerization is more portable. Trade-off: about 5.6 GB of models are re-downloaded into a volume, and the host Ollama must be stopped to free VRAM.
- Screenshots: `docker compose ps`, and the UI working at `localhost:3000`.

## 6. Task 4: User interface (~150 words)
- **Chat:** streaming answers, `[n]` citation chips that expand the exact source chunk (file, page, heading, score), a relevance score vs the threshold, a model picker, history in localStorage (the last 3 exchanges are sent), stop and clear buttons.
- **Knowledge base:** a document list with upload and delete, a chunk browser (filter by page or text, raw chunk text and metadata), and a retrieval tester (top-k with cosine, dense and BM25 ranks, and whether the bot would refuse).
- Screenshots: the chat with citations open, a refusal, the chunk browser and the retrieval tester.

## 7. Prompt design (~100 words)
- Show the system prompt (`backend/rag/generation.py`) next to the baseline prompt. Key differences: no general-knowledge fallback, mandatory `[n]` citations, an exact not-found sentence, partial-answer handling, reply in the question's language, history used only to resolve references.
- Observation from the evaluation: how well each model follows the rules (qwen 0.5b vs llama 8b).

## 8. Limitations and failure cases (~150 words)
- Tables are linearised and lose spaces, so numeric and table questions are fragile.
- No OCR; the image on page 10 is invisible.
- English-only embeddings, so Vietnamese retrieval is weaker. Future fix: `bge-m3`.
- The relevance score measures retrieval, not correctness, so a passing question can still get a wrong answer.
- Small models ignore citation and grounding instructions.
- The BM25 index is in-memory; there's no auth and no multi-user support.

## 9. Lessons learned and future work (~150 words)
- Look at the chunks before tuning the LLM: the chunk inspector found the heading bug, which no prompt tweak would have fixed.
- Measure the baseline's assumptions: the dead relevance gate.
- An embed-then-swap ingest is needed for robustness; the first version deleted the index when Ollama was down.
- Future work:
  - Query rewriting for follow-ups
  - A cross-encoder reranker
  - Table-aware chunking
  - A multilingual embedder
  - Chunk editing in the UI (designed as a nice-to-have)
  - Automatic LLM-as-judge evaluation

## 10. Conclusion (~80 words)

## Appendix
- How to run: link to the README. GitHub URL: https://github.com/Thai-Anh-Nguyen/Local-RAG-System
- Demo video links (Tasks 1–4)
