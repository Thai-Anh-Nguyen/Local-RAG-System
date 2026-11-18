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
- PyMuPDF vs pypdf: PyMuPDF keeps `#` headings and markdown tables; pypdf gives flat text. **Measured:** baseline (pypdf, plain splitter) gives 55 chunks; structured (PyMuPDF, heading-aware) gives 69 smaller chunks that each belong to one section or course. Parsing takes about 25 s on CPU for PyMuPDF's layout model, and is near-instant for pypdf.
- A second bug found with the chunk inspector: some course titles are bold text rather than headings ("**CS307: Machine Learning**"), so the ML course description was labeled "CS306: Computer Security", and citations would have pointed to the wrong course. Fixed by also detecting bold lines that start with a course code (unit-tested).
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
27 answerable questions, k = 5. A "hit" means a retrieved chunk contains the question's evidence string.

| Config | hit@1 | hit@5 | MRR |
|---|---|---|---|
| A: baseline (pypdf, plain chunks, no prefix, L2) | 11/27 | 20/27 | 0.540 |
| B: + PyMuPDF heading-aware chunks (dense) | 18/27 | 23/27 | 0.723 |
| C: + nomic task prefixes (dense) | 18/27 | 23/27 | 0.725 |
| D: + hybrid BM25/RRF, stemmed (**shipped**) | 17/27 | **25/27** | **0.760** |

What the per-question table (`retrieval.md`) shows:
- **Structure is the biggest win** (hit@1 rises from 11 to 18). R1 (56 credits), C3 (Java) and O6 (Stanford/CMU/Cornell) go from miss to rank 1. Breadcrumbs put the section name in every chunk, and the sections no longer mix.
- **Prefixes changed almost nothing** here (MRR 0.723 → 0.725). This is an honest negative result: the documented nomic best practice gave no measurable gain on this small corpus.
- **Hybrid** rescues R8 (honors, miss → 2) and C6 (cross-listed MATH308/ENG301, miss → 2), both exact-term matches. It also improves C5, O1 and R6. Cost: R2 drops from rank 1 to rank 5, so hit@1 falls by 1. Trade-off: better recall, slightly worse precision at 1.
- Two questions are missed in all four configurations:
  - **R10** (names of the core courses): the answer sits in a numbered list item under "General Curriculum Requirements" that the question's wording doesn't match.
  - **V1** (Vietnamese): the English-only embedder can't match it.
- Story from development worth telling: before the fixes, a retrieval probe showed "prerequisites for CS311" missing the CS311 chunk entirely, and "what does Operating Systems cover" ranking CS210 Networks first. Hybrid search initially *hurt* "How are students assessed?" because BM25 had no stemming ("assessed" ≠ "assessment") and boosted generic words. Adding a Snowball stemmer fixed it. This was found with the UI's retrieval tester.

### 4.3 Relevance gate
- **Baseline gate is dead code (confirmed).** The mean of `1/(1+L2)` over the top 5 ranged from **0.453 to 0.691** across all 30 questions, never near its 0.3 threshold. It **fired on 0/30**: even "What is the capital of France?" (0.453) passed. Reason: nomic vectors are about unit length, so L2 distance ≤ 2 and the score is ≥ 1/3 by construction. In practice the baseline's "use general knowledge" path also ran whenever the prompt told the model it could, which is how it answered "Paris" and invented a tuition structure.
- **Final gate:** the best cosine of the *current* question's top chunks, against a threshold of 0.62. Full data is in `report/eval/gate-calibration.md`.
  - English handbook questions: the lowest is 0.636 (R8), the median about 0.79.
  - Off-topic questions span 0.46–0.64, which **overlaps** with them.
  - 0.62 refuses 8 of 10 off-topic questions and no English handbook question. The two that pass ("laptop", "transformers") were declined by llama's grounded prompt.
  - Wrongly refused: only V1 (Vietnamese, 0.546). Its retrieval had already failed, so refusing beats a wrong answer.
  - In-domain questions the handbook can't answer (tuition 0.667, head of department 0.660) can't be separated by any threshold; the prompt must handle them.
- **Bug found during calibration:** the gate first scored the history-augmented query, so an off-topic question right after an on-topic one inherited its relevance (World Cup 0.704, sourdough 0.737, both passing). It was fixed by gating on the current question alone. Report this; it's a good "lesson learned".
- Caveat: the scores depend on the embedding runtime. The host's Ollama 0.21 gave off-topic scores about 0.1 higher than the container's 0.40.16, so re-calibrate whenever the runtime or model changes.

### 4.4 Answer quality: baseline vs final, small vs large model
Automatic heuristics (from `summary.md`). **"facts ok" is a string match and is unreliable**: for example, the baseline's C3 answer lists five languages including Java and is marked ok, but it's wrong. Replace these with your manual grades.

| Setup | facts ok (27) | unanswerable declined (3) | wrongly refused | valid citations | median latency (RTX 5050) |
|---|---|---|---|---|---|
| baseline: qwen2.5:0.5b, temp 0.7, general-knowledge fallback | 13 | **0** | 0 | no citations | 6.6 s |
| final pipeline + qwen2.5:0.5b | 13 | 2 | 1 (V1) | 3/26 | 7.9 s |
| final pipeline + **llama3.1:8b** (default) | **23** | **3** | 1 (V1) | **26/26** | **1.6 s** |

Takeaways:
- The pipeline fixes **safety**: the baseline answered all 3 unanswerable questions (it said Paris, invented a tuition structure, and called the handbook the department head), while the final pipeline declined them.
- **Model size drives accuracy and instruction-following.** qwen 0.5b, even with the good pipeline, ignores citation rules (3/26 valid). It dumps raw excerpts (C1, C5) and makes things up (C6 lists "CS 314…CS 336"; U2 "The head … is [1] [2][3]"). llama 8b cites validly every time.
- Latency: llama is *faster* than both qwen setups at the median, because qwen rambles while llama gives short grounded answers. Warm llama runs at about 53 tokens/s, 100% on the GPU.

**Suggested examples for the brief's four categories** (verify against `answers.md` and use your own judgment):
- **Correct:**
  - C1: "The prerequisites for CS311: Advanced Deep Learning are CS313. [1]"
  - F1: the follow-up "What are its prerequisites?" after a CS313 question gives "CS208 [1]", so history-aware retrieval worked.
  - U1/U2: declined, which is correct for unanswerable questions.
- **Incomplete:**
  - R6: names two of the three Capstone alternatives and leaves out "one intermediate + one advanced" and the honors consequence.
  - C4: lists 3 of the 5 "on holding" courses (misses CS210 and CS3xx); table rows are split across chunks.
  - O4: lists only summative assessments and misses the formative-assessment principle.
- **Incorrect (small models / baseline):**
  - baseline R1: "requires at least 8 credits" (should be 56).
  - final-qwen R8: "Yes, you can graduate with honors" without the capstone (should be no).
  - baseline U3: answers "Paris" from general knowledge.
- **Ambiguous:**
  - T1: llama faithfully answers "Bachelor of Science in Applied Mathematics", which is what the handbook says (an apparent copy-paste error in the source). The system was faithful to a wrong document.
  - R8 (llama): it hedged ("does not explicitly state…") even though the handbook implies the answer, a cautious and partly correct reply.
- **Retrieval-caused failures:**
  - R10: the core courses weren't retrieved, and llama honestly says they "are not explicitly listed", which is grounded but incomplete.
  - V1: the Vietnamese question was refused because retrieval failed.

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
- History helps retrieval but must not leak into the relevance gate (the off-topic follow-up bug).
- Hybrid search needs normal IR hygiene: without stemming, BM25 made one question worse.
- Thresholds are runtime-specific: an Ollama upgrade shifted the cosine scores by about 0.1.
- Measure, don't assume: nomic's prefixes, a documented best practice, gave about +0.002 MRR here.
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
