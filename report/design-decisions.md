# Design decisions — Local RAG system (CS311 final project)

Settled in the design interview on 2026-11-17. Source spec: `final-project/Final-project-ADL.pdf`.

## Scope
- Build a clean version that meets the spec first, then extend it.
- Task 2 handbook: **CS_MajorHandbook.pdf** (19 pages, about 5.5k words, all text selectable, so no OCR is needed).
- 4-hour budget. Claude builds the code, smoke tests, evaluation run and report skeleton. The student records the videos and writes the final report prose.

## Stack
- LangChain + ChromaDB + Ollama (same as the baseline, so the comparison stays fair).
- Python 3.12 via `uv`; conda is not used.
- Minimal dependencies: fastapi, uvicorn, langchain-core, langchain-text-splitters, langchain-ollama, langchain-chroma, chromadb (client), pymupdf4llm, pypdf, pydantic-settings, python-multipart, sse-starlette; pytest + httpx for development.
- Frontend: React + Vite + TypeScript, Tailwind v4, react-markdown. nginx serves it and proxies `/api`.
- Chroma runs as its own container and the API connects over HTTP.

## Models
- LLM default: `llama3.1:8b`. Baseline for comparison: `qwen2.5:0.5b`. A model picker in the UI, filtered by an allowlist in `.env`.
- Embeddings: `nomic-embed-text` with the `search_document: ` / `search_query: ` task prefixes (the baseline omits them).
- `num_ctx=8192` is set explicitly. Temperature is 0.1.

## Run modes
- **Full** (default): `docker compose up` runs frontend + api + chroma + ollama (GPU via CDI), with an `ollama-init` service that pulls the models. The container's Ollama is not published on host port 11434. Stop the host `ollama` service first to free VRAM.
- **Dev**: only chroma runs in Docker; the API runs with `uv run` and the frontend with `vite dev`, both against host Ollama at 127.0.0.1:11434.
- **CPU fallback**: a `compose.cpu.yml` override for machines without an NVIDIA GPU.

## Ingestion
- PyMuPDF (`pymupdf4llm`) is the default parser; pypdf can be selected via `.env`. `.pdf`, `.md` and `.txt` are supported.
- Per-page markdown; page number and nearest heading are kept as metadata.
- `RecursiveCharacterTextSplitter` at 1000/200 (same as the baseline; tuned later during evaluation).
- Chunk ID = sha1(file content hash + page + chunk index). Re-ingesting an unchanged file does nothing; a changed file replaces its chunks.
- Three ways in: the CLI `python -m rag.ingest <file|dir>... [--reset]`, auto-ingest at API startup when the collection is empty, and `POST /api/documents` upload.

## Answer policy (baseline → fix)
| Baseline (`run_txt.py`) | Fix |
|---|---|
| Temperature 0.7 | 0.1 |
| "If context is insufficient, use your general knowledge" | Answer only from the context; otherwise say the handbook doesn't cover it |
| L2 distance and `1/(1+d)` averaged over top-5 vs. threshold 0.3. Likely never fires, because the score is at least 0.33 for unit vectors (*verify on real data*) | Cosine similarity, **top-1** score; below the threshold the API refuses **without calling the LLM**; threshold calibrated from observed scores |
| Sources show the filename only | Numbered `[n]` citations with file, page, heading and snippet |
| Embeddings without nomic prefixes | Prefixes applied |

The UI labels the score "retrieval relevance". It measures whether the retrieved text is relevant, not whether the answer is correct.

## API (`/api`, no auth — local only)
- `GET /health`
- `POST /chat` → SSE: `sources` → `token`* → `done` | `refusal` | `error`; body `{question, history≤3, model?}`
- `POST /retrieve` → top-k chunks with scores, no LLM call
- `GET /documents`, `POST /documents` (upload), `DELETE /documents/{source}`
- `GET /chunks?source=&page=&contains=&limit=&offset=`, `GET /chunks/{id}`
- Nice-to-have (built last): `PATCH /chunks/{id}` (re-embed, `edited` badge), `POST /chunks` (manual note), `DELETE /chunks/{id}`. Edits are lost if the source file changes and is re-ingested.

## Conversation history
- Stored in the browser (localStorage). The client sends the last 3 turns.
- The retrieval query is the current question plus the previous user question. LLM query rewriting is deferred.

## Repo layout (repo root = submission root `ADL-240039-project`)
```
source_code/  pyproject.toml, uv.lock, .env.example, backend/{rag,api}, frontend/,
              test/{unit,api,e2e}, data/handbooks/, scripts/package_submission.sh
docker/       Dockerfiles, nginx.conf, compose.yml, compose.cpu.yml
report/       this file, report draft, eval results
demo_video/
README.md
```

## Build order (each step ends with a commit and a passing test)
1. Scaffold
2. Parse + chunk + ingest CLI, with unit tests
3. Chroma + embeddings + `/retrieve` `/chunks` `/documents` → **joint retrieval check with the student**
4. Generation + SSE `/chat` + threshold calibration, with API tests
5. Frontend: chat page, then the Knowledge base page and retrieval tester
6. Docker (full GPU / dev / CPU) + end-to-end smoke test
7. README
8. Evaluation set (about 25 questions, hit@k plus manual grading) + run + report skeleton
9. Nice-to-have: chunk editing
10. Packaging script → `ADL-240039-project.zip`

## Notes for the report
- **Known limitation, multilingual:** the bot replies in the question's language, but `nomic-embed-text` and the handbook are English, so Vietnamese questions retrieve worse. Future fix: a multilingual embedding model such as `bge-m3`.
- **Baseline relevance gate is likely dead code** (see the table above); verify and quantify.
- **Missing nomic task prefixes in the baseline;** measure the hit@k difference with and without them.
- **pypdf vs. PyMuPDF on course-requirement tables;** show an example chunk from each.
- **Host Ollama vs. containerized Ollama tradeoff:** the spec doesn't require the LLM inside Docker, but full containerization is more portable. The GPU needs nvidia-container-toolkit (CDI on Fedora with SELinux).
- **Confidence score scope:** it's retrieval relevance, not answer correctness.
