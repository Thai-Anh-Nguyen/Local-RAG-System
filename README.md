# Local RAG Handbook Assistant

A chatbot for the **Fulbright University Vietnam B.S. Computer Science major handbook** that runs entirely on your own machine: retrieval-augmented generation with Ollama, ChromaDB, FastAPI and a React web UI, packaged with Docker. Built as the CS311 Advanced Deep Learning final project.

- Answers **only from the handbook**, with numbered citations (`[1]`, `[2]`) linked to the exact page and section.
- **Declines** questions the handbook can't answer, and does so *without* calling the LLM, so it can't make something up.
- **Hybrid retrieval**: vector search (nomic-embed-text) plus BM25 keyword search, fused with Reciprocal Rank Fusion. The keyword side finds exact course codes like `CS311`.
- Ingests **PDF, Markdown and text**. Upload from the UI or use the CLI.
- **Knowledge base inspector**: browse every chunk as text, and see what any question retrieves, with scores, before the LLM is involved.
- Streams answers token by token. The browser keeps the conversation history, and follow-up questions use the earlier context.

## Contents

- [Architecture](#architecture)
- [Quick start (Docker)](#quick-start-docker)
- [Development mode](#development-mode)
- [Ingesting documents](#ingesting-documents)
- [Asking from the terminal](#asking-from-the-terminal)
- [Configuration](#configuration)
- [API](#api)
- [Tests](#tests)
- [Project structure](#project-structure)
- [What changed vs. the baseline](#what-changed-vs-the-baseline)
- [Known limitations](#known-limitations)

## Architecture

![Local RAG system flowchart: Browser (React + Vite) → nginx → FastAPI (rag core, in-process BM25) → ChromaDB vector search and Ollama embeddings/generation](docs/images/architecture.webp)

**Ingestion:** PDF → PyMuPDF (markdown with headings and tables) → ligature repair → heading-aware chunks (1000 chars, 200 overlap, each starting with its `[Section > Heading]` breadcrumb) → `search_document:`-prefixed embeddings → Chroma. Chunk IDs are stable hashes, so re-ingesting an unchanged file does nothing.

**Question answering:**
1. The retrieval query is the question plus the previous user question, so follow-ups like "what about its prerequisites?" still work.
2. Hybrid retrieval returns the top 5 chunks.
3. If the best cosine similarity is below the threshold (0.62), the API returns a **refusal** without calling the LLM.
4. Otherwise it builds a grounded prompt (numbered excerpts, cite-or-say-not-found rules, last 3 exchanges) and streams the answer from llama3.1:8b at temperature 0.1.

| Service | Image | Role |
|---|---|---|
| `frontend` | nginx + built React app | UI on **http://localhost:3000**, proxies `/api` |
| `api` | python:3.12-slim + uv | FastAPI, ingestion, retrieval, prompting |
| `chroma` | chromadb/chroma:1.5.9 | Vector store (named volume) |
| `ollama` | ollama/ollama:0.40.16 | Embedding + LLM inference (GPU via CDI) |
| `ollama-init` | ollama/ollama:0.40.16 | One-shot model pull |

## Quick start (Docker)

**Requirements:** Docker with Compose v2, about 20 GB free disk (images plus about 5.6 GB of models). An NVIDIA GPU with 8 GB+ VRAM is recommended; there's a CPU fallback.

```bash
git clone https://github.com/Thai-Anh-Nguyen/Local-RAG-System.git
```
```bash
cd Local-RAG-System
```

### With an NVIDIA GPU

One-time host setup: install the NVIDIA Container Toolkit and generate a CDI spec. Shown for Fedora/RHEL; on Ubuntu/Debian, install `nvidia-container-toolkit` from the same NVIDIA repo with apt.

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
```
```bash
sudo dnf install -y nvidia-container-toolkit
```
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

Check that containers can see the GPU:

```bash
docker run --rm --device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi
```

If Ollama is also installed on the host, stop it so the two don't compete for VRAM:

```bash
sudo systemctl stop ollama
```

Start the stack:

```bash
docker compose -f docker/compose.yml up -d --build
```

The first start pulls `nomic-embed-text`, `llama3.1:8b` and `qwen2.5:0.5b`. Follow it with `docker compose -f docker/compose.yml logs -f ollama-init`. When it finishes, the API indexes the bundled handbook and the UI is at **http://localhost:3000**.

> Re-run `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` after every NVIDIA driver update. `--gpus all` and the `driver: nvidia` compose syntax need the legacy NVIDIA runtime; this repo uses CDI. Details are in [docs/gpu-setup-notes.md](docs/gpu-setup-notes.md).

### Without a GPU

```bash
docker compose -f docker/compose.yml -f docker/compose.cpu.yml up -d --build
```

llama3.1:8b is slow on CPU. Pick `qwen2.5:0.5b` in the UI's model menu for quicker, if weaker, answers.

### Useful commands

```bash
docker compose -f docker/compose.yml ps
```
```bash
docker compose -f docker/compose.yml logs -f api
```
```bash
docker compose -f docker/compose.yml down
```

`down` keeps the models and the vector index in named volumes. Add `-v` to delete them too. Uploaded documents are stored on the host in `source_code/data/handbooks/uploads/`, which is mounted into the API container.

## Development mode

Run the API and UI on the host with hot reload, using a local Ollama at `127.0.0.1:11434`. Only Chroma runs in Docker.

**Requirements:** [uv](https://docs.astral.sh/uv/), Node.js 20+, [Ollama](https://ollama.com/download), Docker.

```bash
ollama pull nomic-embed-text && ollama pull llama3.1:8b && ollama pull qwen2.5:0.5b
```
```bash
docker compose -f docker/compose.dev.yml up -d
```
```bash
cd source_code && cp .env.example .env && uv sync
```
```bash
uv run uvicorn api.main:app --reload --port 8080
```

In a second terminal:

```bash
cd source_code/frontend && npm install && npm run dev
```

Open **http://localhost:5173**. Vite proxies `/api` to port 8080. The API auto-ingests `data/handbooks/` on first start.

## Ingesting documents

The API indexes everything in `source_code/data/handbooks/` when the collection is empty. To add or re-index documents:

- **UI:** *Knowledge base → Upload PDF / Markdown / text*. Remove a document with the ✕ next to it.
- **CLI** (from `source_code/`, dev mode):

```bash
uv run python -m rag.ingest data/handbooks
```
```bash
uv run python -m rag.ingest path/to/any.pdf notes.md --parser pypdf
```
```bash
uv run python -m rag.ingest data/handbooks --reset
```

With the Docker stack, run the same CLI inside the API container (from the repo root):

```bash
docker compose -f docker/compose.yml exec api python -m rag.ingest data/handbooks --reset
```

To see how a file is parsed and chunked without touching the database, use the following command:

```bash
uv run python scripts/inspect_chunks.py data/handbooks/CS_MajorHandbook.pdf --grep "Capstone" --show 2
```

Files are fingerprinted by their content plus the parser and chunk settings. An unchanged file is skipped, and a changed file, or a changed `RAG_CHUNK_SIZE`, `RAG_PDF_PARSER` and so on, is re-indexed. Embedding happens before the old chunks are removed, so a failed re-index leaves the previous version intact.

## Asking from the terminal

`scripts/ask.py` calls the same `/api/chat` endpoint as the web UI and streams the answer and its sources. Run it from `source_code/`:

```bash
uv run python scripts/ask.py "What are the prerequisites for CS311?"
```
```bash
uv run python scripts/ask.py --model qwen2.5:0.5b "What is the minimum major GPA to enroll in the Capstone?"
```
```bash
uv run python scripts/ask.py
```
Run it with no question for an interactive chat with history. Add `--url http://localhost:8080` to use the dev-mode API instead of Docker.

## Configuration

All settings are environment variables with the `RAG_` prefix, read from `source_code/.env`. Start from [`.env.example`](source_code/.env.example). In Docker the same file is used, and the service addresses are overridden in `compose.yml`.

| Variable | Default | Meaning |
|---|---|---|
| `RAG_LLM_MODEL` | `llama3.1:8b` | Default answer model |
| `RAG_LLM_ALLOWLIST` | `llama3.1:8b,qwen2.5:0.5b` | Models selectable in the UI (only pulled ones are shown) |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_EMBED_PREFIXES` | `true` | Add nomic's `search_document:` / `search_query:` prefixes |
| `RAG_TEMPERATURE` | `0.1` | Sampling temperature |
| `RAG_NUM_CTX` | `8192` | LLM context window (Ollama's default would truncate) |
| `RAG_PDF_PARSER` | `pymupdf` | `pymupdf` (headings and tables) or `pypdf` (plain text) |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `200` | Characters per chunk / overlap |
| `RAG_RETRIEVAL_MODE` | `hybrid` | `hybrid` (vector + BM25, RRF) or `dense` (vector only) |
| `RAG_TOP_K` | `5` | Chunks given to the LLM |
| `RAG_RELEVANCE_THRESHOLD` | `0.62` | Below this best cosine similarity, refuse without calling the LLM |
| `RAG_HISTORY_TURNS` | `3` | Previous Q/A pairs included in the prompt |
| `RAG_OLLAMA_URL`, `RAG_CHROMA_HOST`, `RAG_CHROMA_PORT`, `RAG_COLLECTION`, `RAG_DATA_DIR` | see `.env.example` | Service addresses and paths |

To change which models Docker pulls, edit [`docker/models.env`](docker/models.env).

## API

Interactive docs: `http://localhost:8080/docs` in dev mode.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Ollama, Chroma and model availability |
| `GET` | `/api/models` | Allowlisted models that are actually pulled |
| `POST` | `/api/chat` | `{question, history?, model?}`, returned as a Server-Sent Events stream: `sources` → `token`… → `done`, or a single `refusal` / `error` |
| `POST` | `/api/retrieve` | `{question, k?, mode?}`: top-k chunks with cosine scores, dense/BM25 ranks and the refusal decision (no LLM call) |
| `GET` | `/api/documents` | Indexed documents with page and chunk counts |
| `POST` | `/api/documents` | Multipart upload (`.pdf`, `.md`, `.txt`), indexed immediately |
| `DELETE` | `/api/documents/{source}` | Remove a document and its chunks |
| `GET` | `/api/chunks` | Chunk text and metadata; filters `source`, `page`, `contains`, plus `limit` and `offset` |
| `GET` | `/api/chunks/{id}` | One chunk |

There is no authentication. The app is meant to run locally.

## Tests

```bash
cd source_code && uv run pytest
```

Unit and API tests use an in-memory Chroma, fake embeddings and a fake LLM, so they need no running services. They cover parsing, ligature repair, chunk headings and IDs, embedding prefixes, hybrid retrieval, the SSE event order, refusals, uploads and the chunk endpoints.

Against a running stack:

```bash
uv run python test/e2e/smoke.py --url http://localhost:3000
```
```bash
uv run python test/e2e/retrieval_probe.py --url http://localhost:8080 --mode dense
```

`smoke.py` checks health, cited answers to five handbook questions and an off-topic refusal. `retrieval_probe.py` prints the top-k chunks and scores for a fixed question set, with hit@k and the score ranges used to calibrate the refusal threshold.

## Project structure

```
├── source_code/
│   ├── backend/
│   │   ├── rag/            # framework-free core: parsing, chunking, embeddings, store, retrieval, generation, ingest CLI
│   │   └── api/            # FastAPI app: routes_chat.py (SSE), routes_kb.py (health/docs/chunks/retrieve)
│   ├── frontend/           # React + Vite + TypeScript + Tailwind (Chat and Knowledge base pages)
│   ├── scripts/            # ask.py (terminal chat), inspect_chunks.py (parse/chunk viewer)
│   ├── test/               # unit/, api/ (pytest), e2e/ (smoke + retrieval probe), eval/
│   ├── data/handbooks/     # CS_MajorHandbook.pdf (+ uploads/)
│   ├── pyproject.toml, uv.lock, .env.example
├── docker/                 # Dockerfiles, nginx.conf, compose.yml, compose.cpu.yml, compose.dev.yml, models.env
├── docs/                   # GPU (CDI) setup notes, video recording commands
├── report/                 # design decisions, evaluation results, report
└── demo_video/             # demo links
```

## What changed vs. the baseline

The course's `run_txt.py` is the starting point.

| Area | Baseline | This project |
|---|---|---|
| Documents | `.txt` only | `.pdf` (PyMuPDF or pypdf), `.md`, `.txt`; CLI and UI upload |
| Chunking | Fixed 1000/200 characters, no structure | Heading-aware sections, then 1000/200; page and heading metadata and a breadcrumb per chunk |
| Embeddings | nomic-embed-text without task prefixes | `search_document:` / `search_query:` prefixes |
| Retrieval | Vector only (L2 distance) | Hybrid vector + BM25 with RRF, cosine similarity |
| Relevance gate | Average of `1/(1+L2)` over the top 5 vs 0.3; never fires (0/30 in the evaluation) | Best cosine of the current question vs a calibrated 0.62; refusal skips the LLM |
| Prompt | Fall back to "general knowledge" | Grounded: excerpts only, numbered citations, explicit not-found answer |
| Temperature | 0.7 | 0.1 |
| Interface | CLI, one question | Web UI with streaming, citations, history, model picker, knowledge-base inspector |
| Deployment | Manual | Docker Compose (GPU, CPU and dev variants) |

## Evaluation results

30 handbook questions; details are in [report/eval/](report/eval/).

| Retrieval (27 answerable, k = 5) | hit@5 | MRR |
|---|---|---|
| Baseline (pypdf, plain chunks, L2) | 20/27 | 0.540 |
| + heading-aware PyMuPDF chunks | 23/27 | 0.723 |
| + nomic prefixes | 23/27 | 0.725 |
| + hybrid BM25 (shipped) | **25/27** | **0.760** |

| Answers (auto heuristics) | key facts present | unanswerable declined | valid citations | median latency |
|---|---|---|---|---|
| Baseline + qwen2.5:0.5b | 13/27 | 0/3 | none | 6.6 s |
| Final + qwen2.5:0.5b | 13/27 | 2/3 | 3/26 | 7.9 s |
| Final + llama3.1:8b | **23/27** | **3/3** | **26/26** | **1.6 s** |

## Known limitations

- **Text in images is not indexed.** There is no OCR; for example, the curriculum diagram on page 10 of the handbook yields no text.
- **PDF extraction artifacts.** Some words lose their spaces inside tables ("Once ayear", "DeepLearning"). Broken ligatures ("Descrip8on") are repaired, but other glyph problems may remain.
- **Vietnamese questions** get answers in Vietnamese, but retrieval is weaker because nomic-embed-text and the handbook are English. A multilingual embedder such as `bge-m3` would help.
- **The relevance score measures retrieval, not answer correctness.** A question can pass the threshold and still be answered imperfectly by the LLM.
- **The refusal gate is a coarse filter.** Off-topic and handbook questions overlap in similarity: off-topic scored 0.46–0.64, and the weakest handbook question 0.636. At 0.62 the gate refuses most off-topic questions, and the grounded prompt catches the rest with llama3.1:8b. In-domain questions the handbook doesn't answer (tuition, staff) always reach the LLM. See [report/eval/gate-calibration.md](report/eval/gate-calibration.md). Re-calibrate if you change the embedding model or the Ollama version.
- **Small models** (`qwen2.5:0.5b`) often ignore citation rules or misread tables. This is shown in the evaluation.
- The BM25 index is rebuilt in memory when the collection changes; that's fine for handbook-sized collections.
- There is no authentication or multi-user isolation; it's intended for local use.
