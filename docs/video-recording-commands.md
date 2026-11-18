# Video recording commands (Tasks 1–3)

Every command below was dry-run on 2026-11-18 against the Docker GPU stack. Two working directories are used:

- **repo root** (`RAG project/`): `docker compose` commands.
- **`source_code/`**: `uv run` commands.

## Before you record
```bash
sudo systemctl stop ollama                                   # free the GPU for the Ollama container
docker compose -f docker/compose.yml up -d                   # (repo root) stack running
curl -s localhost:3000/api/health | jq '{status, chunks: .chroma.chunks}'   # expect "ok", 69
```
Warm the models before you start, so the first on-camera answer isn't a 20 s cold load:
```bash
cd source_code && uv run python scripts/ask.py --no-sources "How many credits does the CS major require?"
uv run python scripts/ask.py --no-sources --model qwen2.5:0.5b "How many credits does the CS major require?"
```
A tip: `clear` between steps, and use a large terminal font.

---

## Task 1: PDF processing and retrieval
Goal: show that a PDF goes from parsing to chunking to embedding to storage, and that retrieval returns the right chunks.

**1. The source PDF** (repo root)
```bash
ls -lh source_code/data/handbooks/
```

**2. Parse and chunk (offline, no database)** (`source_code/`)
```bash
uv run python scripts/inspect_chunks.py data/handbooks/CS_MajorHandbook.pdf --show 2
```
Prints 18 pages / about 40k characters, then 69 chunks across 8 sections (median 721 chars). Ignore the pymupdf "RapidOCR is missing" line: the PDF is text-based, so no OCR is needed.

```bash
uv run python scripts/inspect_chunks.py data/handbooks/CS_MajorHandbook.pdf --raw-page 8 --show 0
```
Prints page 8 as parsed (Markdown headings and bullets are kept).

```bash
uv run python scripts/inspect_chunks.py data/handbooks/CS_MajorHandbook.pdf --grep "minimum major GPA" --show 2
```
Prints a chunk with its `[Section > Heading]` breadcrumb, page number and id.

**3. Compare with the baseline parser** (optional, good talking point)
```bash
uv run python scripts/inspect_chunks.py data/handbooks/CS_MajorHandbook.pdf --parser pypdf --show 1
```
pypdf gives 55 chunks and only **1 section**. It loses the headings, so chunks have no breadcrumb.

**4. Embed and store in Chroma (the ingest CLI)** (repo root)
```bash
docker compose -f docker/compose.yml exec api python -m rag.ingest data/handbooks --reset
```
`indexed CS_MajorHandbook.pdf (18 pages, 69 chunks)` takes about 20 s.

```bash
docker compose -f docker/compose.yml exec api python -m rag.ingest data/handbooks
```
`unchanged … (already indexed, skipped)`: re-ingest is idempotent, using a fingerprint of the file plus the chunking settings.

**5. Inspect what is stored**
```bash
curl -s "localhost:3000/api/chunks?contains=Capstone&limit=2" | jq '.total, (.items[] | {page, section, heading, text: .text[:160]})'
```

**6. Retrieval works**: top chunks with cosine scores, no LLM involved.
```bash
curl -s localhost:3000/api/retrieve -H 'Content-Type: application/json' -d '{"question":"What is the minimum GPA to enroll in the Capstone?","k":3}' | jq '{top_score, threshold, would_refuse, hits: [.hits[] | {n, score, page, heading}]}'
```
```bash
curl -s localhost:3000/api/retrieve -H 'Content-Type: application/json' -d '{"question":"What is the minimum GPA to enroll in the Capstone?","k":1}' | jq -r '.hits[0].text'
```
The text contains "minimum major GPA of 3.5".

```bash
cd source_code && uv run python test/e2e/retrieval_probe.py --url http://localhost:3000
```
This runs 10 handbook questions and 3 off-topic ones, showing the rank of the expected chunk and how low the off-topic scores are.

---

## Task 2: handbook chatbot from the terminal
`scripts/ask.py` calls the same `/api/chat` endpoint as the web UI and streams the answer. Run it from `source_code/`.

**Correct answers** (with citations and sources)
```bash
uv run python scripts/ask.py "What are the prerequisites for CS311?"
```
```bash
uv run python scripts/ask.py "How many credits are required for the Computer Science major?"
```

**Incomplete answer.** The bot hedges instead of stating that honors require the capstone.
```bash
uv run python scripts/ask.py "Can I graduate with honors if I don't do the capstone?"
```

**Ambiguous questions.** These are under-specified, so the bot picks one reading and dumps everything.
```bash
uv run python scripts/ask.py "What are the requirements?"
```
```bash
uv run python scripts/ask.py "What courses should I take?"
```

**Incorrect answer (small model).** qwen2.5:0.5b answered with only citation markers, for example `[1] [2][3]`.
```bash
uv run python scripts/ask.py --model qwen2.5:0.5b "What is the minimum major GPA to enroll in the Capstone?"
```
Compare it with llama on the same question:
```bash
uv run python scripts/ask.py --model llama3.1:8b "What is the minimum major GPA to enroll in the Capstone?"
```

**Not in the handbook.** This passes the gate, then the LLM declines.
```bash
uv run python scripts/ask.py "How much is tuition for the CS program?"
```
Expect "I couldn't find this in the handbook."

**Off-topic.** The gate refuses it before the LLM is called.
```bash
uv run python scripts/ask.py "What is the capital of France?"
```
Expect `[refused before the LLM: relevance 0.500 < threshold 0.62]`.

**Follow-up with history** (interactive mode; type the lines below, then `exit`)
```bash
uv run python scripts/ask.py --no-sources
```
```
How many credits does the CS major require?
What about the minor requirements?
```
"And the minor?" on its own is refused (0.609). This is the known edge case for very short follow-ups, and worth showing as a limitation.

**Evaluation results** (the full run takes minutes, so show the saved results):
```bash
cat ../report/eval/summary.md
```
To re-run it: `uv run python test/eval/evaluate.py --ollama http://127.0.0.1:11435 --chroma-port 8000`.

Answers vary slightly between runs (temperature 0.1), so rehearse each question once before recording.

---

## Task 3: Docker deployment
Run everything from the repo root.

**1. Prerequisites: GPU visible to Docker via CDI**
```bash
docker --version && docker compose version
```
```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```
```bash
nvidia-ctk cdi list
```
```bash
sudo systemctl stop ollama
```
The host Ollama is stopped so the container gets the GPU.

**2. The configuration** (scroll through it briefly)
```bash
cat docker/compose.yml
```
The stack is 5 services: `ollama` (GPU via `devices: nvidia.com/gpu=all`), `ollama-init` (pulls the models), `chroma`, `api` and `frontend`.

```bash
cat docker/models.env
```

**3. Fresh start**
```bash
docker compose -f docker/compose.yml down
```
Never add `-v` here: it would delete the 5.5 GB of downloaded models.

```bash
docker compose -f docker/compose.yml up -d --build
```
```bash
docker compose -f docker/compose.yml ps
```
Expect `ollama` and `api` (healthy), `chroma`, `frontend`, and `ollama-init` exited (0).

```bash
docker compose -f docker/compose.yml logs ollama-init
```
```bash
docker compose -f docker/compose.yml logs api | tail -20
```
These show the startup ingest and Uvicorn running.

**4. Health and GPU use**
```bash
curl -s localhost:3000/api/health | jq
```
```bash
cd source_code && uv run python scripts/ask.py --no-sources "What are the prerequisites for CS311?" && cd ..
```
```bash
docker compose -f docker/compose.yml exec ollama ollama ps
```
Shows `llama3.1:8b … 100% GPU`.

```bash
docker compose -f docker/compose.yml exec ollama nvidia-smi
```

**5. End-to-end smoke test**
```bash
cd source_code && uv run python test/e2e/smoke.py && cd ..
```
Prints ALL PASSED: health, 5 cited answers, and 1 refusal.

**6. The app** is at http://localhost:3000. That is Task 4's video, but end Task 3 by showing that it loads.

**CPU-only variant** (mention it, no need to run it): `docker compose -f docker/compose.yml -f docker/compose.cpu.yml up -d --build`

**Stop**
```bash
docker compose -f docker/compose.yml down
```
