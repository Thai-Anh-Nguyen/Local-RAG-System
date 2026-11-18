#!/usr/bin/env bash
# Task 1 demo: PDF processing -> chunks -> embeddings in the Chroma vector database.
# Needs the Docker stack running. Run: bash source_code/scripts/demo/task1.sh
source "$(dirname "${BASH_SOURCE[0]}")/typer.sh"
cd "$REPO_ROOT"
clear
sleep "${START_DELAY:-1.5}"

run "cd source_code"
run "uv run python scripts/inspect_chunks.py data/handbooks/CS_MajorHandbook.pdf --grep 'minimum major GPA'" 9
run "cd .."
clear

run "docker compose -f docker/compose.yml exec api python -m rag.ingest data/handbooks --reset" 4
run "docker compose -f docker/compose.yml exec api python -m rag.ingest data/handbooks" 4
clear

run "docker compose -f docker/compose.yml exec api python -m rag.vectors --contains 'minimum major GPA' --limit 1 --query 'What is the minimum GPA for the capstone?'" 12

[[ -n $DONE_FILE ]] && touch "$DONE_FILE"
sleep 3
