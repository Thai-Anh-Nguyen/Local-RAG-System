#!/usr/bin/env bash
# Task 2 demo: the handbook chatbot answering from the terminal (correct, incomplete, ambiguous,
# not-in-handbook). Needs the Docker stack running. Run: bash source_code/scripts/demo/task2.sh
source "$(dirname "${BASH_SOURCE[0]}")/typer.sh"
cd "$REPO_ROOT/source_code"
clear
sleep "${START_DELAY:-1.5}"

run "uv run python scripts/ask.py \"What are the prerequisites for CS311?\"" 6
clear
run "uv run python scripts/ask.py \"Can I graduate with honors if I don't do the capstone?\"" 8
clear
run "uv run python scripts/ask.py \"What courses should I take?\"" 12
clear
run "uv run python scripts/ask.py \"How much is tuition for the CS program?\"" 8

[[ -n $DONE_FILE ]] && touch "$DONE_FILE"
sleep 3
