#!/usr/bin/env bash
# Task 3 demo: Docker deployment from scratch, then the app in the browser.
# Off-screen prep: docker/nvidia checks pass and `sudo systemctl stop ollama` (host Ollama off).
# Run: bash source_code/scripts/demo/record.sh task3 demo_video/task3-docker-deployment.mp4
source "$(dirname "${BASH_SOURCE[0]}")/typer.sh"
cd "$REPO_ROOT"
clear
sleep "${START_DELAY:-1.5}"

run "docker compose -f docker/compose.yml down" 2
run "docker compose -f docker/compose.yml up -d --build" 4

show_browser http://localhost:3000 8 --last
