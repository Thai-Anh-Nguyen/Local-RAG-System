# Auto-typing helpers for demo recordings. Source this from a task script.
#   say "text"   types a shell comment (narration on screen)
#   run "cmd"    types the command at a fake prompt, then runs it in this shell (so `cd` sticks)
#   pause N      waits N seconds (scaled by PAUSE_SCALE)
# Env: TYPE_DELAY (seconds per character, default 0.035; 0 = instant), PAUSE_SCALE (default 1; 0 = no pauses)

TYPE_DELAY=${TYPE_DELAY:-0.035}
PAUSE_SCALE=${PAUSE_SCALE:-1}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

_prompt() {
  local dir=${PWD#"$REPO_ROOT"}
  printf '\033[1;36m%s\033[0m \033[1;32m❯\033[0m ' "RAG project${dir}"
}

_type() {
  local text=$1 i
  for ((i = 0; i < ${#text}; i++)); do
    printf '%s' "${text:i:1}"
    if [[ $TYPE_DELAY != 0 ]]; then
      # small jitter so it looks typed, slightly longer after spaces
      if [[ ${text:i:1} == " " ]]; then sleep "$(awk -v d="$TYPE_DELAY" 'BEGIN{srand(); print d*1.8}')"
      else sleep "$(awk -v d="$TYPE_DELAY" -v r="$RANDOM" 'BEGIN{print d*(0.6+(r%80)/100)}')"; fi
    fi
  done
}

pause() { sleep "$(awk -v s="$1" -v k="$PAUSE_SCALE" 'BEGIN{print s*k}')"; }

say() {
  _prompt
  printf '\033[2m'
  _type "# $1"
  printf '\033[0m\n'
  pause 0.8
}

run() {
  _prompt
  _type "$1"
  pause 0.6
  printf '\n'
  eval "$1"
  pause "${2:-3}"
}

# Open URL in a clean Chrome kiosk window (throwaway profile: no tabs, bookmarks or account),
# fullscreen it with niri, keep it on screen for N seconds, then close it.
#   show_browser URL [SECONDS] [--last]
# --last: signal the recorder to stop while the page is still on screen (closing Chrome makes niri
# focus a neighbouring window, which must never end up in the video).
show_browser() {
  local url=$1 secs=${2:-8} last=${3:-} profile before new_id i
  profile=$(mktemp -d "${XDG_RUNTIME_DIR:-/tmp}/rag-demo-chrome.XXXXXX")
  before=$(niri msg --json windows | python3 -c 'import json,sys; print(" ".join(str(w["id"]) for w in json.load(sys.stdin)))')
  google-chrome --user-data-dir="$profile" --no-first-run --no-default-browser-check \
    --disable-sync --disable-extensions --kiosk "$url" >/dev/null 2>&1 &
  local chrome_pid=$!
  for i in $(seq 100); do
    new_id=$(niri msg --json windows | BEFORE="$before" python3 -c '
import json, os, sys
old = set(os.environ["BEFORE"].split())
print(next((str(w["id"]) for w in json.load(sys.stdin) if str(w["id"]) not in old), ""))')
    [[ -n $new_id ]] && break
    sleep 0.1
  done
  if [[ -n $new_id ]]; then
    niri msg action focus-window --id "$new_id"
    sleep 1  # --kiosk asks for fullscreen itself; only toggle if it did not get it
    niri msg --json windows | ID="$new_id" python3 -c '
import json, os, sys
w = next(w for w in json.load(sys.stdin) if str(w["id"]) == os.environ["ID"])
sys.exit(0 if w["layout"]["window_size"][0] >= 1900 else 1)' ||
      niri msg action fullscreen-window --id "$new_id"
  fi
  sleep "$secs"
  if [[ $last == --last && -n ${DONE_FILE:-} ]]; then
    touch "$DONE_FILE"
    sleep 4  # recorder stops ~2 s after DONE_FILE; keep the page up until it has
  fi
  kill "$chrome_pid" 2>/dev/null; wait "$chrome_pid" 2>/dev/null
  local demo_id
  demo_id=$(niri msg --json windows | python3 -c '
import json, sys
print(next((w["id"] for w in json.load(sys.stdin) if w.get("app_id") == "rag-demo"), ""))')
  [[ -n $demo_id ]] && niri msg action focus-window --id "$demo_id"
  (sleep 5; rm -rf "$profile") >/dev/null 2>&1 &
}
