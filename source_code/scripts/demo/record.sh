#!/usr/bin/env bash
# Record a demo script in a fullscreen kitty window with wf-recorder (niri / Wayland).
#   bash source_code/scripts/demo/record.sh task1 [output.mp4] [--audio]
# Recording starts just before the script begins typing and stops right after its last output.
# Uses your own kitty config; CLEAN_LOOK=1 forces an opaque window with a large font (FONT_SIZE).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
task=${1:?usage: record.sh task1|task2|task3 [output.mp4] [--audio]}
out=${2:-$repo/demo_video/$task.mp4}
audio=(); [[ ${3:-} == --audio ]] && audio=(--audio)
done_file=$(mktemp -u "${XDG_RUNTIME_DIR:-/tmp}/demo-done.XXXXXX")
mkdir -p "$(dirname "$out")"

look=()
[[ ${CLEAN_LOOK:-0} == 1 ]] && look=(-o background_opacity=1 -o font_size="${FONT_SIZE:-22}")

kitty --class rag-demo "${look[@]}" -o confirm_os_window_close=0 -o mouse_hide_wait=0.5 \
  --title "$task demo" env START_DELAY=4 DONE_FILE="$done_file" bash "$here/$task.sh" &
kitty_pid=$!

# niri: fullscreen the demo window and refuse to record unless it really covers the screen,
# otherwise the recording would capture whatever else is open.
win_state() {  # prints "<id> <width>x<height>" of the rag-demo window, or nothing
  niri msg --json windows | python3 -c '
import json, sys
for w in json.load(sys.stdin):
    if w.get("app_id") == "rag-demo":
        ws = w["layout"]["window_size"]; print(w["id"], "%dx%d" % (ws[0], ws[1])); break'
}
abort() { echo "$1; not recording" >&2; kill "$kitty_pid" 2>/dev/null || true; exit 1; }

output_name=$(niri msg --json outputs | python3 -c '
import json, sys
print(next(iter(json.load(sys.stdin))))')
screen=$(niri msg --json outputs | python3 -c '
import json, sys
o = next(iter(json.load(sys.stdin).values()))["logical"]; print("%dx%d" % (o["width"], o["height"]))')

win_id='' size=''
for _ in $(seq 50); do
  read -r win_id size < <(win_state) || true
  [[ -n $win_id ]] && break
  sleep 0.1
done
[[ -n $win_id ]] || abort "demo window never appeared"
niri msg action focus-window --id "$win_id"
[[ $size == "$screen" ]] || niri msg action fullscreen-window --id "$win_id"
for _ in $(seq 30); do
  read -r _ size < <(win_state) || true
  [[ $size == "$screen" ]] && break
  sleep 0.1
done
[[ $size == "$screen" ]] || abort "demo window is '$size', not fullscreen $screen"
sleep 0.5

# -D (--no-damage): capture every frame. Without it wf-recorder only grabs frames when the screen
# changes, so static stretches (a long parse, a page on screen) are dropped from the video.
wf-recorder -D -o "$output_name" -r 30 -c libx264 -p preset=veryfast -p crf=23 -x yuv420p \
  "${audio[@]}" -y -f "$out" >"${XDG_RUNTIME_DIR:-/tmp}/wf-recorder-$task.log" 2>&1 &
rec_pid=$!
started=$(date +%s.%N)

while [[ ! -e $done_file ]] && kill -0 "$kitty_pid" 2>/dev/null; do sleep 0.3; done
sleep 2
kill -INT "$rec_pid"; wait "$rec_pid" || true
wall=$(awk -v a="$started" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
wait "$kitty_pid" || true
rm -f "$done_file"
echo "saved $out (recorded ${wall}s wall clock, video $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out")s)"
