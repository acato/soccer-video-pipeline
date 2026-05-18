#!/usr/bin/env bash
# Run kickoff detector on game_22, game_21, rush sequentially.
# Waits for any in-flight game_20 scan to complete first.
set -e
PY=~/Downloads/soccer-video-pipeline/.venv/bin/python
DET=~/Downloads/soccer-video-pipeline/scripts/detect_kickoffs.py

# Wait for any in-flight detect_kickoffs run to finish (excluding ourselves)
SELF=$$
while pgrep -f "detect_kickoffs.py" | grep -v "^${SELF}$" > /dev/null 2>&1; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] previous scans done; starting queue"

scan() {
  local label=$1 video=$2 endsec=$3
  echo "[$(date +%H:%M:%S)] $label: scanning $endsec sec..."
  $PY $DET --video "$video" --start 0 --end "$endsec" --interval 5 \
    --out "/tmp/kickoff_${label}.jsonl" \
    --per-frame-out "/tmp/kickoff_${label}_frames.jsonl" \
    > "/tmp/kickoff_${label}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $label: done. $(wc -l < /tmp/kickoff_${label}.jsonl) goals detected."
}

scan game_22 "/Users/aless/soccer-working/2026-04-26 Spokane Shadow - Reign GA11.mp4" 7228
scan game_21 "/Users/aless/soccer-working/2026-04-25 Eastern WA Surf - Reign GA11.mp4" 7220
scan rush    "/Users/aless/soccer-working/2026-02-07 - Rush - GA2008.mp4"             6845

echo "[$(date +%H:%M:%S)] all scans done"
