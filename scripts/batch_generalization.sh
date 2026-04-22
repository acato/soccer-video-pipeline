#!/usr/bin/env bash
# Batch generalization run.
#
# For each configured game:
#   1. Symlink the encoded .mp4 into soccer-working (if missing)
#   2. DELETE any dup job by video SHA to force a fresh run
#   3. POST /jobs with the game's team config, get new job_id
#   4. Poll /jobs/<id> every 5 min until status in {complete, failed}
#   5. Run calibrate_offsets.py to discover per-game offsets
#   6. Run evaluate_detection.py with those offsets
#   7. Write a structured summary to ~/soccer-runs/state/batch/<label>.json
#
# Sequential execution — each game takes ~3h, four games ≈ 12h total.
#
# Usage:
#   nohup bash scripts/batch_generalization.sh \
#        > ~/soccer-runs/logs/batch.log 2>&1 &
#
# Monitor:
#   tail -f ~/soccer-runs/logs/batch.log
#   ls ~/soccer-runs/state/batch/
set -uo pipefail

REPO="${SOCCER_REPO:-$HOME/Downloads/soccer-video-pipeline}"
API_URL="${SOCCER_API_URL:-http://127.0.0.1:8088}"
SOCCER_WORKING="/Users/aless/soccer-working"
BATCH_DIR="$HOME/soccer-runs/state/batch"
LOG_DIR="$HOME/soccer-runs/logs"
PY="$REPO/.venv/bin/python"
POLL_INTERVAL_SEC=300    # 5-minute poll
POLL_TIMEOUT_SEC=21600   # 6-hour cap per game (well above normal ~3h)

mkdir -p "$BATCH_DIR" "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

cd "$REPO"

# Game list: label | symlink_name | source_video | half1_gt | half2_gt |
#            team_color | gk_color | opponent_color | opponent_gk | half2_game_offset
#
# half2_game_offset: 2700 = 45-min halves (U19+), 2400 = 40-min halves (U15)
GAMES=(
  # Folder 1 — 08 GA (U19) vs Sporting AC (same team as Rush, new opponent)
  "folder_01_sporting_ac|2025-01-01-ga-vs-sporting-ac.mp4|/Volumes/transit/Games/1/1752784779079_video-3706a2d1-2dc5-4bc0-8dad-c94f0262e36c-1752785855.26862-encoded.mp4|/Volumes/transit/Games/1/08 GA (U19) vs Sporting AC_1st Half.json|/Volumes/transit/Games/1/08 GA (U19) vs Sporting AC_2nd Half.json|white|neon_yellow|blue|neon_green|2700"
  # Folder 13 — 08 GA (U19) at Capital FC (same team as Rush, away game)
  "folder_13_capital_fc|2026-at-capital-fc.mp4|/Volumes/transit/Games/13/1772040274070_07_08_GA_at_Capital_FC-47845ea7-c6f8-4118-9c56-cef1866addc8-1772040752.107228-encoded.mp4|/Volumes/transit/Games/13/08 GA (U19)_1st Half.json|/Volumes/transit/Games/13/08 GA (U19)_2nd Half.json|white|neon_yellow|red|neon_green|2700"
  # Folder 4 — Seattle Reign 2011 GA (U15) vs NPSA WW Surf (different team, U15)
  "folder_04_reign_u15_wwsurf|2026-01-18-reign-u15-vs-wwsurf.mp4|/Volumes/transit/Games/4/1768796597623_seattle-reign-academy-2011-ga-vs-ww-surf-2011-ga-11-ga-vs-ww-surf-jan-18-d74e23a0-0ae6-49de-8ebf-d05cfc700ce9-1768798011.373838-encoded.mp4|/Volumes/transit/Games/4/2026-01-18_Seattle Reign 2011 GA (U15) vs NPSA WW Surf U15 (W)_1st Half.json|/Volumes/transit/Games/4/2026-01-18_Seattle Reign 2011 GA (U15) vs NPSA WW Surf U15 (W)_2nd Half.json|blue|neon_green|purple|neon_yellow|2400"
  # Folder 10 — Seattle Reign 2011 GA (U15) vs Washington Rush U15
  "folder_10_reign_u15_rush|2026-02-07-reign-u15-vs-rush.mp4|/Volumes/transit/Games/10/1770526794598_reign-2011-vs-wa-rush-2026-02-07-fddf6486-a04f-40a7-96f6-73ecb7c9b548-1770527705.883494-encoded.mp4|/Volumes/transit/Games/10/2026-02-07_Seattle Reign 2011 GA (U15) vs Washington Rush U15 (W)_1st Half.json|/Volumes/transit/Games/10/2026-02-07_Seattle Reign 2011 GA (U15) vs Washington Rush U15 (W)_2nd Half.json|blue|neon_green|white|neon_yellow|2400"
)

process_game() {
  local spec="$1"
  IFS='|' read -r LABEL SYMLINK SRC H1 H2 OF_COLOR GK_COLOR OPP_COLOR OPP_GK H2OFFSET <<< "$spec"

  log "=== START $LABEL ==="
  local RESULT_FILE="$BATCH_DIR/${LABEL}.json"
  if [ -f "$RESULT_FILE" ]; then
    log "Already complete: $RESULT_FILE exists. Skipping."
    return 0
  fi

  # 1. Symlink video
  local VIDEO_PATH="$SOCCER_WORKING/$SYMLINK"
  if [ ! -e "$VIDEO_PATH" ]; then
    ln -sf "$SRC" "$VIDEO_PATH"
    log "Symlinked $SYMLINK -> $SRC"
  fi

  # 2. Build job payload + DELETE any dup by SHA (force fresh run)
  local PAYLOAD
  PAYLOAD=$("$PY" - <<PYEOF
import json
p = {
    "nas_path": "$SYMLINK",
    "reel_types": ["keeper", "highlights"],
    "match_config": {
        "team": {
            "team_name": "subject_team",
            "outfield_color": "$OF_COLOR",
            "gk_color": "$GK_COLOR",
        },
        "opponent": {
            "team_name": "opponent",
            "outfield_color": "$OPP_COLOR",
            "gk_color": "$OPP_GK",
        },
    },
}
print(json.dumps(p))
PYEOF
)

  # Check for existing job by SHA and delete it
  local SHA
  SHA=$("$PY" - "$VIDEO_PATH" <<'PYEOF'
import sys, hashlib
with open(sys.argv[1], "rb") as f:
    h = hashlib.sha256()
    while True:
        chunk = f.read(8 * 1024 * 1024)
        if not chunk:
            break
        h.update(chunk)
print(h.hexdigest())
PYEOF
)
  local EXISTING
  EXISTING=$(curl -s "$API_URL/jobs" | "$PY" -c "
import json, sys
sha = sys.argv[1]
jobs = json.load(sys.stdin)
for j in jobs:
    if j.get('video_file', {}).get('sha256') == sha:
        print(j['job_id'])
        break
" "$SHA" 2>/dev/null || true)
  if [ -n "$EXISTING" ]; then
    log "Deleting existing job $EXISTING for SHA $SHA"
    curl -sf -X DELETE "$API_URL/jobs/$EXISTING" > /dev/null 2>&1 || true
  fi

  # 3. Submit
  log "Submitting job: $LABEL"
  local RESP
  RESP=$(curl -sf -X POST "$API_URL/jobs" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")
  if [ -z "$RESP" ]; then
    log "FAIL: submission returned empty — API down?"
    return 1
  fi
  local JOB_ID
  JOB_ID=$(echo "$RESP" | "$PY" -c "import json,sys;print(json.load(sys.stdin)['job_id'])")
  log "Submitted: job_id=$JOB_ID"

  # 4. Poll until terminal
  local ELAPSED=0
  local STATUS
  while [ "$ELAPSED" -lt "$POLL_TIMEOUT_SEC" ]; do
    sleep "$POLL_INTERVAL_SEC"
    ELAPSED=$((ELAPSED + POLL_INTERVAL_SEC))
    STATUS=$(curl -s "$API_URL/jobs/$JOB_ID" | "$PY" -c "import json,sys;d=json.load(sys.stdin);print(d.get('status',''))")
    local PROGRESS
    PROGRESS=$(curl -s "$API_URL/jobs/$JOB_ID" | "$PY" -c "import json,sys;d=json.load(sys.stdin);print(d.get('progress',0))")
    log "  [$LABEL] status=$STATUS progress=$PROGRESS elapsed=${ELAPSED}s"
    case "$STATUS" in
      complete|failed|cancelled) break ;;
    esac
  done

  if [ "$STATUS" != "complete" ]; then
    log "FAIL: status=$STATUS after ${ELAPSED}s"
    "$PY" - "$RESULT_FILE" "$LABEL" "$JOB_ID" "$STATUS" <<'PYEOF'
import json, sys
dest, label, job_id, status = sys.argv[1:5]
with open(dest, "w") as f:
    json.dump({
        "label": label, "job_id": job_id, "status": status,
        "error": "pipeline did not complete",
    }, f, indent=2)
PYEOF
    return 1
  fi

  # 5. Calibrate offsets
  local EVENTS_FILE="/tmp/soccer-pipeline/$JOB_ID/diagnostics/dual_pass_events.jsonl"
  if [ ! -f "$EVENTS_FILE" ]; then
    log "FAIL: events file missing at $EVENTS_FILE"
    return 1
  fi
  local CALIB_JSON="/tmp/batch_${LABEL}_calib.json"
  "$PY" "$REPO/scripts/calibrate_offsets.py" \
    "$EVENTS_FILE" "$H1" "$H2" \
    --half2-game-offset "$H2OFFSET" --tolerance 45 \
    > "$CALIB_JSON"
  log "Calibrator output: $(cat $CALIB_JSON | head -20)"

  # Extract the offsets (first JSON object — the script also prints a
  # "Suggested ..." line which we drop)
  local VO H2VS
  VO=$("$PY" -c "
import json, sys
# Read until we've parsed the full JSON object (single dict at start)
with open(sys.argv[1]) as f:
    txt = f.read()
# JSON object is at the start until the first line starting with 'Suggested'
end = txt.find('\nSuggested')
if end == -1:
    end = len(txt)
d = json.loads(txt[:end])
print(d.get('video_offset', 0))
" "$CALIB_JSON")
  H2VS=$("$PY" -c "
import json, sys
with open(sys.argv[1]) as f:
    txt = f.read()
end = txt.find('\nSuggested')
if end == -1:
    end = len(txt)
d = json.loads(txt[:end])
print(d.get('half2_video_start', 0))
" "$CALIB_JSON")

  log "  Calibrated: vo=$VO half2_start=$H2VS half2_game_offset=$H2OFFSET"

  # 6. Evaluate with calibrated offsets
  local EVAL_JSON="/tmp/batch_${LABEL}_eval.json"
  "$PY" "$REPO/scripts/evaluate_detection.py" \
    --tolerance 90 \
    --video-offset "$VO" \
    --half2-start "$H2VS" \
    --half2-game-offset "$H2OFFSET" \
    --gt-file "$H1" \
    --gt-file "$H2" \
    --json-out "$EVAL_JSON" \
    "$EVENTS_FILE" > /dev/null 2>&1

  # 7. Save summary
  "$PY" - "$EVAL_JSON" "$CALIB_JSON" "$RESULT_FILE" "$LABEL" "$JOB_ID" <<'PYEOF'
import json, sys
eval_path, calib_path, dest, label, job_id = sys.argv[1:6]
with open(eval_path) as f:
    e = json.load(f)
with open(calib_path) as f:
    txt = f.read()
end = txt.find("\nSuggested")
c = json.loads(txt[:end] if end != -1 else txt)
out = {
    "label": label,
    "job_id": job_id,
    "status": "complete",
    "calibration": c,
    "overall": e.get("overall", {}),
    "per_type": {
        t: {k: m[k] for k in ("gt", "detected", "tp", "fp", "precision", "recall", "f1")}
        for t, m in e.get("per_type", {}).items() if t != "__overall__"
    },
    "gt_total": e.get("gt_total"),
    "detected_total": e.get("detected_total"),
}
with open(dest, "w") as f:
    json.dump(out, f, indent=2)
print(f"WROTE {dest}: F1={e['overall']['f1']:.3f}")
PYEOF

  log "=== DONE $LABEL F1=$("$PY" -c "import json;print(f\"{json.load(open('$EVAL_JSON'))['overall']['f1']:.3f}\")") ==="
}

log "Batch generalization run starting — ${#GAMES[@]} games"
for spec in "${GAMES[@]}"; do
  process_game "$spec" || log "ERROR processing $spec"
done
log "Batch complete. Results in $BATCH_DIR/"
