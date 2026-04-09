#!/usr/bin/env bash
#
# poll_current_run.sh — Mac-side poller for the currently active detection run.
#
# Invoked every N minutes by launchd (see infra/com.soccer.runpoller.plist).
# Reads state/current_run.json from the repo checkout, queries the pipeline
# API for job status, snapshots that status to ~/soccer-runs/state/, and when
# the job completes it runs scripts/evaluate_detection.py with --json-out and
# writes the final results.
#
# Design goals:
#   - Idempotent: safe to run at any cadence; does nothing if already finished.
#   - Resilient: survives API down, missing files, eval failures without
#     corrupting state (all writes use temp-then-rename).
#   - Observable: heartbeat file + rotating log on every tick.
#   - Zero state in Claude: everything lives on the Mac filesystem.
#
set -u
set -o pipefail
# Intentionally NOT -e: we want to handle errors gracefully and keep the
# heartbeat updated even when sub-steps fail.

# ── Config ─────────────────────────────────────────────────────────────────
REPO="${SOCCER_REPO:-$HOME/Downloads/soccer-video-pipeline}"
STATE_DIR="${SOCCER_STATE_DIR:-$HOME/soccer-runs/state}"
LOGS_DIR="${SOCCER_LOGS_DIR:-$HOME/soccer-runs/logs}"

mkdir -p "$STATE_DIR" "$LOGS_DIR"

MANIFEST="$REPO/state/current_run.json"
HEARTBEAT="$STATE_DIR/poller_last_run"
POLLER_LOG="$LOGS_DIR/poller.log"

# Log rotation: keep poller.log under ~2 MB.
if [[ -f "$POLLER_LOG" ]]; then
  SIZE=$(stat -f %z "$POLLER_LOG" 2>/dev/null || echo 0)
  if [[ "$SIZE" -gt 2000000 ]]; then
    mv "$POLLER_LOG" "$POLLER_LOG.1"
  fi
fi

log() {
  local ts
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  printf '[%s] %s\n' "$ts" "$*" | tee -a "$POLLER_LOG" >&2
}

notify() {
  # Best-effort desktop notification; silently ignore if unavailable.
  local title="$1"
  local message="$2"
  osascript -e "display notification \"${message//\"/\\\"}\" with title \"${title//\"/\\\"}\"" 2>/dev/null || true
}

atomic_write() {
  # atomic_write <dest> <stdin>
  local dest="$1"
  local tmp="${dest}.tmp.$$"
  cat > "$tmp"
  mv "$tmp" "$dest"
}

# Heartbeat — always update, even if the rest of the script bails.
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$HEARTBEAT"

# ── Locate python ──────────────────────────────────────────────────────────
PY=""
for cand in "$REPO/.venv/bin/python" "/opt/homebrew/bin/python3.12" "/opt/homebrew/bin/python3" "/usr/bin/python3"; do
  if [[ -x "$cand" ]]; then
    PY="$cand"
    break
  fi
done
if [[ -z "$PY" ]]; then
  log "FATAL: no python interpreter found"
  exit 0  # exit 0 so launchd doesn't throttle/disable us
fi

# ── Load manifest ──────────────────────────────────────────────────────────
if [[ ! -f "$MANIFEST" ]]; then
  log "no manifest at $MANIFEST; nothing to track"
  exit 0
fi

# Parse once, cache values.
read -r RUN_NUMBER RUN_LABEL JOB_ID API_URL EVENTS_FILE BASELINE_RUN BASELINE_F1 < <(
  "$PY" - "$MANIFEST" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
run_number = d["run_number"]
run_label = d.get("run_label", f"run_{run_number}")
job_id = d["job_id"]
api_url = d["api_url"].rstrip("/")
events_file = d["events_file_template"].format(job_id=job_id)
baseline = d.get("baseline", {}) or {}
base_run = baseline.get("run", "?")
base_f1 = baseline.get("f1", 0.0)
print(run_number, run_label, job_id, api_url, events_file, base_run, base_f1)
PYEOF
)

STATUS_FILE="$STATE_DIR/${RUN_LABEL}_status.json"
RESULT_FILE="$STATE_DIR/${RUN_LABEL}_result.json"
EVAL_LOG="$LOGS_DIR/${RUN_LABEL}_eval.log"
EVAL_JSON="$STATE_DIR/${RUN_LABEL}_eval.json"

# Short-circuit if the run is already fully evaluated.
if [[ -f "$RESULT_FILE" ]]; then
  log "run $RUN_NUMBER already has result at $RESULT_FILE; idle"
  exit 0
fi

# ── Query the API ──────────────────────────────────────────────────────────
log "polling run=$RUN_NUMBER job=$JOB_ID api=$API_URL"

RESP_FILE=$(mktemp -t soccerpoll.XXXXXX)
trap 'rm -f "$RESP_FILE"' EXIT

HTTP_CODE=$(curl -sS -o "$RESP_FILE" -w '%{http_code}' --max-time 15 "$API_URL/jobs/$JOB_ID" 2>>"$POLLER_LOG" || echo "000")

if [[ "$HTTP_CODE" != "200" ]]; then
  log "api returned http=$HTTP_CODE — snapshot as unreachable"
  "$PY" - "$STATUS_FILE" "$RUN_NUMBER" "$JOB_ID" "$HTTP_CODE" <<'PYEOF'
import json, os, sys, tempfile, datetime
dest, run_num, job_id, http = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
snap = {
    "run_number": run_num,
    "job_id": job_id,
    "status": "unreachable",
    "http_code": http,
    "polled_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
tmp = dest + ".tmp"
open(tmp, "w").write(json.dumps(snap, indent=2))
os.replace(tmp, dest)
PYEOF
  exit 0
fi

# Parse the API response into a status snapshot.
"$PY" - "$RESP_FILE" "$STATUS_FILE" "$RUN_NUMBER" <<'PYEOF'
import json, os, sys, datetime
resp_path, dest, run_num = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = json.load(open(resp_path))
snap = {
    "run_number": run_num,
    "job_id": data.get("job_id"),
    "status": data.get("status"),
    "progress_pct": data.get("progress_pct"),
    "updated_at": data.get("updated_at"),
    "error": data.get("error"),
    "reel_types": data.get("reel_types"),
    "polled_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
tmp = dest + ".tmp"
open(tmp, "w").write(json.dumps(snap, indent=2))
os.replace(tmp, dest)
PYEOF

STATUS=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('status',''))" "$STATUS_FILE")
PROGRESS=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('progress_pct',''))" "$STATUS_FILE")
log "run=$RUN_NUMBER status=$STATUS progress=$PROGRESS"

case "$STATUS" in
  complete)
    log "run $RUN_NUMBER COMPLETE — running evaluation"
    if [[ ! -f "$EVENTS_FILE" ]]; then
      log "ERROR: events file missing at $EVENTS_FILE"
      "$PY" - "$RESULT_FILE" "$RUN_NUMBER" "$JOB_ID" "$EVENTS_FILE" <<'PYEOF'
import json, os, sys, datetime
dest, run_num, job_id, events = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
payload = {
    "run_number": run_num,
    "job_id": job_id,
    "status": "error",
    "error": "events_file_missing",
    "events_file": events,
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
tmp = dest + ".tmp"
open(tmp, "w").write(json.dumps(payload, indent=2))
os.replace(tmp, dest)
PYEOF
      notify "Soccer Pipeline" "Run $RUN_NUMBER complete but events file missing"
      exit 0
    fi

    # Build the evaluate command from the manifest so extra args (e.g.
    # --tolerance 30) flow through without code changes.
    EVAL_ARGS_JSON=$("$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps(d.get('evaluate_args', [])))" "$MANIFEST")
    EVAL_SCRIPT=$("$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('evaluate_script','scripts/evaluate_detection.py'))" "$MANIFEST")

    log "evaluating: $PY $REPO/$EVAL_SCRIPT $EVENTS_FILE [args=$EVAL_ARGS_JSON] --json-out $EVAL_JSON"

    # shellcheck disable=SC2046
    if "$PY" "$REPO/$EVAL_SCRIPT" "$EVENTS_FILE" \
        $("$PY" -c "import json,sys;print(' '.join(json.loads(sys.argv[1])))" "$EVAL_ARGS_JSON") \
        --json-out "$EVAL_JSON" \
        > "$EVAL_LOG" 2>&1; then
      log "evaluation succeeded; building result file"
      "$PY" - "$EVAL_JSON" "$RESULT_FILE" "$MANIFEST" "$EVAL_LOG" <<'PYEOF'
import json, os, sys, datetime
eval_json, dest, manifest, log_path = sys.argv[1:5]
eval_data = json.load(open(eval_json))
man = json.load(open(manifest))
overall = eval_data.get("overall", {})
payload = {
    "run_number": man["run_number"],
    "run_label": man.get("run_label", f"run_{man['run_number']}"),
    "job_id": man["job_id"],
    "commit_sha": man.get("commit_sha"),
    "status": "success",
    "completed_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "overall": overall,
    "per_type": eval_data.get("per_type", {}),
    "gt_total": eval_data.get("gt_total"),
    "detected_total": eval_data.get("detected_total"),
    "gt_counts": eval_data.get("gt_counts"),
    "detected_counts": eval_data.get("detected_counts"),
    "baseline": man.get("baseline", {}),
    "targets": man.get("targets", {}),
    "delta_f1_vs_baseline": (overall.get("f1", 0.0) - man.get("baseline", {}).get("f1", 0.0)),
    "eval_log": log_path,
}
tmp = dest + ".tmp"
open(tmp, "w").write(json.dumps(payload, indent=2))
os.replace(tmp, dest)
PYEOF
      F1=$("$PY" -c "import json,sys;print(round(json.load(open(sys.argv[1])).get('overall',{}).get('f1',0),3))" "$RESULT_FILE")
      DELTA=$("$PY" -c "import json,sys;print(round(json.load(open(sys.argv[1])).get('delta_f1_vs_baseline',0),3))" "$RESULT_FILE")
      log "RESULT: run $RUN_NUMBER F1=$F1 (delta vs baseline $BASELINE_RUN: $DELTA)"
      notify "Soccer Pipeline" "Run $RUN_NUMBER done — F1 $F1 (baseline $BASELINE_RUN=$BASELINE_F1, Δ$DELTA)"
    else
      log "ERROR: evaluation failed — see $EVAL_LOG"
      "$PY" - "$RESULT_FILE" "$RUN_NUMBER" "$JOB_ID" "$EVAL_LOG" <<'PYEOF'
import json, os, sys, datetime
dest, run_num, job_id, log_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
payload = {
    "run_number": run_num,
    "job_id": job_id,
    "status": "eval_failed",
    "eval_log": log_path,
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
tmp = dest + ".tmp"
open(tmp, "w").write(json.dumps(payload, indent=2))
os.replace(tmp, dest)
PYEOF
      notify "Soccer Pipeline" "Run $RUN_NUMBER: evaluation FAILED"
    fi
    ;;

  failed|error)
    log "job $JOB_ID reported status=$STATUS — recording failure"
    "$PY" - "$RESULT_FILE" "$RUN_NUMBER" "$JOB_ID" "$STATUS" "$STATUS_FILE" <<'PYEOF'
import json, os, sys, datetime
dest, run_num, job_id, status, status_file = sys.argv[1:6]
snap = json.load(open(status_file))
payload = {
    "run_number": int(run_num),
    "job_id": job_id,
    "status": "job_failed",
    "job_status": status,
    "error": snap.get("error"),
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
tmp = dest + ".tmp"
open(tmp, "w").write(json.dumps(payload, indent=2))
os.replace(tmp, dest)
PYEOF
    notify "Soccer Pipeline" "Run $RUN_NUMBER FAILED"
    ;;

  *)
    # ingesting/detecting/segmenting/assembling — still in flight, nothing more to do.
    ;;
esac

exit 0
