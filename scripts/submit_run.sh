#!/usr/bin/env bash
# submit_run.sh — safely submit the run described in state/current_run.json.
#
# GUARDRAILS (the whole point of this script — see commit guardrails doc):
#   1. Read expected commit_sha from state/current_run.json.
#   2. Verify the running worker's worker_commit.txt matches it.
#      If it doesn't, REFUSE to submit — the worker is running stale code.
#   3. (Optional) auto-restart the worker when called with --restart.
#   4. After successful submission, PATCH state/current_run.json.job_id
#      with the returned job_id and update submitted_at.
#
# Run on the Mac:
#   bash scripts/submit_run.sh          # strict: fails if worker_commit mismatch
#   bash scripts/submit_run.sh --restart # restart worker first, then submit

set -euo pipefail

RESTART=0
for arg in "$@"; do
  case "$arg" in
    --restart) RESTART=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

REPO="${SOCCER_REPO:-$HOME/Downloads/soccer-video-pipeline}"
STATE_DIR="${SOCCER_STATE_DIR:-$HOME/soccer-runs/state}"
API_URL="${SOCCER_API_URL:-http://127.0.0.1:8088}"
MANIFEST="$REPO/state/current_run.json"
WORKER_COMMIT_FILE="$STATE_DIR/worker_commit.txt"

cd "$REPO"

if [ ! -f "$MANIFEST" ]; then
  echo "[submit] missing manifest: $MANIFEST" >&2
  exit 2
fi

EXPECTED_SHA="$(python3 -c "import json; print(json.load(open('$MANIFEST'))['commit_sha'])")"
JOB_PAYLOAD="$(python3 -c "
import json
m = json.load(open('$MANIFEST'))
# Build the submission payload from manifest.job_payload if present,
# otherwise keep the current behavior of failing loudly.
p = m.get('job_payload')
if not p:
    raise SystemExit('manifest.job_payload is missing — cannot submit')
print(json.dumps(p))
")"

if [ "$RESTART" = "1" ]; then
  echo "[submit] --restart: pulling and restarting worker..."
  git pull --ff-only
  bash "$REPO/scripts/restart_pipeline.sh"
fi

# GUARDRAIL: verify worker is running the expected commit
if [ ! -f "$WORKER_COMMIT_FILE" ]; then
  echo "[submit] FAIL: $WORKER_COMMIT_FILE missing — can't verify worker code." >&2
  echo "         Run: bash scripts/restart_pipeline.sh" >&2
  exit 1
fi

WORKER_SHA_SHORT="$(head -1 "$WORKER_COMMIT_FILE")"
WORKER_SHA_FULL="$(sed -n 2p "$WORKER_COMMIT_FILE")"
EXPECTED_SHORT="${EXPECTED_SHA:0:7}"
WORKER_SHORT="${WORKER_SHA_SHORT:0:7}"

# Use ancestor check rather than strict equality. The manifest commit_sha
# points to the commit that introduced the CODE changes, but the worker is
# typically running a later commit (HEAD at restart time) that includes
# the manifest update itself. As long as the expected commit is reachable
# from the worker commit, the code is live in the worker's memory.
if [ "$WORKER_SHORT" = "$EXPECTED_SHORT" ]; then
  echo "[submit] worker commit OK ($WORKER_SHORT == $EXPECTED_SHORT)"
elif git merge-base --is-ancestor "$EXPECTED_SHA" "${WORKER_SHA_FULL:-$WORKER_SHA_SHORT}" 2>/dev/null; then
  echo "[submit] worker commit OK ($WORKER_SHORT is descendant of $EXPECTED_SHORT)"
else
  cat >&2 <<EOF
[submit] FAIL: worker commit does not include manifest commit
  expected (manifest): $EXPECTED_SHORT ($EXPECTED_SHA)
  running  (worker):   $WORKER_SHORT ($WORKER_SHA_FULL)

The worker's loaded code does not include the commit the manifest says
we want to test. Either the worker was restarted from a stale checkout
or someone force-pushed over the branch.

Fix with:
  bash scripts/restart_pipeline.sh
  bash scripts/submit_run.sh
Or in one shot:
  bash scripts/submit_run.sh --restart
EOF
  exit 1
fi

# Check the repo's current HEAD matches too (we're on the committed manifest)
REPO_SHA="$(git rev-parse --short HEAD)"
if [ "$REPO_SHA" != "$EXPECTED_SHORT" ]; then
  echo "[submit] WARN: repo HEAD ($REPO_SHA) != manifest commit_sha ($EXPECTED_SHORT)" >&2
  echo "         Did you forget to bump commit_sha in state/current_run.json?" >&2
fi

echo "[submit] submitting job to $API_URL/jobs..."
RESPONSE="$(curl -sf -X POST "$API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d "$JOB_PAYLOAD")" || {
    echo "[submit] FAIL: API submission returned non-2xx" >&2
    exit 1
  }

JOB_ID="$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")"
STATUS="$(echo  "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "[submit] submitted: job_id=$JOB_ID status=$STATUS"

# Patch manifest with the new job_id and submitted_at
python3 - <<PY
import json, pathlib
p = pathlib.Path("$MANIFEST")
d = json.loads(p.read_text())
d["job_id"] = "$JOB_ID"
d["submitted_at"] = "$TS"
tmp = p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(d, indent=2) + "\n")
tmp.replace(p)
print("[submit] manifest patched:", d["job_id"])
PY

echo "[submit] next: commit + push state/current_run.json to track the run"
