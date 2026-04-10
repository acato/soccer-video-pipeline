#!/usr/bin/env bash
# restart_pipeline.sh — idempotent restart of the Celery worker + uvicorn API
# on the Mac with a fresh Python import cache.
#
# GUARDRAIL: writes the git HEAD sha of the running repo to
# ~/soccer-runs/state/worker_commit.txt so downstream tools can verify
# the worker actually loaded the expected code.
#
# Run on the Mac:
#   bash scripts/restart_pipeline.sh
#
# Exit codes:
#   0  workers restarted and healthy, commit recorded
#   1  health check failed
#   2  not in repo / no .venv

set -euo pipefail

REPO="${SOCCER_REPO:-$HOME/Downloads/soccer-video-pipeline}"
STATE_DIR="${SOCCER_STATE_DIR:-$HOME/soccer-runs/state}"
API_PORT="${SOCCER_API_PORT:-8088}"
UVICORN_LOG="${SOCCER_UVICORN_LOG:-/tmp/uvicorn.log}"
CELERY_LOG="${SOCCER_CELERY_LOG:-/tmp/celery.log}"

cd "$REPO" || { echo "missing repo: $REPO" >&2; exit 2; }
[ -d .venv ] || { echo "missing .venv in $REPO" >&2; exit 2; }

mkdir -p "$STATE_DIR"

echo "[restart] killing existing workers..."
# Match both the celery parent and uvicorn on our port
pkill -f "celery -A src.api.worker" 2>/dev/null || true
pkill -f "uvicorn src.api.app"     2>/dev/null || true
sleep 2
# Force any stragglers
pkill -9 -f "celery -A src.api.worker" 2>/dev/null || true
pkill -9 -f "uvicorn src.api.app"      2>/dev/null || true
sleep 1

echo "[restart] loading env..."
set -a; source infra/.env; set +a
export PYTHONPATH="$(pwd)"
# Ensure homebrew binaries (ffmpeg/ffprobe) are on PATH. A bare 'nohup'
# from an ssh shell inherits a minimal PATH that does NOT include
# /opt/homebrew/bin, and the API fails any job with
#   "Failed to extract metadata: [Errno 2] No such file or directory: 'ffprobe'"
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$PATH"

echo "[restart] starting uvicorn on :$API_PORT..."
nohup .venv/bin/uvicorn src.api.app:app \
  --host 0.0.0.0 --port "$API_PORT" \
  > "$UVICORN_LOG" 2>&1 &
UVICORN_PID=$!
disown "$UVICORN_PID" 2>/dev/null || true

echo "[restart] starting celery worker..."
nohup .venv/bin/celery -A src.api.worker.app worker --loglevel=info \
  > "$CELERY_LOG" 2>&1 &
CELERY_PID=$!
disown "$CELERY_PID" 2>/dev/null || true

# Wait for API health (up to 30s)
echo "[restart] waiting for API health..."
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    echo "[restart] API healthy after ${i}s"
    break
  fi
  sleep 1
  if [ "$i" -eq 30 ]; then
    echo "[restart] FAIL: API did not become healthy" >&2
    exit 1
  fi
done

# Wait for celery ready line (up to 20s)
echo "[restart] waiting for celery ready..."
for i in $(seq 1 20); do
  if grep -q "celery@.*ready" "$CELERY_LOG" 2>/dev/null; then
    echo "[restart] celery ready after ${i}s"
    break
  fi
  sleep 1
  if [ "$i" -eq 20 ]; then
    echo "[restart] WARN: celery ready line not seen in ${CELERY_LOG}" >&2
  fi
done

# GUARDRAIL: record the commit the worker loaded
COMMIT_SHA="$(git rev-parse --short HEAD)"
COMMIT_FULL="$(git rev-parse HEAD)"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TMP_FILE="$STATE_DIR/worker_commit.txt.tmp"
cat > "$TMP_FILE" <<EOF
$COMMIT_SHA
$COMMIT_FULL
$TS
uvicorn_pid=$UVICORN_PID
celery_pid=$CELERY_PID
EOF
mv "$TMP_FILE" "$STATE_DIR/worker_commit.txt"

echo "[restart] worker_commit.txt -> $COMMIT_SHA @ $TS"
echo "[restart] uvicorn_pid=$UVICORN_PID  celery_pid=$CELERY_PID"
echo "[restart] done."
