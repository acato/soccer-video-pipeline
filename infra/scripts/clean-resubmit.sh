#!/usr/bin/env bash
# Clean up all state and redeploy for a fresh job submission.
# Usage: bash infra/scripts/clean-resubmit.sh
set -euo pipefail

echo "=== Tearing down ==="
make down 2>/dev/null || true

echo "=== Clearing state ==="
# Clear old job files
rm -f /tmp/soccer-pipeline/jobs/*.json 2>/dev/null || true
# Clear worker/api logs
> /tmp/soccer-pipeline/worker.log 2>/dev/null || true
> /tmp/soccer-pipeline/api.log 2>/dev/null || true
# Clear Python bytecode cache
find src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== Flushing Redis ==="
docker compose -f infra/docker-compose.redis.yml up -d 2>/dev/null || true
sleep 2
docker exec infra-redis-1 redis-cli FLUSHALL 2>/dev/null || true
docker compose -f infra/docker-compose.redis.yml down 2>/dev/null || true

echo "=== Deploying ==="
bash infra/scripts/setup.sh

echo "=== Ready for job submission ==="
