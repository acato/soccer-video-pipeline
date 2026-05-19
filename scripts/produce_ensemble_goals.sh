#!/usr/bin/env bash
# Produce kickoff-ensemble GOAL detections for a list of game videos.
#
# Wrapper around the ensemble pipeline that:
#   1. Generates pattern + formation candidates from cached YOLO frames
#   2. Verifies formation candidates with base FP8 (one model swap)
#   3. (Optional) Verifies pattern candidates with v11 LoRA — currently disabled
#      because vLLM 0.21.0 breaks v11; re-enable once 0.19.1 is re-pinned.
#   4. Writes per-game goal lists at /tmp/kickoff_<game>_ensemble_goals.jsonl
#      ready to be merged into dual_pass_events.jsonl via
#      scripts/merge_ensemble_into_events.py.
#
# Usage:
#   bash scripts/produce_ensemble_goals.sh game_22 game_21 rush game_20
#
# Prerequisites:
#   - /tmp/kickoff_<game>_frames.jsonl per game (run detect_kickoffs.py first)
#   - mac SSH access to llm host
#   - /tmp/restore_v11.sh exists OR pass --no-restore to skip v11 restoration

set -euo pipefail

PY=~/Downloads/soccer-video-pipeline/.venv/bin/python
SCRIPTS=~/Downloads/soccer-video-pipeline/scripts

BASE_PATH=/home/aless/.cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct-FP8/snapshots/4bf2c2f39c37c0fede78bede4056e1f18cdf8109
BASE_NAME=qwen3-vl-32b-base
VLLM_BIN=/home/aless/vllm-env/bin/vllm
VLLM_HOST=10.10.2.222
VLLM_PORT=8000
LLM_HOST=llm

declare -A VIDEO_PATHS=(
  [game_22]="/Users/aless/soccer-working/2026-04-26 Spokane Shadow - Reign GA11.mp4"
  [game_21]="/Users/aless/soccer-working/2026-04-25 Eastern WA Surf - Reign GA11.mp4"
  [rush]="/Users/aless/soccer-working/2026-02-07 - Rush - GA2008.mp4"
  [game_20]="/Users/aless/soccer-working/2026-04-18 Celtic - Reign GA 11.mp4"
)

wait_endpoint() {
  local expected_name=$1
  local max_wait=${2:-300}
  local elapsed=0
  echo "  waiting for vLLM endpoint to expose '$expected_name'..."
  until curl -s -m 3 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" 2>/dev/null \
        | grep -q "\"id\":\"${expected_name}\""; do
    sleep 10
    elapsed=$((elapsed + 10))
    if [[ $elapsed -ge $max_wait ]]; then
      echo "  ERROR: $expected_name did not come up in ${max_wait}s"; return 1
    fi
  done
  echo "  ready after ${elapsed}s"
}

GAMES=("$@")
if [[ ${#GAMES[@]} -eq 0 ]]; then
  echo "Usage: $0 game1 [game2 ...]"; exit 2
fi

# Step 1: candidate generation
echo "=== generating candidates ==="
for g in "${GAMES[@]}"; do
  echo "  $g"
  ssh mac "cd ~/Downloads/soccer-video-pipeline && ${PY} scripts/generate_formation_candidates.py \
    --per-frame /tmp/kickoff_${g}_frames.jsonl \
    --out /tmp/kickoff_${g}_formation.jsonl 2>&1 | tail -2"
  ssh mac "cd ~/Downloads/soccer-video-pipeline && ${PY} scripts/filter_late_candidates.py \
    /tmp/kickoff_${g}_formation.jsonl /tmp/kickoff_${g}_formation_safe.jsonl ${g} 2>&1"
done

# Step 2: swap to base FP8 (one swap, all games)
echo "=== swapping to base FP8 ==="
ssh "$LLM_HOST" "pkill -f vllm-env/bin/vllm; sleep 6" || true
ssh "$LLM_HOST" "nohup /home/aless/run-vllm.sh ${VLLM_BIN} serve ${BASE_PATH} \
  --tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 --port ${VLLM_PORT} --host ${VLLM_HOST} --dtype auto \
  --served-model-name ${BASE_NAME} --quantization fp8 > /tmp/vllm_base.log 2>&1 &"
wait_endpoint "$BASE_NAME"

# Patch MODEL constant
ssh mac "sed -i.bak 's/MODEL = \"qwen3-vl-32b\"/MODEL = \"qwen3-vl-32b-base\"/' \
  ~/Downloads/soccer-video-pipeline/scripts/verify_kickoffs_vlm_v3.py"

# Step 3: verify formation candidates per game
echo "=== verifying formation candidates with base FP8 ==="
for g in "${GAMES[@]}"; do
  echo "  $g"
  ssh mac "cd ~/Downloads/soccer-video-pipeline && ${PY} scripts/verify_kickoffs_vlm_v3.py \
    --video \"${VIDEO_PATHS[$g]}\" \
    --in /tmp/kickoff_${g}_formation_safe.jsonl \
    --out /tmp/kickoff_${g}_ensemble_goals.jsonl \
    --workdir /tmp/kickoff_vlm_ensemble 2>&1 | tail -3"
done

# Restore MODEL constant
ssh mac "sed -i.bak 's/MODEL = \"qwen3-vl-32b-base\"/MODEL = \"qwen3-vl-32b\"/' \
  ~/Downloads/soccer-video-pipeline/scripts/verify_kickoffs_vlm_v3.py"

echo
echo "DONE. Per-game ensemble goals at /tmp/kickoff_<game>_ensemble_goals.jsonl"
echo "NOTE: vLLM is still serving the base FP8 model. To restore v11, run:"
echo "  bash /tmp/restore_v11.sh"
echo "  (or launch with full vllm path — see /tmp/restore_v11.sh comments)"
echo
echo "To merge into a dual_pass_events.jsonl for eval:"
echo "  ${PY} ${SCRIPTS}/merge_ensemble_into_events.py \\"
echo "      --dual-pass /tmp/soccer-pipeline/<job_id>/diagnostics/dual_pass_events.jsonl \\"
echo "      --ensemble /tmp/kickoff_<game>_ensemble_goals.jsonl \\"
echo "      --out      /tmp/soccer-pipeline/<job_id>/diagnostics/dual_pass_events_augmented.jsonl"
