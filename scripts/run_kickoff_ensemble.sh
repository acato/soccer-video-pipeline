#!/usr/bin/env bash
# Kickoff-detection ensemble orchestrator.
#
# For each game (video + per-frame YOLO data + GT file):
#   1. Generate kickoff_pattern candidates via the detect_kickoffs offline iterator
#   2. Generate formation_cluster candidates via generate_formation_candidates.py
#   3. Stop current vLLM, start base Qwen3-VL-32B-Instruct-FP8
#   4. Verify formation candidates with base (catches kickoff_restart on new venues)
#   5. Stop base, restart v11 (catches celebration_cut on rush-style cameras)
#   6. Verify pattern candidates with v11
#   7. Combine + dedup + game-bounds filter (combined_pipeline.py)
#
# Outputs per-game verified goal list at /tmp/kickoff_${game}_verified.jsonl
#
# Usage:
#   bash scripts/run_kickoff_ensemble.sh game_22 game_21 rush ...
#
# Prerequisites: per_frame data already cached at /tmp/kickoff_${game}_frames.jsonl
# (run scripts/detect_kickoffs.py first to generate)

set -euo pipefail

PY=~/Downloads/soccer-video-pipeline/.venv/bin/python
SCRIPTS=~/Downloads/soccer-video-pipeline/scripts

# Path to the v11 LoRA-merged FP8 (default production endpoint)
V11_PATH=/mnt/transit/soccer-finetune/checkpoints/v11-32b/fp8-c150
V11_NAME=qwen3-vl-32b
V11_QUANT=compressed-tensors

# Path to the base Qwen3-VL-32B-Instruct-FP8 (HF cache, no LoRA)
BASE_PATH=/home/aless/.cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct-FP8/snapshots/4bf2c2f39c37c0fede78bede4056e1f18cdf8109
BASE_NAME=qwen3-vl-32b-base
BASE_QUANT=fp8

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
  local max_wait=${2:-1500}  # default 25 min for NFS load
  local elapsed=0
  echo "waiting for vLLM endpoint to expose '$expected_name'..."
  until curl -s -m 3 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" 2>/dev/null \
        | grep -q "\"id\":\"${expected_name}\""; do
    sleep 10
    elapsed=$((elapsed + 10))
    if [[ $elapsed -ge $max_wait ]]; then
      echo "ERROR: $expected_name did not come up in ${max_wait}s"
      return 1
    fi
  done
  echo "  ready after ${elapsed}s"
}

stop_vllm() {
  echo "stopping current vLLM..."
  ssh "$LLM_HOST" "pkill -f vllm-env/bin/vllm; sleep 6" || true
}

start_model() {
  local path=$1 name=$2 quant=$3
  echo "starting model: $name from $path"
  ssh "$LLM_HOST" "nohup /home/aless/run-vllm.sh ${VLLM_BIN} serve ${path} \
    --tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 --port ${VLLM_PORT} --host ${VLLM_HOST} --dtype auto \
    --served-model-name ${name} --quantization ${quant} > /tmp/vllm_${name}.log 2>&1 &"
  wait_endpoint "$name"
}

verify_with_model() {
  # $1 = game label, $2 = source candidate file label (pattern|formation)
  #                                  ${game}_${source}.jsonl produces ${game}_${source}_verified.jsonl
  local game=$1 src=$2
  local video=${VIDEO_PATHS[$game]}
  local in_file="/tmp/kickoff_${game}_${src}.jsonl"
  local out_file="/tmp/kickoff_${game}_${src}_verified.jsonl"
  if [[ ! -f $(ssh mac "test -f $in_file && echo $in_file") ]]; then
    # File check via ssh; simpler approach: just try and let the script fail
    :
  fi
  echo "  verifying $game/$src..."
  ssh mac "cd ~/Downloads/soccer-video-pipeline && ${PY} scripts/verify_kickoffs_vlm_v3.py \
    --video \"$video\" --in $in_file --out $out_file --workdir /tmp/kickoff_vlm_orchestrated \
    2>&1 | tail -2"
}

GAMES=("$@")
if [[ ${#GAMES[@]} -eq 0 ]]; then
  echo "Usage: $0 game1 [game2 ...]"; exit 2
fi

# Step 1: candidate generation (no VLM)
echo "=== Step 1: generate candidates ==="
for g in "${GAMES[@]}"; do
  ssh mac "cd ~/Downloads/soccer-video-pipeline && ${PY} scripts/detect_kickoffs_offline.py \
    --per-frame /tmp/kickoff_${g}_frames.jsonl --out /tmp/kickoff_${g}_pattern.jsonl 2>&1 | tail -2"
  ssh mac "cd ~/Downloads/soccer-video-pipeline && ${PY} scripts/generate_formation_candidates.py \
    --per-frame /tmp/kickoff_${g}_frames.jsonl --out /tmp/kickoff_${g}_formation.jsonl 2>&1 | tail -2"
done

# Step 2: verify FORMATION candidates with base FP8
echo "=== Step 2: swap to base FP8 ==="
stop_vllm
start_model "$BASE_PATH" "$BASE_NAME" "$BASE_QUANT"

# Patch MODEL constant temporarily
ssh mac "sed -i 's/MODEL = \"qwen3-vl-32b\"/MODEL = \"qwen3-vl-32b-base\"/' \
  ~/Downloads/soccer-video-pipeline/scripts/verify_kickoffs_vlm_v3.py"

for g in "${GAMES[@]}"; do
  verify_with_model "$g" "formation"
done

# Step 3: verify PATTERN candidates with v11 (restore the LoRA model)
echo "=== Step 3: restore v11 ==="
stop_vllm
start_model "$V11_PATH" "$V11_NAME" "$V11_QUANT"

# Patch back
ssh mac "sed -i 's/MODEL = \"qwen3-vl-32b-base\"/MODEL = \"qwen3-vl-32b\"/' \
  ~/Downloads/soccer-video-pipeline/scripts/verify_kickoffs_vlm_v3.py"

for g in "${GAMES[@]}"; do
  verify_with_model "$g" "pattern"
done

# Step 4: combine + score
echo "=== Step 4: combined scoring ==="
ssh mac "cd ~/Downloads/soccer-video-pipeline && ${PY} scripts/combined_pipeline.py"

echo "DONE. Per-game verified outputs at /tmp/kickoff_<game>_{formation,pattern}_verified.jsonl"
