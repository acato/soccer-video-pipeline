#!/usr/bin/env bash
# Swap vLLM model on the LLM server.
#
# Called by DualPassDetector / ModelManager with these env vars:
#   SWAP_TARGET_NAME  — model name to serve (--served-model-name)
#   SWAP_TARGET_PATH  — HuggingFace/local model path (--model)
#   SWAP_TARGET_LORA  — LoRA adapter path (optional)
#   SWAP_TARGET_TIER  — "tier1" or "tier2"
#   SWAP_VLLM_URL     — vLLM base URL (e.g. http://10.10.2.222:8000)
#
# Prerequisites:
#   - SSH access to the LLM server (Host 'llm' in ~/.ssh/config)
#   - ~/run-vllm.sh on the LLM server (stops Ollama, starts vLLM)
#   - ~/vllm-env/ virtualenv with vLLM installed
#
# The script:
#   1. Checks if target model is already loaded (skip swap)
#   2. Stops the current vLLM process on the LLM server
#   3. Kills zombie GPU workers
#   4. Starts a new vLLM process with tier-specific GPU config
#   5. Waits for the health endpoint to respond
set -euo pipefail

: "${SWAP_TARGET_NAME:?SWAP_TARGET_NAME not set}"
: "${SWAP_TARGET_PATH:?SWAP_TARGET_PATH not set}"
: "${SWAP_VLLM_URL:?SWAP_VLLM_URL not set}"

# Extract host from URL (e.g. http://10.10.2.222:8000 → 10.10.2.222)
VLLM_HOST=$(echo "$SWAP_VLLM_URL" | sed -E 's|https?://([^:]+):?[0-9]*/?|\1|')
VLLM_PORT=$(echo "$SWAP_VLLM_URL" | grep -oE ':[0-9]+' | tr -d ':')
VLLM_PORT="${VLLM_PORT:-8000}"

echo "[swap] Target: name=${SWAP_TARGET_NAME} path=${SWAP_TARGET_PATH} tier=${SWAP_TARGET_TIER:-unknown}"
echo "[swap] Server: ${VLLM_HOST}:${VLLM_PORT}"

# ── Step 0: Check if target model is already loaded ────────────────────
CURRENT=$(curl -sf "${SWAP_VLLM_URL}/v1/models" 2>/dev/null || echo "OFFLINE")
if echo "$CURRENT" | grep -q "\"id\":\"${SWAP_TARGET_NAME}\""; then
    echo "[swap] Model ${SWAP_TARGET_NAME} is already loaded. No swap needed."
    exit 0
fi

# ── Step 1: Stop current vLLM ───────────────────────────────────────────
echo "[swap] Stopping current vLLM..."
ssh llm "pkill -f 'vllm.entrypoints' 2>/dev/null || true" 2>/dev/null || true
sleep 3

# Kill zombie GPU workers
ssh llm "pkill -9 -f 'VLLM::Worker' 2>/dev/null || true" 2>/dev/null || true
sleep 2

# Verify GPUs are free
GPU_MEM=$(ssh llm "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1}END{print s}'" || echo "?")
echo "[swap] GPU memory after cleanup: ${GPU_MEM}MB"

# ── Step 2: Build vLLM command (tier-specific GPU config) ──────────────
VLLM_ARGS="--model ${SWAP_TARGET_PATH}"
VLLM_ARGS+=" --served-model-name ${SWAP_TARGET_NAME}"
VLLM_ARGS+=" --port ${VLLM_PORT}"
VLLM_ARGS+=" --dtype auto"
VLLM_ARGS+=" --trust-remote-code"
VLLM_ARGS+=" --enforce-eager"

if [ "${SWAP_TARGET_TIER:-}" = "tier1" ]; then
    # 8B model — single GPU, larger batch size
    VLLM_ARGS+=" --max-model-len 8192"
    VLLM_ARGS+=" --max-num-seqs 8"
    VLLM_ARGS+=" --gpu-memory-utilization 0.90"
    VLLM_ARGS+=" --dtype bfloat16"
    if [ -n "${SWAP_TARGET_LORA:-}" ]; then
        VLLM_ARGS+=" --enable-lora"
        VLLM_ARGS+=" --lora-modules ${SWAP_TARGET_NAME}=${SWAP_TARGET_LORA}"
        VLLM_ARGS+=" --max-lora-rank 8"
    fi
elif [ "${SWAP_TARGET_TIER:-}" = "tier2" ]; then
    # 32B FP8 — both GPUs with tensor parallelism
    VLLM_ARGS+=" --max-model-len 16384"
    VLLM_ARGS+=" --max-num-seqs 4"
    VLLM_ARGS+=" --gpu-memory-utilization 0.92"
    VLLM_ARGS+=" --tensor-parallel-size 2"
    VLLM_ARGS+=" --kv-cache-dtype fp8"
else
    # Fallback: reasonable defaults
    VLLM_ARGS+=" --max-model-len 8192"
    VLLM_ARGS+=" --gpu-memory-utilization 0.90"
fi

# ── Step 3: Start new vLLM via run-vllm.sh wrapper ─────────────────────
echo "[swap] Starting vLLM: ${SWAP_TARGET_NAME}"
ssh llm "nohup bash -c 'source ~/vllm-env/bin/activate && ~/run-vllm.sh python3 -m vllm.entrypoints.openai.api_server ${VLLM_ARGS}' > /tmp/vllm-swap.log 2>&1 &"

# ── Step 4: Wait for health endpoint ────────────────────────────────────
echo "[swap] Waiting for vLLM health..."
MAX_WAIT=180
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))

    if curl -sf "${SWAP_VLLM_URL}/v1/models" > /dev/null 2>&1; then
        # Verify correct model is loaded
        LOADED=$(curl -sf "${SWAP_VLLM_URL}/v1/models" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data.get('data', []):
    print(m['id'])
" 2>/dev/null || echo "unknown")
        echo "[swap] vLLM ready after ${ELAPSED}s. Loaded: ${LOADED}"
        exit 0
    fi

    if [ $((ELAPSED % 15)) -eq 0 ]; then
        echo "[swap] Still waiting... (${ELAPSED}s)"
    fi
done

echo "[swap] ERROR: vLLM did not start within ${MAX_WAIT}s"
echo "[swap] Last log:"
ssh llm "tail -20 /tmp/vllm-swap.log" 2>/dev/null || true
exit 1
