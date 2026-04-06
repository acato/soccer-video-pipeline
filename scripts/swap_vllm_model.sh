#!/usr/bin/env bash
# Swap vLLM model on the LLM server.
#
# Called by ModelManager with these env vars:
#   SWAP_TARGET_NAME  — model name to serve (--served-model-name)
#   SWAP_TARGET_PATH  — HuggingFace/local model path (--model)
#   SWAP_TARGET_LORA  — LoRA adapter path (optional)
#   SWAP_TARGET_TIER  — "tier1" or "tier2"
#   SWAP_VLLM_URL     — vLLM base URL (e.g. http://10.10.2.222:8000)
#
# Prerequisites:
#   - SSH access to the LLM server (Host 'llm' in ~/.ssh/config)
#   - ~/run-vllm.sh on the LLM server (stops Ollama, starts vLLM)
#
# The script:
#   1. Stops the current vLLM process on the LLM server
#   2. Starts a new vLLM process with the target model
#   3. Waits for the health endpoint to respond
set -euo pipefail

: "${SWAP_TARGET_NAME:?SWAP_TARGET_NAME not set}"
: "${SWAP_TARGET_PATH:?SWAP_TARGET_PATH not set}"
: "${SWAP_VLLM_URL:?SWAP_VLLM_URL not set}"

# Extract host from URL (e.g. http://10.10.2.222:8000 → 10.10.2.222)
VLLM_HOST=$(echo "$SWAP_VLLM_URL" | sed -E 's|https?://([^:]+):?[0-9]*/?|\1|')
VLLM_PORT=$(echo "$SWAP_VLLM_URL" | grep -oE ':[0-9]+' | tr -d ':')
VLLM_PORT="${VLLM_PORT:-8000}"

echo "[swap] Target: name=${SWAP_TARGET_NAME} path=${SWAP_TARGET_PATH} lora=${SWAP_TARGET_LORA:-none}"
echo "[swap] Server: ${VLLM_HOST}:${VLLM_PORT}"

# ── Step 1: Stop current vLLM ───────────────────────────────────────────
echo "[swap] Stopping current vLLM..."
ssh llm "pkill -f 'vllm.entrypoints' 2>/dev/null || true; sleep 2" 2>/dev/null || true

# ── Step 2: Build vLLM command ──────────────────────────────────────────
VLLM_CMD="python -m vllm.entrypoints.openai.api_server"
VLLM_CMD+=" --model ${SWAP_TARGET_PATH}"
VLLM_CMD+=" --served-model-name ${SWAP_TARGET_NAME}"
VLLM_CMD+=" --host ${VLLM_HOST}"
VLLM_CMD+=" --port ${VLLM_PORT}"
VLLM_CMD+=" --dtype auto"
VLLM_CMD+=" --trust-remote-code"

# Tier-specific settings
if [ "${SWAP_TARGET_TIER:-}" = "tier1" ]; then
    # 8B model — smaller context is fine
    VLLM_CMD+=" --max-model-len 16384"
    if [ -n "${SWAP_TARGET_LORA:-}" ]; then
        VLLM_CMD+=" --enable-lora"
        VLLM_CMD+=" --lora-modules ${SWAP_TARGET_NAME}=${SWAP_TARGET_LORA}"
    fi
elif [ "${SWAP_TARGET_TIER:-}" = "tier2" ]; then
    # 32B FP8 — needs more context
    VLLM_CMD+=" --max-model-len 16384"
fi

# ── Step 3: Start new vLLM in background ────────────────────────────────
echo "[swap] Starting vLLM: ${VLLM_CMD}"
ssh llm "nohup ${VLLM_CMD} > /tmp/vllm-swap.log 2>&1 &"

# ── Step 4: Wait for health endpoint ────────────────────────────────────
echo "[swap] Waiting for vLLM health..."
MAX_WAIT=120
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf "${SWAP_VLLM_URL}/v1/models" > /dev/null 2>&1; then
        echo "[swap] vLLM ready after ${ELAPSED}s"
        # Verify correct model is loaded
        LOADED=$(curl -sf "${SWAP_VLLM_URL}/v1/models" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data.get('data', []):
    print(m['id'])
" 2>/dev/null || echo "unknown")
        echo "[swap] Loaded model(s): ${LOADED}"
        exit 0
    fi
    sleep 3
    ELAPSED=$((ELAPSED + 3))
    if [ $((ELAPSED % 15)) -eq 0 ]; then
        echo "[swap] Still waiting... (${ELAPSED}s)"
    fi
done

echo "[swap] ERROR: vLLM did not start within ${MAX_WAIT}s"
echo "[swap] Last log:"
ssh llm "tail -20 /tmp/vllm-swap.log" 2>/dev/null || true
exit 1
