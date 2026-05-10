#!/usr/bin/env bash
# Serve v8 (ViT unfrozen) FP8 model via vLLM 0.19.1.
set -euo pipefail
CHECKPOINT_TAG="${1:-c300}"
MODEL_DIR=/mnt/transit/soccer-finetune/checkpoints/v8-32b/fp8-${CHECKPOINT_TAG}

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: model dir does not exist: $MODEL_DIR" >&2
  exit 1
fi

source ~/vllm-env/bin/activate

# HARD PIN: 0.20.0+ silently breaks Qwen3-VL-32B-FP8 (model emits EOS on every
# request → empty completions → pipeline returns 0 events). Lost hours to
# this bug three times. Refuse to launch if vllm is not exactly 0.19.1.
VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null)
if [ "$VLLM_VER" != "0.19.1" ]; then
  echo "ERROR: vllm version is '$VLLM_VER' — must be exactly 0.19.1." >&2
  echo "Fix: ~/vllm-env/bin/pip install vllm==0.19.1" >&2
  exit 1
fi

exec ~/run-vllm.sh vllm serve "$MODEL_DIR" \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --port 8000 \
  --host 10.10.2.222 \
  --dtype auto \
  --served-model-name qwen3-vl-32b \
  --quantization compressed-tensors
