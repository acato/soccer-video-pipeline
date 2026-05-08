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
exec ~/run-vllm.sh vllm serve "$MODEL_DIR" \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --port 8000 \
  --host 10.10.2.222 \
  --dtype auto \
  --served-model-name qwen3-vl-32b \
  --quantization compressed-tensors
