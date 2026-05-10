#!/usr/bin/env bash
# Serve QL3 LoRA-merged + FP8-quantized model via vLLM.
# v6 c757 (Qwen3-VL-32B-Instruct + v6-32b LoRA c757, FP8_DYNAMIC, ~33 GB).
set -euo pipefail
MODEL_DIR=/mnt/transit/soccer-finetune/checkpoints/v6-32b/fp8-c757

source ~/vllm-env/bin/activate
exec ~/run-vllm.sh vllm serve "$MODEL_DIR" \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 \
  --port 8000 \
  --host 10.10.2.222 \
  --dtype auto \
  --served-model-name qwen3-vl-32b \
  --quantization compressed-tensors
