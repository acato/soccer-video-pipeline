#!/usr/bin/env bash
# Serve QL3 LoRA-merged + FP8-quantized model via vLLM on the LLM server.
#
# Model: Qwen3-VL-32B-Instruct merged with checkpoint-400 LoRA, then FP8_DYNAMIC
#        quantized via llmcompressor on Criscato02. ~33 GB on disk.
# Endpoint: http://10.10.2.222:8000, served as "qwen3-vl-32b" so the Mac
#           pipeline does not need .env changes.
#
# Pre-requisites:
#   - Stop swift_deploy (frees GPUs):  pkill -f "swift deploy"
#   - vLLM env activated by run-vllm.sh
set -euo pipefail
MODEL_DIR=/mnt/transit/soccer-finetune/checkpoints/v5-32b/fp8-c400

source ~/vllm-env/bin/activate
exec ~/run-vllm.sh vllm serve "$MODEL_DIR" \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000 \
  --host 10.10.2.222 \
  --dtype auto \
  --served-model-name qwen3-vl-32b \
  --quantization compressed-tensors
