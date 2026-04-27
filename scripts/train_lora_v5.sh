#!/usr/bin/env bash
# Train Qwen3-VL-32B LoRA on the v5 dataset.
#
# Runs on the LLM server (10.10.2.222) which has 2x RTX 3090 + NVLink.
# Tensor parallelism via DeepSpeed ZeRO-3 splits the bf16 base across both
# GPUs, leaving room for LoRA adapters + activations within each 24 GB.
#
# Memory budget per GPU (estimated):
#   Base bf16 sharded: 32GB / 2 = 16 GB
#   LoRA (rank 64, LLM only, ViT frozen): ~150 MB
#   Optimizer state (Adam, LoRA only): ~600 MB
#   Activations w/ gradient_checkpointing: ~3-5 GB
#   PyTorch/CUDA overhead: ~1-2 GB
#   ──────────────────────────────────────
#   Per-GPU peak: ~21-23 GB (within 24 GB)
#
# Pre-requisites on LLM server:
#   - vLLM stopped (training and inference share the GPUs)
#   - swift-env activated, ms-swift installed
#   - Dataset converted (run convert_v5_to_swift.py on Mac first)
#
# Usage:
#   # On the LLM server:
#   bash train_lora_v5.sh [smoke|full]
#     smoke = 2 steps for memory-fit test
#     full  = real training run

set -euo pipefail

MODE="${1:-smoke}"
DATA_DIR="/mnt/transit/soccer-finetune/lora_dataset_v5_swift"
OUT_DIR="/mnt/transit/soccer-finetune/checkpoints/v5-32b"

if [ "$MODE" = "smoke" ]; then
    DATA="$DATA_DIR/smoke.jsonl"
    EXTRA="--max_steps 2"
    OUT_DIR="/mnt/transit/soccer-finetune/checkpoints/v5-32b-smoke"
elif [ "$MODE" = "full" ]; then
    DATA="$DATA_DIR/train.jsonl"
    EXTRA="--num_train_epochs 2"
else
    echo "usage: $0 [smoke|full]" >&2
    exit 2
fi

# Verify GPUs are free (vLLM stopped)
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE_MB" -lt 20000 ]; then
    echo "ERROR: GPU 0 only has ${FREE_MB}MB free. Stop vLLM first:"
    echo "  pkill -f 'vllm.entrypoints'"
    exit 1
fi

mkdir -p "$OUT_DIR"

source ~/swift-env/bin/activate

# DeepSpeed ZeRO-3 across 2 GPUs
deepspeed --num_gpus=2 \
    --module swift.cli.sft \
    --model Qwen/Qwen3-VL-32B-Instruct \
    --dataset "$DATA" \
    --val_dataset "$DATA_DIR/val.jsonl" \
    --output_dir "$OUT_DIR" \
    $EXTRA \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --max_pixels 921600 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --freeze_vit true \
    --gradient_checkpointing true \
    --torch_dtype bfloat16 \
    --deepspeed default-zero3 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps 200 \
    --logging_steps 5 \
    --report_to none \
    --enable_thinking false \
    2>&1 | tee "$OUT_DIR/training.log"
