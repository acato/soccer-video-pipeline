#!/usr/bin/env bash
# Train Qwen3-VL-32B LoRA v7 on the new-camera-augmented v7 dataset.
#
# v7 = "grow v6" continue-training. Base model is v6 c757 LoRA pre-merged
# into the bf16 base (merged-bf16-c757), so this is effectively a fresh LoRA
# on top of v6's accumulated learning. Avoids LoRA-on-LoRA optimizer
# state issues that --resume_from_checkpoint would hit.
#
# Dataset: lora_dataset_v7_swift = v6 12k examples + Games 20/21/22 raw,
# rebalanced again. 11,954 train / 3,366 val.
#
# Goal: adapt v6's good language-side learning (event types, JSON schema,
# outcome rules, prompt obedience) to the new high+wide camera distribution
# that broke Run 66 (F1 0.279 on Game 20).
#
# Hyperparameters vs v6:
#   - base = merged-bf16-c757 (incorporates v6 LoRA into base weights)
#   - LR 1e-4 → 3e-5 (lower; we're refining, not learning from scratch)
#   - max_steps 250 (~0.33 epochs) instead of 1 full epoch
#   - everything else identical to v6
#
# Wall time estimate: 250 steps × ~5 min/step (similar throughput to v6) ≈ 21h
#
# Usage:
#   bash train_lora_v7.sh [smoke|full]
set -euo pipefail

MODE="${1:-smoke}"
DATA_DIR="/mnt/transit/soccer-finetune/lora_dataset_v7_swift"
BASE_MODEL="/mnt/transit/soccer-finetune/checkpoints/v6-32b/merged-bf16-c757"
OUT_DIR="/mnt/transit/soccer-finetune/checkpoints/v7-32b"

if [ "$MODE" = "smoke" ]; then
    DATA="$DATA_DIR/smoke.jsonl"
    EXTRA="--max_steps 2"
    OUT_DIR="/mnt/transit/soccer-finetune/checkpoints/v7-32b-smoke"
elif [ "$MODE" = "full" ]; then
    DATA="$DATA_DIR/train.jsonl"
    EXTRA="--max_steps 250"
else
    echo "usage: $0 [smoke|full]" >&2
    exit 2
fi

# Ensure smoke jsonl exists when training in smoke mode (rebalance only writes
# train.jsonl + val.jsonl + val_small.jsonl)
if [ "$MODE" = "smoke" ] && [ ! -f "$DATA" ]; then
    head -100 "$DATA_DIR/train.jsonl" > "$DATA"
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

CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True swift sft \
    --device_map auto \
    --max_memory '{0: "22GiB", 1: "14GiB"}' \
    --model "$BASE_MODEL" \
    --model_type qwen3_vl \
    --dataset "$DATA" \
    --val_dataset "$DATA_DIR/val_small.jsonl" \
    --output_dir "$OUT_DIR" \
    $EXTRA \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --optim paged_adamw_8bit \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 8192 \
    --max_pixels 518400 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --freeze_vit true \
    --gradient_checkpointing true \
    --use_logits_to_keep true \
    --torch_dtype bfloat16 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 6 \
    --eval_strategy steps \
    --eval_steps 50 \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory true \
    --report_to none \
    --enable_thinking false \
    2>&1 | tee "$OUT_DIR/training.log"
