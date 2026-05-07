#!/usr/bin/env bash
# Train v8 — Qwen3-VL-32B with ViT UNFROZEN to fix the new-camera vision bottleneck.
#
# Run 67-70 (v7 c100 multi-game eval) confirmed: continue-training with frozen
# ViT moved nothing on the new-venue camera. v7 detected ZERO goals across 16
# new-camera GT goals. The visual encoder cannot perceive the ball at 3-5 px;
# only ViT updates can change that.
#
# v8 differences vs v7:
#   - --freeze_vit false  (THE pivot — visual encoder trains)
#   - LR 3e-5 → 1e-5  (ViT pretraining is sensitive; lower keeps useful features)
#   - warmup_ratio 0.05 → 0.10  (longer warmup, stability)
#   - max_steps 250 → 500  (visual adaptation is slower than language)
#   - max_memory bumped (more grad memory needed for ViT)
#
# Base: merged-bf16-c757 (same as v7 — v7 didn't move language F1, so no benefit
# starting from v7's adapter). Keeps the experiment clean: only ViT-unfreeze
# variable changes.
#
# Wall estimate: ~50h (250→500 steps + ViT backward pass slows step ~30-50%)
#
# Usage:
#   bash train_lora_v8.sh [smoke|full]
set -euo pipefail

MODE="${1:-smoke}"
DATA_DIR="/mnt/transit/soccer-finetune/lora_dataset_v7_swift"
BASE_MODEL="/mnt/transit/soccer-finetune/checkpoints/v6-32b/merged-bf16-c757"
OUT_DIR="/mnt/transit/soccer-finetune/checkpoints/v8-32b"

if [ "$MODE" = "smoke" ]; then
    DATA="$DATA_DIR/smoke.jsonl"
    EXTRA="--max_steps 2"
    OUT_DIR="/mnt/transit/soccer-finetune/checkpoints/v8-32b-smoke"
elif [ "$MODE" = "full" ]; then
    DATA="$DATA_DIR/train.jsonl"
    EXTRA="--max_steps 500"
else
    echo "usage: $0 [smoke|full]" >&2
    exit 2
fi

# Smoke jsonl is 100 examples
if [ "$MODE" = "smoke" ] && [ ! -f "$DATA" ]; then
    head -100 "$DATA_DIR/train.jsonl" > "$DATA"
fi

# Verify GPUs free (vLLM stopped)
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE_MB" -lt 20000 ]; then
    echo "ERROR: GPU 0 only has ${FREE_MB}MB free. Stop vLLM:" >&2
    echo "  pkill -f 'vllm.*fp8'" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

source ~/swift-env/bin/activate

CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True swift sft \
    --device_map auto \
    --max_memory '{0: "23GiB", 1: "23GiB"}' \
    --model "$BASE_MODEL" \
    --model_type qwen3_vl \
    --dataset "$DATA" \
    --val_dataset "$DATA_DIR/val_small.jsonl" \
    --output_dir "$OUT_DIR" \
    $EXTRA \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --optim paged_adamw_8bit \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.10 \
    --max_length 8192 \
    --max_pixels 518400 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --freeze_vit false \
    --gradient_checkpointing true \
    --use_logits_to_keep true \
    --torch_dtype bfloat16 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 6 \
    --eval_strategy steps \
    --eval_steps 100 \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory true \
    --report_to none \
    --enable_thinking false \
    2>&1 | tee "$OUT_DIR/training.log"
