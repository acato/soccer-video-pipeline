#!/usr/bin/env bash
# Train Qwen3-VL-32B LoRA on the v6 dataset (rebalanced for rare classes).
#
# v6 differences from v5 (see scripts/rebalance_v5_to_v6.py for details):
#   - rare classes (catch / goal / shot_stop_diving) oversampled 6-13× to
#     give them comparable gradient signal to throw_in (~1500 windows each)
#   - hard-negative "none" windows (within ±90s of shot/goal in same game)
#     kept verbatim — gives the model contrast for shot_on_target
#   - penalty class dropped from training (5 examples, hopeless)
#   - val_small built from goal-rich game_13 (instead of stratified random)
#     so eval_loss actually reflects rare-class quality
#
# Wall time: ~81h (v6 train is 12,103 examples vs v5's 8854; v5 took 59.5h).
#
# Same memory budget + flags as v5 (proven to fit on 2× RTX 3090):
#   QLoRA (NF4 + double quant) + device_map=auto + paged_adamw_8bit + ViT frozen.
#
# Pre-requisites on LLM server:
#   - vLLM stopped (training and inference share the GPUs)
#   - swift-env activated, ms-swift installed
#   - Dataset converted: scripts/convert_v5_to_swift.py against v6 with
#     --frames-prefix /mnt/transit/soccer-finetune/lora_dataset_v5/ (frames
#     live in v5; v6 records reference them by relative path)
#
# Usage:
#   # On the LLM server:
#   bash train_lora_v6.sh [smoke|full]

set -euo pipefail

MODE="${1:-smoke}"
DATA_DIR="/mnt/transit/soccer-finetune/lora_dataset_v6_swift"
OUT_DIR="/mnt/transit/soccer-finetune/checkpoints/v6-32b"

if [ "$MODE" = "smoke" ]; then
    DATA="$DATA_DIR/smoke.jsonl"
    EXTRA="--max_steps 2"
    OUT_DIR="/mnt/transit/soccer-finetune/checkpoints/v6-32b-smoke"
elif [ "$MODE" = "full" ]; then
    DATA="$DATA_DIR/train.jsonl"
    EXTRA="--num_train_epochs 1"
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

CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True swift sft \
    --device_map auto \
    --max_memory '{0: "22GiB", 1: "14GiB"}' \
    --model Qwen/Qwen3-VL-32B-Instruct \
    --dataset "$DATA" \
    --val_dataset "$DATA_DIR/val_small.jsonl" \
    --output_dir "$OUT_DIR" \
    $EXTRA \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
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
    --save_steps 200 \
    --save_total_limit 4 \
    --eval_strategy steps \
    --eval_steps 200 \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory true \
    --report_to none \
    --enable_thinking false \
    2>&1 | tee "$OUT_DIR/training.log"
