#!/usr/bin/env bash
# Train v11 — trajectory+acceleration-aware continue-train from v8 c300.
#
# v10 c50/c150/c200 disambiguation (Runs 85/89-92/93-96) showed coordinate
# trajectory features plateaued at 1/16 new-venue goal recall; eval_loss
# curve didn't track F1. v11 adds an acceleration-profile line to the
# [ball-track] block — sharp deceleration distinguishes goal/save from
# miss/pass even when the ball is too small to resolve visually.
#
# v11 differences vs v10:
#   - --dataset lora_dataset_v11_swift  (trajectory + acceleration in prompts)
#   - everything else identical to v10 (same base, LR, ViT unfrozen, LoRA cfg)
#
# Wall estimate: ~24h (300 steps × ~280s/step including longer prompts).
#
# Run on llm AFTER vLLM is stopped:
#   pkill -f 'vllm.*fp8' && sleep 5
#   bash train_lora_v11.sh full
set -euo pipefail

MODE="${1:-smoke}"
DATA_DIR="/mnt/transit/soccer-finetune/lora_dataset_v11_swift"
BASE_MODEL="/mnt/transit/soccer-finetune/checkpoints/v8-32b/merged-bf16-c300"
OUT_DIR="/mnt/transit/soccer-finetune/checkpoints/v11-32b"

if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: base model dir does not exist: $BASE_MODEL" >&2
    echo "Run scripts/merge_lora_v8.py first to produce the bf16 merge of v8 c300." >&2
    exit 1
fi

if [ "$MODE" = "smoke" ]; then
    DATA="$DATA_DIR/smoke.jsonl"
    EXTRA="--max_steps 2"
    OUT_DIR="${OUT_DIR}-smoke"
elif [ "$MODE" = "full" ]; then
    DATA="$DATA_DIR/train.jsonl"
    EXTRA="--max_steps 300"
else
    echo "usage: $0 [smoke|full]" >&2
    exit 2
fi

if [ "$MODE" = "smoke" ] && [ ! -f "$DATA" ]; then
    head -100 "$DATA_DIR/train.jsonl" > "$DATA"
fi

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
    --save_steps 50 \
    --save_total_limit 8 \
    --eval_strategy steps \
    --eval_steps 50 \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory true \
    --report_to none \
    --enable_thinking false \
    2>&1 | tee "$OUT_DIR/training.log"
