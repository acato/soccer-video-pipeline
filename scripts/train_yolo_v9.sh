#!/bin/bash
# Fine-tune YOLOv8 ball detector on manually annotated new-venue frames.
# Base: yolov8_soccer_uisikdag (already soccer-tuned, 4-class). We override
# heads to a single class (ball).
#
# Run on llm:
#   bash ~/soccer-video-pipeline/scripts/train_yolo_v9.sh
#
# Outputs: ~/yolo_v9_runs/v9_<run>/weights/best.pt  →  rsync'd to NAS at end

set -euo pipefail

DATA="/mnt/transit/soccer-finetune/yolo_ball_v9/data.yaml"
BASE="/home/aless/yolov8_soccer_uisikdag.pt"
PROJECT="/home/aless/yolo_v9_runs"
NAME="v9"
EPOCHS="${EPOCHS:-120}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-8}"          # batch=8 @ 1280 fits one 3090 (~18GB)
DEVICE="${DEVICE:-0}"        # single GPU; the other can serve vLLM if needed
PATIENCE="${PATIENCE:-30}"
SEED="${SEED:-42}"

mkdir -p "$PROJECT"

echo "===================================================================="
echo "  v9 YOLO ball detector training"
echo "  data:    $DATA"
echo "  base:    $BASE"
echo "  imgsz:   $IMGSZ   batch: $BATCH   epochs: $EPOCHS   device: $DEVICE"
echo "===================================================================="

~/quant-env/bin/yolo train \
  model="$BASE" \
  data="$DATA" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  device="$DEVICE" \
  project="$PROJECT" \
  name="$NAME" \
  patience="$PATIENCE" \
  seed="$SEED" \
  optimizer="AdamW" \
  lr0=0.001 \
  cos_lr=True \
  warmup_epochs=3 \
  mosaic=0.5 \
  mixup=0.0 \
  hsv_h=0.010 \
  hsv_s=0.40 \
  hsv_v=0.30 \
  degrees=2.0 \
  translate=0.05 \
  scale=0.30 \
  fliplr=0.5 \
  copy_paste=0.0 \
  exist_ok=True

# Find the best.pt (ultralytics may suffix the run dir if name conflicts)
BEST=$(ls -t "$PROJECT"/v9*/weights/best.pt 2>/dev/null | head -1)
if [ -z "$BEST" ]; then
  echo "ERROR: best.pt not found under $PROJECT" >&2
  exit 1
fi

echo
echo "best weights: $BEST"
DEST_DIR="/mnt/transit/soccer-finetune/yolo_ball_v9/weights"
mkdir -p "$DEST_DIR"
cp -v "$BEST" "$DEST_DIR/v9_best.pt"
LAST="$(dirname "$BEST")/last.pt"
[ -f "$LAST" ] && cp -v "$LAST" "$DEST_DIR/v9_last.pt"

# Validation summary
echo
echo "=== val summary ==="
~/quant-env/bin/yolo val \
  model="$BEST" \
  data="$DATA" \
  imgsz="$IMGSZ" \
  device="$DEVICE" \
  project="$PROJECT" \
  name="v9_val" \
  exist_ok=True 2>&1 | tail -25

echo
echo "DONE. v9 weights at $DEST_DIR/v9_best.pt"
