#!/bin/bash
# Download ML model weights
# Run once before first deployment
MODELS_DIR="$(dirname "$0")"
set -e

echo "Downloading YOLOv8m base weights..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" 2>/dev/null || \
  curl -L "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt" -o "$MODELS_DIR/yolov8m.pt"

echo "Models downloaded to $MODELS_DIR"
echo ""
echo "NOTE: For best soccer detection results, fine-tune on SoccerNet dataset."
echo "See docs/contracts/model_registry.md for fine-tuning instructions."
