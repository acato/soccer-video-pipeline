#!/bin/bash
# Download the model weights the pipeline needs at inference time.
# All artifacts are public on HuggingFace Hub under acatorcini/.
set -e

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Downloading models to: $MODELS_DIR"

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' CLI not found. Install via: pip install huggingface_hub"
  exit 1
fi

# YOLOv9 ball detector — both .pt (AGPL Ultralytics runtime) and .onnx
# (Apache 2.0 onnxruntime path) are downloaded; pick one to use.
echo "→ ball detector (Ultralytics + ONNX)"
hf download acatorcini/yolov9-soccer-ball \
  v9b_best.pt v9b_best.onnx \
  --local-dir "$MODELS_DIR"

# YOLOv8 soccer player detector — obtain from a public source (e.g.,
# github.com/uisikdag/weed_soccer_models). The pipeline references the
# file at $MODELS_DIR/yolov8_soccer_uisikdag.pt — drop it there manually
# if it isn't present.
if [ ! -f "$MODELS_DIR/yolov8_soccer_uisikdag.pt" ]; then
  echo ""
  echo "WARNING: player detector not present at"
  echo "  $MODELS_DIR/yolov8_soccer_uisikdag.pt"
  echo "Obtain it manually from a public soccer-fine-tuned YOLOv8 source."
fi

# The 32B VLM model is NOT downloaded here — vLLM pulls it on first launch
# from the HF Hub repo:
#   acatorcini/qwen3-vl-32b-soccer-v11-fp8
# That's a 34 GB download into the HF cache (~/.cache/huggingface/hub).

echo ""
echo "Done. See docs/install.md for next steps (vLLM serve commands)."
