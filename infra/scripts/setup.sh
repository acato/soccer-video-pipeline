#!/usr/bin/env bash
# setup.sh — Detect hardware, generate .env, start the stack.
#
# Three deployment modes (auto-detected):
#   NVIDIA  — Full Docker stack with GPU passthrough
#   MPS     — Redis in Docker, worker + API run natively (Apple Silicon GPU)
#   CPU     — Full Docker stack, CPU-only inference
#
# Usage:
#   ./infra/scripts/setup.sh                                # interactive
#   ./infra/scripts/setup.sh /mnt/nas/soccer /mnt/nas/out   # non-interactive
set -euo pipefail

INFRA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT_DIR="$(cd "$INFRA_DIR/.." && pwd)"
ENV_FILE="$INFRA_DIR/.env"
COMPOSE_BASE="$INFRA_DIR/docker-compose.yml"
COMPOSE_GPU="$INFRA_DIR/docker-compose.gpu.yml"
COMPOSE_REDIS="$INFRA_DIR/docker-compose.redis.yml"
COMPOSE_CMD=""

# Deployment mode: nvidia | mps | cpu
MODE="cpu"
GPU_COUNT=1
USE_GPU=false

echo "=== Soccer Video Pipeline — Setup ==="
echo ""

# ── 1. Prerequisites ────────────────────────────────────────────────────────

if ! command -v docker &>/dev/null; then
    echo "FATAL: 'docker' not found. Install Docker first."
    exit 1
fi

if docker compose version &>/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "FATAL: Neither 'docker compose' nor 'docker-compose' found."
    exit 1
fi

# ── 2. NAS Paths ────────────────────────────────────────────────────────────

if [ $# -ge 2 ]; then
    NAS_MOUNT_PATH="$1"
    NAS_OUTPUT_PATH="$2"
else
    read -rp "NAS source path (read-only, where match MP4s live) [/mnt/nas/soccer]: " NAS_MOUNT_PATH
    NAS_MOUNT_PATH="${NAS_MOUNT_PATH:-/mnt/nas/soccer}"
    read -rp "NAS output path (writable, for finished reels) [${NAS_MOUNT_PATH}/output]: " NAS_OUTPUT_PATH
    NAS_OUTPUT_PATH="${NAS_OUTPUT_PATH:-${NAS_MOUNT_PATH}/output}"
fi

WORKING_DIR="${3:-/tmp/soccer-pipeline}"

echo ""
echo "Paths:"
echo "  Source:  $NAS_MOUNT_PATH"
echo "  Output:  $NAS_OUTPUT_PATH"
echo "  Working: $WORKING_DIR"

# ── 3. Hardware Detection ──────────────────────────────────────────────────

# Check NVIDIA first (Linux with CUDA GPUs)
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NVIDIA_COUNT" -gt 0 ]; then
        echo ""
        echo "NVIDIA GPU(s) detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'

        if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
            echo "  Docker GPU access: OK"
            MODE="nvidia"
            GPU_COUNT="$NVIDIA_COUNT"
            USE_GPU=true
        else
            echo ""
            echo "WARNING: NVIDIA GPUs found but Docker cannot access them."
            echo "Install the NVIDIA Container Toolkit:"
            echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \\"
            echo "    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
            echo '  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \'
            echo '    | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \'
            echo '    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list'
            echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
            echo "  sudo nvidia-ctk runtime configure --runtime=docker"
            echo "  sudo systemctl restart docker"
            echo ""
            echo "Falling back to CPU mode."
        fi
    fi
fi

# Check Apple Silicon MPS (macOS with M-series chip)
if [ "$MODE" = "cpu" ] && [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    # Verify PyTorch can see MPS
    MPS_OK=false
    if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        MPS_OK=true
    elif python3.11 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        MPS_OK=true
    elif python3.12 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
        MPS_OK=true
    fi

    if [ "$MPS_OK" = "true" ]; then
        MODE="mps"
        GPU_COUNT=1
        USE_GPU=true
        echo ""
        echo "Apple Silicon detected — MPS GPU acceleration enabled."
        echo "  Redis will run in Docker; worker + API will run natively."
    else
        echo ""
        echo "Apple Silicon detected but PyTorch MPS not available."
        echo "Install PyTorch with MPS support:"
        echo "  pip install -r requirements.txt"
        echo ""
        echo "Falling back to CPU mode (Docker)."
    fi
fi

if [ "$MODE" = "cpu" ]; then
    echo ""
    echo "No GPU detected — workers will use CPU."
fi

# ── 4. Generate .env ─────────────────────────────────────────────────────────

# Redis URL differs: Docker-internal vs localhost
if [ "$MODE" = "mps" ]; then
    REDIS_HOST="localhost"
else
    REDIS_HOST="redis"
fi

cat > "$ENV_FILE" <<ENVEOF
# Generated by setup.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Mode: $MODE
# Re-run infra/scripts/setup.sh to regenerate.

# ── Paths ────────────────────────────────────────────────────────────────────
NAS_MOUNT_PATH=$NAS_MOUNT_PATH
NAS_OUTPUT_PATH=$NAS_OUTPUT_PATH
WORKING_DIR=$WORKING_DIR

# ── Queue ────────────────────────────────────────────────────────────────────
CELERY_BROKER_URL=redis://$REDIS_HOST:6379/0
CELERY_RESULT_BACKEND=redis://$REDIS_HOST:6379/1
REDIS_URL=redis://$REDIS_HOST:6379/0

# ── GPU / Workers ────────────────────────────────────────────────────────────
GPU_COUNT=$GPU_COUNT
USE_GPU=$USE_GPU

# ── Detection ────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH=$INFRA_DIR/models/yolov8m.pt
YOLO_INFERENCE_SIZE=1280
DETECTION_FRAME_STEP=3
MIN_EVENT_CONFIDENCE=0.65
CHUNK_DURATION_SEC=30
CHUNK_OVERLAP_SEC=2.0

# ── Clip padding ─────────────────────────────────────────────────────────────
PRE_EVENT_PAD_SEC=3.0
POST_EVENT_PAD_SEC=5.0

# ── Output encoding ──────────────────────────────────────────────────────────
OUTPUT_CODEC=copy
OUTPUT_CRF=18
OUTPUT_AUDIO_CODEC=copy

# ── NAS reliability ──────────────────────────────────────────────────────────
MAX_NAS_RETRY=3
NAS_RETRY_DELAY_SEC=5.0
WATCH_POLL_INTERVAL_SEC=10.0
WATCH_STABLE_TIME_SEC=30.0
ENVEOF

echo ""
echo "Generated $ENV_FILE (mode: $MODE)"

# ── 5. Ensure directories exist ─────────────────────────────────────────────

mkdir -p "$NAS_MOUNT_PATH" 2>/dev/null || true
mkdir -p "$NAS_OUTPUT_PATH" 2>/dev/null || true
mkdir -p "$WORKING_DIR" 2>/dev/null || true
mkdir -p "$INFRA_DIR/models" 2>/dev/null || true

# ── 6. Download model weights if missing ─────────────────────────────────────

if [ ! -f "$INFRA_DIR/models/yolov8m.pt" ]; then
    echo ""
    echo "Downloading YOLOv8m weights..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$INFRA_DIR/models/yolov8m.pt" \
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$INFRA_DIR/models/yolov8m.pt" \
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
    else
        echo "WARNING: Neither wget nor curl found. Download yolov8m.pt manually to $INFRA_DIR/models/"
    fi
fi

# ── 7. Start services ───────────────────────────────────────────────────────

echo ""

get_lan_ip() {
    if [ "$(uname -s)" = "Darwin" ]; then
        ipconfig getifaddr en0 2>/dev/null || echo "localhost"
    else
        hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost"
    fi
}

if [ "$MODE" = "mps" ]; then
    # ── MPS mode: Redis in Docker, worker + API native ───────────────────
    echo "Starting Redis in Docker..."
    $COMPOSE_CMD -f "$COMPOSE_REDIS" up -d

    # Wait for Redis
    echo "Waiting for Redis..."
    for i in $(seq 1 15); do
        if docker run --rm --network host redis:7-alpine redis-cli ping 2>/dev/null | grep -q PONG; then
            break
        fi
        sleep 1
    done

    # Stop any previous native processes
    pkill -f "celery.*soccer_pipeline" 2>/dev/null || true
    pkill -f "uvicorn.*src.api.app" 2>/dev/null || true
    sleep 1

    # Find the right python
    PYTHON=""
    for py in python3.11 python3.12 python3.13 python3; do
        if command -v "$py" &>/dev/null && "$py" -c "import torch" 2>/dev/null; then
            PYTHON="$py"
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "FATAL: No Python with PyTorch found. Run: pip install -r requirements.txt"
        exit 1
    fi
    echo "Using $PYTHON ($(${PYTHON} --version 2>&1))"

    # Export env vars for native processes
    set -a
    source "$ENV_FILE"
    set +a
    export PYTHONPATH="$ROOT_DIR"

    # Start worker in background
    echo "Starting Celery worker (MPS)..."
    cd "$ROOT_DIR"
    nohup $PYTHON -m celery -A src.api.worker worker \
        --loglevel=info --concurrency=1 \
        --pidfile="$WORKING_DIR/celery_worker.pid" \
        > "$WORKING_DIR/worker.log" 2>&1 &
    WORKER_PID=$!
    echo "  Worker PID: $WORKER_PID (log: $WORKING_DIR/worker.log)"

    # Start API in background
    echo "Starting API server..."
    nohup $PYTHON -m uvicorn src.api.app:create_app --factory \
        --host 0.0.0.0 --port 8080 \
        > "$WORKING_DIR/api.log" 2>&1 &
    API_PID=$!
    echo "  API PID: $API_PID (log: $WORKING_DIR/api.log)"

    # Wait for API to be ready
    echo "Waiting for API..."
    for i in $(seq 1 20); do
        if curl -sf http://localhost:8080/health &>/dev/null; then
            break
        fi
        sleep 1
    done

    LAN_IP=$(get_lan_ip)
    echo ""
    echo "=== Stack is up (MPS mode) ==="
    echo "  API:     http://$LAN_IP:8080"
    echo "  Monitor: http://$LAN_IP:8080/ui"
    echo "  Device:  MPS (Apple Silicon GPU)"
    echo ""
    echo "  Worker log: tail -f $WORKING_DIR/worker.log"
    echo "  API log:    tail -f $WORKING_DIR/api.log"
    echo "  Stop:       make down"

else
    # ── NVIDIA or CPU mode: full Docker stack ────────────────────────────
    echo "Starting Docker stack..."

    COMPOSE_FILES="-f $COMPOSE_BASE"
    if [ "$MODE" = "nvidia" ]; then
        COMPOSE_FILES="$COMPOSE_FILES -f $COMPOSE_GPU"
        echo "  Mode: NVIDIA GPU — $GPU_COUNT worker(s)"
    else
        echo "  Mode: CPU — $GPU_COUNT worker(s)"
    fi

    $COMPOSE_CMD $COMPOSE_FILES --env-file "$ENV_FILE" up -d --build

    LAN_IP=$(get_lan_ip)
    echo ""
    echo "=== Stack is up ($MODE mode) ==="
    echo "  API:     http://$LAN_IP:8080"
    echo "  Flower:  http://$LAN_IP:5555"
    echo "  Monitor: http://$LAN_IP:8080/ui"
fi

echo ""
echo "Submit a job:"
echo "  curl -X POST http://localhost:8080/jobs \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"nas_path\": \"match.mp4\", \"reel_types\": [\"goalkeeper\", \"highlights\"]}'"
