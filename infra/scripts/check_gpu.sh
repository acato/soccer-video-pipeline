#!/bin/bash
# Verify NVIDIA Container Toolkit is installed and GPUs are accessible.
# Run this on the Linux host before 'make up'.
set -euo pipefail

echo "=== GPU Setup Check ==="

# 1. NVIDIA driver
if ! command -v nvidia-smi &>/dev/null; then
    echo "FAIL: nvidia-smi not found. Install NVIDIA drivers first:"
    echo "  sudo apt-get install -y nvidia-driver-535"
    exit 1
fi
echo "NVIDIA driver:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# 2. NVIDIA Container Toolkit
if ! dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
    echo "WARN: nvidia-container-toolkit not installed. Installing..."
    echo ""
    echo "Run the following commands:"
    echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo '  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list'
    echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo nvidia-ctk runtime configure --runtime=docker"
    echo "  sudo systemctl restart docker"
    exit 1
fi
echo "nvidia-container-toolkit installed"

# 3. Docker GPU access
if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "FAIL: Docker cannot access GPUs. Run:"
    echo "  sudo nvidia-ctk runtime configure --runtime=docker"
    echo "  sudo systemctl restart docker"
    exit 1
fi
echo "Docker GPU access: OK"

# 4. Count GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo ""
echo "Detected $GPU_COUNT GPU(s). Set in .env:"
echo "  GPU_COUNT=$GPU_COUNT"
echo "  USE_GPU=true"
echo ""
echo "=== GPU ready ==="
