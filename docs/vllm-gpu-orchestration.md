# vLLM GPU Orchestration — Deployment & Coexistence Guide

**Target System:** LLM Server (10.10.2.222) — Ubuntu, 2x RTX 3090 24GB (no NVLink)
**Last Updated:** 2026-03-29

---

## 1. The Problem

Four AI workloads compete for two GPUs with no hardware memory isolation:

| Priority | Workload | GPU Needs | Runtime |
|----------|----------|-----------|---------|
| 1 (highest) | Soccer Pipeline (vLLM Qwen3-VL-32B) | Both GPUs, tensor-parallel | Batch, 30-90 min |
| 2 | ComfyUI + Flux/Wan (image/video gen) | GPU 0 (~17-19GB) | Interactive |
| 3 | Ollama (HA home safety) | GPU 1 (~8GB) | Always-on |

vLLM needs the full 48GB pool (Qwen3-VL-32B split across two cards). No two workloads can share a GPU — each consumes most of a card`s 24GB.

## 2. Default GPU Assignment

```
GPU 0 (bus 04, PCIe switch): ComfyUI (Docker) — Flux/Wan models
GPU 1 (bus 07, direct CPU):  Ollama (systemd)  — home safety LLM
```

Services are pinned:
- Ollama: CUDA_VISIBLE_DEVICES=1 in /etc/systemd/system/ollama.service.d/gpu.conf
- ComfyUI: device_ids: ['0'] in docker-compose.yml

## 3. vLLM Container

**File:** docker-compose.vllm.yml (alongside pipeline compose files)

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-qwen
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      --model Qwen/Qwen3-VL-32B-Instruct
      --quantization fp8
      --tensor-parallel-size 2
      --gpu-memory-utilization 0.95
      --max-model-len 31488
      --max-num-seqs 4
      --trust-remote-code
      --host 0.0.0.0
      --port 8000
    ports:
      - "8000:8000"
    volumes:
      - /home/aless/.cache/huggingface:/root/.cache/huggingface
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 80
      start_period: 300s
    restart: "no"
```

**Key settings:**
- **FP8 quantization**: ~18GB per GPU. Full BF16 would need ~64GB and exceed the pool.
- **tensor-parallel-size 2**: Splits model across both 3090s. Works without NVLink (PCIe comms slower but fine for batch).
- **gpu-memory-utilization 0.95**: Leaves ~1.2GB headroom per card; KV cache needs room for 31K tokens.
- **max-model-len 31488**: Actual running config. Pipeline sends 45s chunks at 4 FPS (180 frames, ~11K visual tokens + ~2K text). Encoder cache budget is 16,384 visual tokens.
- **max-num-seqs 4**: Safety bound — Celery processes chunks sequentially.
- **restart: "no"**: Lifecycle managed by wrapper script, not Docker.
- **HuggingFace cache**: Reuses host cache to avoid re-downloading ~18GB of weights.

## 4. GPU Dispatcher (gpu-dispatcher.sh)

**Location:** /home/aless/scripts/gpu-dispatcher.sh

### Commands

| Command | Effect |
|---------|--------|
| acquire-all | Snapshot running services, stop ComfyUI + Ollama, verify VRAM free |
| release-all | Read snapshot, restart only services that were running before |
| status | Show GPU allocation, service states, lock status |

### How It Works

1. **Acquire**: Records which services are active in a state file, stops them, waits 5s for VRAM release, verifies via nvidia-smi
2. **Lock**: Creates ~/.local/state/gpu-dispatcher.lock — prevents double acquisition
3. **Release**: Reads state file, restarts only what was running, removes lock + state
4. **Watchdog**: systemd timer (gpu-watchdog.timer) checks every 30 min — if lock is >6h old, auto-releases

The full script is in the GPU orchestration design doc at ~/Documents/gpu-orchestration-design.md on the workstation.

## 5. Pipeline Runner (soccer-pipeline-run.sh)

**Location:** /home/aless/scripts/soccer-pipeline-run.sh

Wraps the full lifecycle with guaranteed cleanup:

```
Phase 1: gpu-dispatcher.sh acquire-all          (~10s)
Phase 2: docker compose -f vllm up, poll /health  (~2-5 min)
Phase 3: docker compose up (API+Worker+Redis)    (~15s)
Phase 4: pipeline_cli.py submit, poll status    (~30-90 min)
Phase 5: EXIT trap: tear down vLLM, release-all (~15s)
```

### Usage

```bash
# Basic
~/scripts/soccer-pipeline-run.sh matches/game.mp4

# With reel types
~/scripts/soccer-pipeline-run.sh matches/game.mp4 --reel keeper,highlights

# Check GPU state anytime
~/scripts/gpu-dispatcher.sh status
```

### Key Design Choices

- **trap EXIT cleanup**: Fires on success, failure, Ctrl+C, or kill. Guarantees vLLM stops and services restore.
- **YOLO on CPU**: When vLLM holds both GPUs, YOLO falls back to CPU. Detection is not the bottleneck.
- **10-minute vLLM timeout**: Model loading is slow (downloading weights on first run, then ~2-5 min from cache).
- **3-hour job timeout**: Safety bound for a single match.

## 6. Pipeline Integration

### .env Settings for vLLM

```env
VLLM_ENABLED=true
VLLM_URL=http://localhost:8000
VLLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
GPU_COUNT=0    # Workers use CPU for YOLO when vLLM holds both GPUs
```

### Pipeline to vLLM Communication

The pipeline calls vLLM via src/detection/chunk_tagger.py using the OpenAI-compatible /v1/chat/completions endpoint at VLLM_URL. The vLLM service is treated as external infrastructure — the pipeline does not start or stop it.

### Ports During Pipeline Run

| Port | Service |
|------|---------|
| 8000 | vLLM (OpenAI-compatible API) |
| 8080 | Pipeline API (job submit/status) |
| 5555 | Flower (Celery monitoring) |
| 6379 | Redis (internal broker) |

ComfyUI (8188) and Ollama (11434) are stopped during pipeline runs.

## 7. File Layout on LLM Server

```
/home/aless/
  scripts/
    gpu-dispatcher.sh            # GPU resource manager
    soccer-pipeline-run.sh       # Pipeline orchestration wrapper
  soccer-pipeline/               # Cloned repo
    docker-compose.yml           # API + Worker + Redis + Flower
    docker-compose.gpu.yml       # GPU overrides
    docker-compose.vllm.yml      # vLLM container
    infra/.env                   # Pipeline config
  comfyui/                       # Stopped during pipeline runs
  .local/state/
    gpu-dispatcher.lock          # Present only during GPU acquisition
    gpu-dispatcher.state         # Services snapshot
  .cache/huggingface/            # Shared model cache
```

## 8. Failure Modes

| Scenario | Recovery |
|----------|----------|
| Pipeline job fails | EXIT trap fires automatically, teardown + restore |
| Ctrl+C | SIGINT caught by EXIT trap, automatic cleanup |
| kill (SIGTERM) | EXIT trap fires, automatic cleanup |
| kill -9 | No trap fires. Watchdog auto-releases after 6h. Manual: gpu-dispatcher.sh release-all |
| vLLM startup timeout | Script exits 1 then cleanup. Check docker logs vllm-qwen |
| Server reboot mid-job | Watchdog clears stale lock. Ollama auto-starts (systemd). ComfyUI auto-starts (restart policy). |
| vLLM OOM | Container crashes then health check timeout then cleanup. Reduce gpu-memory-utilization. |
| Double job submission | acquire-all sees lockfile, exits with error. One job at a time. |

## 9. Prerequisites (First Run)

1. **HuggingFace model cache**: Pre-download weights (~18GB):
   ```bash
   huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --cache-dir ~/.cache/huggingface
   ```

2. **Ollama GPU pinning**: Verify systemd override contains Environment="CUDA_VISIBLE_DEVICES=1"

3. **Sudo access** for Ollama stop/start (passwordless systemctl stop/start ollama)

4. **Watchdog timer**: sudo systemctl enable --now gpu-watchdog.timer

## 10. Evolution Roadmap

| Phase | Timeline | Change |
|-------|----------|--------|
| Phase 1 | 1-2 months | Move Pi-hole off LLM server, evaluate moving Plex to NAS/mini-PC |
| Phase 2 | 3-6 months | Buy used A100 80GB PCIe for single-card vLLM (no TP needed), keep one 3090 for ComfyUI |
| Phase 3 | 6-12 months | Dedicated mini-PC for Plex+Immich+Pi-hole, evaluate SGLang as vLLM replacement |

**Key decisions:**
- Do NOT change motherboard/platform — X570 + 5950X is not the bottleneck
- Do NOT buy RTX 5090 — only 8GB more VRAM for $2,000+
- A100 80GB is the sweet spot — solves VRAM ceiling, eliminates TP overhead
