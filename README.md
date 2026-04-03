# Soccer Video Processing Pipeline

> **First time? Start here: [START_HERE.md](START_HERE.md)** — a step-by-step guide that gets you from zero to your first goalkeeper reel.

> **Intended use:** This software is designed for youth and amateur soccer analytics and personal experimentation. It is not affiliated with, endorsed by, or approved by any professional league, federation, or governing body.

Agent-based system that processes soccer match recordings from a NAS and automatically produces:

- **Goalkeeper Reel** — every GK action (saves, distribution, one-on-ones)
- **Highlights Reel** — shots, goals, near-misses, great plays
- **Player Reel** *(Phase 2)* — on-demand reel for any outfield player

## Prerequisites

This pipeline combines local video processing with AI-powered event classification. Below is what you need to provide — everything else is included in the repo or auto-installed on first run.

### Required

| Resource | What | Notes |
|----------|------|-------|
| **Docker Desktop** | Runs Redis, and optionally the worker + API | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| **FFmpeg** | Frame extraction and reel assembly | Auto-installed in Docker mode; on Mac, `setup.sh` installs via Homebrew if missing |
| **Video files** | MP4 recordings of soccer matches | Sideline camera at ~50m, 1080p or 4K. Phone/GoPro/camcorder all work |
| **~30 GB scratch space** | Per concurrent job, for frame extraction and temporary files | Configurable via `WORKING_DIR` (default: `/tmp/soccer-pipeline`) |

### Vision-Language Model (VLM) — Recommended

The pipeline uses motion detection and audio analysis to find *candidate* moments, then a VLM classifies them (shot, save, goal, corner, etc.). **Without a VLM, you get unclassified motion candidates only** — the reel will include many false positives.

You need **one** of the following VLM backends:

| Option | Cost | Hardware | Accuracy | Setup |
|--------|------|----------|----------|-------|
| **Anthropic API** (Claude) | ~$2–5/match | None (cloud) | High | Set `ANTHROPIC_API_KEY` in `.env` |
| **Self-hosted vLLM** | Free (after hardware) | 2× RTX 3090 or equivalent (~48 GB VRAM) | Highest | Run vLLM with a vision model, point `VLLM_URL` at it |
| **Ollama** | Free | 1× GPU with ≥16 GB VRAM | Medium | Run Ollama, set `VLLM_URL=http://localhost:11434/v1` |
| **Any OpenAI-compatible API** | Varies | Varies | Varies | Set `VLLM_URL` to the endpoint |
| **None** | Free | None | Low | Set `VLLM_ENABLED=false` and `VLM_ENABLED=false` |

**Which should I pick?**

- **Easiest**: Anthropic API — sign up at [console.anthropic.com](https://console.anthropic.com), get an API key, paste it in `.env`. No GPU needed. A full match costs a few dollars.
- **Best accuracy, free**: Self-hosted vLLM with `Qwen/Qwen3-VL-32B-Instruct-FP8` on 2× RTX 3090 (tensor-parallel). This is what the pipeline was tuned on.
- **Budget GPU**: Ollama with a smaller vision model (e.g., `llava` or `minicpm-v`) on a single GPU. Lower accuracy but functional.

The VLM backend is configured entirely through environment variables — no code changes needed:

```bash
# Option A: Anthropic API (easiest)
VLM_ENABLED=true
ANTHROPIC_API_KEY=sk-ant-...
VLLM_ENABLED=false

# Option B: Self-hosted vLLM or any OpenAI-compatible endpoint
VLLM_ENABLED=true
VLLM_URL=http://your-server:8000      # default: http://localhost:8000
VLLM_MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8
VLM_ENABLED=false                      # disable Claude fallback (optional)

# Option C: Ollama (OpenAI-compatible mode)
VLLM_ENABLED=true
VLLM_URL=http://localhost:11434/v1
VLLM_MODEL=llava
VLM_ENABLED=false

# Option D: No VLM (motion + audio only)
VLLM_ENABLED=false
VLM_ENABLED=false
```

When both vLLM and Anthropic are configured, the pipeline tries vLLM first and falls back to Claude if it's unavailable.

### Hardware Tiers

The pipeline auto-detects your hardware on `make deploy`:

| Tier | Detection | What runs where | Expected speed |
|------|-----------|-----------------|----------------|
| **NVIDIA GPU** | `nvidia-smi` + Docker toolkit | Everything in Docker with GPU passthrough | ~60 min / match |
| **Apple Silicon** (M1–M4) | MPS detection | Redis in Docker, worker + API native with MPS acceleration | ~60 min / match |
| **CPU only** | Fallback | Everything in Docker, CPU inference | ~120 min / match |

GPU acceleration affects YOLO object detection (Phase 1). The VLM runs on its own backend — your local GPU is not used for VLM inference unless you host it locally.

### Storage Layout

```
Source directory (read-only)     Output directory (writable)
├── game1.mp4                    ├── {job-id}/
├── game2.mp4                    │   ├── goalkeeper_reel.mp4
└── ...                          │   └── highlights_reel.mp4
                                 └── ...
```

These can be local directories, NAS mounts (NFS/SMB), or any mounted filesystem. Set them via:

```bash
NAS_MOUNT_PATH=/path/to/your/videos      # required
NAS_OUTPUT_PATH=/path/to/your/output      # required
```

## Quick Start

```bash
# 1. Deploy (auto-detects NVIDIA GPU / Apple MPS / CPU, generates .env, starts stack)
make deploy
# or non-interactive: infra/scripts/setup.sh /mnt/nas/soccer /mnt/nas/output

# 2. Submit a match for processing
python infra/scripts/pipeline_cli.py submit matches/game_2025_01_15.mp4

# 3. Monitor progress
python infra/scripts/pipeline_cli.py status <job_id> --watch
open http://localhost:8080/ui   # Web dashboard

# 4. List all jobs
python infra/scripts/pipeline_cli.py list
```

### Deployment Modes

`make deploy` auto-detects hardware and picks the right mode:

| Mode | Hardware | Stack |
|------|----------|-------|
| **NVIDIA** | Linux + CUDA GPUs | Full Docker with GPU passthrough |
| **MPS** | Apple Silicon (M1/M2/M3/M4) | Redis in Docker, worker + API run natively |
| **CPU** | Fallback | Full Docker, CPU-only inference |

## Agent System

This project uses Claude Code's multi-agent architecture. Each subdirectory under `agents/` is an independent agent with its own `CLAUDE.md` context:

| Agent | `cd` command | Purpose |
|-------|-------------|---------|
| Architect | `cd agents/architect` | System design, ADRs, interface contracts |
| Developer | `cd agents/developer` | All source code in `src/` |
| SDET | `cd agents/sdet` | Tests, CI, quality gates |
| Video Analyst | `cd agents/video-analyst` | CV/ML models, event taxonomy |
| Pipeline Operator | `cd agents/pipeline-operator` | Infra, deployment, monitoring |

## Development

```bash
make setup              # Install Python deps
make deploy             # Auto-detect hardware, generate .env, start stack
make generate-fixtures  # Create synthetic test videos
make test-unit          # Fast tests, no infra needed
make test-integration   # Requires Docker
make check-nas          # Verify NAS connectivity
make check-gpu          # Verify NVIDIA GPU + container toolkit
```

## Architecture

```
NAS (read-only)          Processing Node              NAS (output)
├── matches/             ┌────────────────────┐       ├── output/
│   ├── game1.mp4  ──►  │  Watcher → Intake  │         │   ├── {job_id}/
│   └── game2.mp4        │  ↓                 │         │   │   ├── goalkeeper_reel.mp4
│                         │  Detection         │  ──►   │   │   └── highlights_reel.mp4
│                         │  (YOLO + ByteTrack)│         │   └── ...
│                         │  ↓                 │
│                         │  Segmentation      │
│                         │  ↓                 │
│                         │  Assembly (FFmpeg) │
│                         └────────────────────┘
```

## Key Design Decisions

- **Streaming-first**: Video never fully loaded into RAM
- **Idempotent**: Re-processing same file = identical output
- **GPU-optional**: NVIDIA CUDA, Apple MPS, or CPU — auto-detected at deploy time
- **NAS-safe**: Source files never modified

See `docs/architecture.md` (generated by Architect agent) for full design.

## Disclaimer

This software is intended for youth and amateur soccer analytics and personal experimentation. It is not affiliated with, endorsed by, or approved by any professional league, federation, or governing body.

This project does not grant rights to process or redistribute copyrighted video content. Users are responsible for compliance with applicable rights and league policies.
