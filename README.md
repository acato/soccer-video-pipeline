# Soccer Video Processing Pipeline

> **First time? Start here: [START_HERE.md](START_HERE.md)** — a step-by-step guide that gets you from zero to your first goalkeeper reel.

> **Intended use:** This software is designed for youth and amateur soccer analytics and personal experimentation. It is not affiliated with, endorsed by, or approved by any professional league, federation, or governing body.

Automatic event detection system that processes soccer match recordings and produces:

- **Goalkeeper Reel** — saves, goal kicks, distribution, corners, one-on-ones
- **Highlights Reel** — shots, goals, near-misses

Drop in a video, set your team colors, and get a reel back. No manual tagging.

## How It Works

The detection pipeline has three phases:

1. **Motion + Audio** — dense frame-differencing finds activity spikes; audio analysis detects whistles and crowd surges. Together they produce ~300 candidate moments per match.
2. **VLM Classification** — a vision-language model (Qwen3-VL, Claude, or any OpenAI-compatible endpoint) classifies each candidate: shot, save, goal, corner, goal kick, throw-in, etc.
3. **Structural Inference** — post-VLM phases use restart patterns to refine classifications. Shot followed by corner = parry save. Shot followed by quick goal kick = miss. Shot with no restart = catch. Kickoff after shot = goal.

After detection, clips are cut with event-specific padding, merged when overlapping, and assembled into reels via FFmpeg.

## Prerequisites

Everything below is what you need to provide. The rest is included in the repo or auto-installed on first run.

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
| **Anthropic API** (Claude) | ~$2-5/match | None (cloud) | High | Set `ANTHROPIC_API_KEY` in `.env` |
| **Self-hosted vLLM** | Free (after hardware) | 2x RTX 3090 or equivalent (~48 GB VRAM) | Highest | Run vLLM with a vision model, point `VLLM_URL` at it |
| **Ollama** | Free | 1x GPU with >=16 GB VRAM | Medium | Run Ollama, set `VLLM_URL=http://localhost:11434/v1` |
| **Any OpenAI-compatible API** | Varies | Varies | Varies | Set `VLLM_URL` to the endpoint |
| **None** | Free | None | Low | Set `VLLM_ENABLED=false` and `VLM_ENABLED=false` |

**Which should I pick?**

- **Easiest**: Anthropic API — sign up at [console.anthropic.com](https://console.anthropic.com), get an API key, paste it in `.env`. No GPU needed. A full match costs a few dollars.
- **Best accuracy, free**: Self-hosted vLLM with `Qwen/Qwen3-VL-32B-Instruct-FP8` on 2x RTX 3090 (tensor-parallel). This is what the pipeline was tuned on.
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
| **Apple Silicon** (M1-M4) | MPS detection | Redis in Docker, worker + API native with MPS acceleration | ~60 min / match |
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
# 1. Deploy (auto-detects GPU, generates .env, starts stack)
make deploy

# 2. Open the web dashboard
open http://localhost:8080/ui

# 3. Or submit via CLI
python infra/scripts/pipeline_cli.py submit game.mp4 --reel keeper,highlights

# 4. Monitor progress
python infra/scripts/pipeline_cli.py status <job_id> --watch
```

The web dashboard at `http://localhost:8080/ui` lets you submit jobs, pick jerseys, set game start time, and download reels — no terminal needed after initial setup.

## Architecture

```
Source video         Detection Pipeline                        Output
                     ┌─────────────────────────────────┐
  game.mp4  ──────►  │  Phase 1: Motion scan            │
                     │  (frame-differencing, ~300 spikes)│
                     │  ↓                                │
                     │  Phase 2: Audio boost             │
                     │  (whistles, crowd surges)         │
                     │  ↓                                │
                     │  Phase 3: VLM classification      │
                     │  (Qwen3-VL / Claude / Ollama)     │
                     │  ↓                                │  ──►  goalkeeper_reel.mp4
                     │  Phases 3a-3g: Structural         │  ──►  highlights_reel.mp4
                     │  inference (goals, saves, corners) │
                     │  ↓                                │
                     │  Segmentation + Assembly (FFmpeg)  │
                     └─────────────────────────────────┘
```

### Detection Phases

| Phase | Name | What it does |
|-------|------|-------------|
| 1 | Motion scan | Dense frame-differencing to find activity spikes |
| 2 | Audio boost | Whistle + crowd surge detection, fills motion gaps |
| 2b-2d | Match structure | Halftime detection, gap filling, spot-check probes |
| 3 | VLM classify | Single-pass classification of each candidate |
| 3a.5 | Save reclass | Goal kicks with save evidence in reasoning -> save |
| 3a.6 | Shot reclass | Rejected verdicts describing shots -> shot |
| 3b | Goal inference | Kickoff rescan to upgrade shots to goals |
| 3c | Set-piece inference | Corner/goal-kick rescan after shots |
| 3c.5 | Restart reclass | Shot + corner = parry, shot + quick goal kick = miss |
| 3d | Corner scan | Independent corner detection in coverage gaps |
| 3e | Reverse restart | Work backwards from restarts to find missed shots |
| 3f | Shot scan | Binary re-probe of remaining rejected candidates |
| 3g | Catch scan | Structural catch inference + VLM probe for GK holding ball |

See [docs/event-detection-reference.md](docs/event-detection-reference.md) for full details on each phase.

### API

FastAPI app at `http://localhost:8080`:

| Endpoint | Description |
|----------|-------------|
| `POST /jobs` | Submit a job (idempotent via SHA-256) |
| `GET /jobs` | List all jobs |
| `GET /jobs/{id}/status` | Progress + status |
| `POST /jobs/{id}/pause` | Pause processing |
| `POST /jobs/{id}/cancel` | Cancel processing |
| `GET /events/{id}/list` | List detected events |
| `PUT /events/{id}/{event_id}` | Override an event classification |
| `POST /events/{id}/assemble` | Re-assemble reels after edits |
| `GET /reels/{id}/{type}` | Download a reel |
| `GET /ui` | Web dashboard |
| `GET /health` | Liveness check |

### Event Types

17 event types defined in `src/detection/models.py`. Each has per-type clip padding, confidence thresholds, and reel target mappings. Key types:

| Type | Reel | Detection method |
|------|------|-----------------|
| `goal` | both | VLM + kickoff inference |
| `shot_on_target` | both | VLM + shot reclassification |
| `shot_off_target` | highlights | Restart reclassification + reverse restart |
| `shot_stop_diving` | goalkeeper | VLM catch + parry inference (shot + corner) |
| `catch` | goalkeeper | Structural (no restart) + VLM probe |
| `corner_kick` | goalkeeper | VLM + independent corner scan |
| `goal_kick` | goalkeeper | VLM + set-piece inference |

## Development

```bash
make setup              # Install Python deps
make deploy             # Auto-detect hardware, generate .env, start stack
make up                 # Start stack (assumes .env exists)
make down               # Stop all services
make test-unit          # Fast tests, no infra needed (354 tests, <1s)
make test-integration   # Requires Docker
make generate-fixtures  # Create synthetic test videos
make check-gpu          # Verify NVIDIA GPU + container toolkit
```

Run a single test:

```bash
pytest tests/unit/test_clipper.py::test_merge_overlapping_clips -m unit -v
```

### Key Design Decisions

- **Streaming-first**: Video is never fully loaded into RAM — chunked FFmpeg processing throughout
- **GPU-optional**: NVIDIA CUDA, Apple MPS, or CPU — auto-detected at deploy time
- **NAS-safe**: Source files are never modified; output is atomic-write with retry
- **VLM-agnostic**: Any OpenAI-compatible vision endpoint works; Claude API as fallback
- **Structural over visual**: Post-VLM inference uses restart patterns (corner, goal kick, kickoff) rather than trying to visually identify save types at 50m distance

### Project Structure

```
src/
  api/           FastAPI app, routes, Celery worker, web UI
  detection/     Motion scan, audio detection, VLM verifier, pipeline orchestrator
  ingestion/     File intake, SHA-256 hashing, job store, NAS watcher
  segmentation/  Clip boundary computation, padding, merging
  assembly/      FFmpeg clip extraction, reel composition
  tracking/      ByteTrack player tracking (used by spatial filter)
infra/
  scripts/       setup.sh, clean-resubmit.sh, pipeline_cli.py
  docker-compose*.yml
  Dockerfile.*
docs/            Architecture, event detection reference, runbooks
tests/           Unit (354), integration, E2E
```

## Disclaimer

This software is intended for youth and amateur soccer analytics and personal experimentation. It is not affiliated with, endorsed by, or approved by any professional league, federation, or governing body.

This project does not grant rights to process or redistribute copyrighted video content. Users are responsible for compliance with applicable rights and league policies.
