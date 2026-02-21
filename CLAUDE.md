# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video processing pipeline that ingests 4K soccer match recordings from a NAS and produces goalkeeper reels (saves, distribution, goal kicks) and highlights reels (shots, goals, key plays). Uses chunked FFmpeg processing (never loads full video into RAM), Celery+Redis for async job orchestration, and YOLOv8/ByteTrack for player/ball detection.

## Build & Test Commands

```bash
make setup                # pip install -r requirements.txt
make deploy               # Auto-detect hardware, generate .env, start stack (preferred)
make test-unit            # Unit tests only (no Docker/FFmpeg needed)
make test-integration     # Spins up Docker test infra, runs integration tests, tears down
make test-e2e             # Full docker-compose stack + E2E tests
make up                   # Start full stack (assumes .env exists)
make down                 # Stop all services (Docker + native)
make check-gpu            # Verify NVIDIA GPU + container toolkit
```

Run a single test file or test:
```bash
pytest tests/unit/test_clipper.py -m unit -v
pytest tests/unit/test_clipper.py::test_merge_overlapping_clips -m unit -v
```

Coverage (CI enforces 75% minimum):
```bash
pytest tests/unit/ -m unit -v --cov=src --cov-report=term-missing --cov-fail-under=75
```

Generate synthetic test videos (requires FFmpeg locally):
```bash
make generate-fixtures
```

## Architecture

### Pipeline Data Flow

```
NAS (read-only) → Intake (ffprobe + SHA-256) → Detection (chunked 30s windows)
  → Segmentation (clip boundaries with padding/merging) → Assembly (FFmpeg concat)
  → NAS Output (atomic write with retry)
```

### Job State Machine

`PENDING → INGESTING → DETECTING → SEGMENTING → ASSEMBLING → COMPLETE` (or `FAILED` with manual retry)

### Key Module Contracts

- **BaseDetector** (`src/detection/base.py`): ABC that all event detectors implement. Per-chunk lifecycle: `detect_frame()` on each frame → `finalize_chunk()` returns Events.
- **PipelineRunner** (`src/detection/event_classifier.py`): Orchestrates all detectors across video chunks, calls `detect_frame` then `finalize_chunk` per chunk.
- **EventLog** (`src/detection/event_log.py`): Append-only JSONL persistence. Deduplication by event_id.
- **compute_clips** (`src/segmentation/clipper.py`): Converts Events → ClipBoundary list with pre/post padding and merge logic.
- **ReelComposer** (`src/assembly/composer.py`): Orchestrates clip extraction → validation → FFmpeg concat.
- **JobStore** (`src/ingestion/job.py`): File-backed JSON persistence for job state. Atomic writes.

### Worker Architecture

`src/api/worker.py` contains the Celery task and `_run_pipeline()` — the single-task orchestrator that chains all stages. Celery is imported lazily so the module works in test environments without Redis. A `_StubTask` placeholder is used when Celery is not installed.

**Config type casting**: `_Config.__getattr__` returns all env var values as strings. `_run_pipeline` explicitly casts them (`int()`, `float()`, `str().lower() in (...)` for bool) before passing to constructors.

### GPU / Device Selection

Three deployment modes, auto-detected by `infra/scripts/setup.sh`:

| Mode | Detection | Worker runs in | GPU device |
|------|-----------|----------------|------------|
| **NVIDIA** | `nvidia-smi` + Docker toolkit | Docker container | `cuda:0` |
| **MPS** | Apple Silicon + PyTorch MPS | Native (Redis in Docker) | `mps` |
| **CPU** | Fallback | Docker container | `cpu` |

Device selection: `PlayerDetector._select_device()` and `ActionClassifier._ensure_loaded()` use a `cuda:0 → mps → cpu` fallback chain. Both handle missing torch gracefully.

### Configuration Pattern

`src/config.py` exposes two interfaces:
1. **Module-level constants** (e.g., `NAS_MOUNT_PATH`) — evaluated at import time from env vars. `_req()` for required, `_opt()` for optional.
2. **`config` object** (`_Config` class) — dynamic accessor that re-reads env vars on each access. Use this in request handlers and tests where env vars are patched per-test via `monkeypatch.setenv`.

Required env vars: `NAS_MOUNT_PATH`, `NAS_OUTPUT_PATH`. Everything else has defaults. See `infra/.env.example` for the full list.

### Testing Patterns

- Tests use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`
- `asyncio_mode = auto` in pytest.ini — async tests just work
- `tests/conftest.py` provides: `set_test_env` (autouse, session-scoped env setup), `mock_config` (per-test env patching via monkeypatch), `sample_video_30s`/`sample_video_10s` (synthetic FFmpeg videos, session-scoped), `sample_events_jsonl` (deterministic event fixtures)
- Unit tests in CI install only minimal deps (no ML/torch) — ML-dependent code paths have low coverage by design
- Synthetic videos use FFmpeg lavfi color source — no real match footage needed

### API

FastAPI app at `src/api/app.py`. Routes in `src/api/routes/`:
- `/jobs` — submit (POST, idempotent via SHA-256), list, status, retry
- `/reels` — info + streaming download
- `/events` — list, override, re-assemble
- `/metrics` — Prometheus counters
- `/ui` — embedded HTML/JS monitoring dashboard (no build step)
- `/health`, `/ready` — liveness + readiness (checks NAS mount)

### Multi-Agent Workspace

Each `agents/{role}/` directory has its own `CLAUDE.md` for specialized sub-agent invocation:
```bash
cd agents/architect && claude       # System design, ADRs
cd agents/developer && claude       # Implementation
cd agents/sdet && claude            # Testing
cd agents/video-analyst && claude   # CV/ML decisions
cd agents/pipeline-operator && claude # Ops/deployment
```

## Infrastructure

- `infra/docker-compose.yml` — Base stack: redis, api, worker (replicas via `$GPU_COUNT`), flower
- `infra/docker-compose.gpu.yml` — NVIDIA GPU overlay (merged automatically when GPUs detected)
- `infra/docker-compose.redis.yml` — Redis-only (used in MPS mode where worker runs natively)
- `infra/scripts/setup.sh` — Auto-detecting deploy script: detects NVIDIA/MPS/CPU, generates `.env`, downloads model weights, starts the appropriate stack
- `infra/scripts/check_gpu.sh` — NVIDIA Container Toolkit verification
- Dockerfiles: `infra/Dockerfile.api` and `infra/Dockerfile.worker` (python:3.11-slim + ffmpeg)

## Event Taxonomy

16 event types defined in `src/detection/models.py` with per-event confidence thresholds and reel target mappings (goalkeeper vs highlights). New detectors should extend `BaseDetector` and register in `PipelineRunner`.
