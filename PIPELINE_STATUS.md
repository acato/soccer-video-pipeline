# Soccer Video Pipeline — Build Status

Last updated: 2026-02-20

## Test Status

| Suite | Tests | Status |
|-------|-------|--------|
| Unit | 202 | ✅ All passing |
| Integration | 26 | ✅ All passing (requires FFmpeg) |
| E2E | 3 | ⏳ Requires running stack (`make deploy`) |

**Total: 228 tests, 0 failures**

## Module Coverage

| Module | Coverage | Notes |
|--------|----------|-------|
| `detection/models.py` | 100% | All event types, confidence logic |
| `detection/event_log.py` | 96% | JSONL append/read/merge |
| `segmentation/clipper.py` | 100% | Padding, merging, filtering |
| `segmentation/deduplicator.py` | 100% | Temporal IoU dedup |
| `ingestion/models.py` | 98% | Job state machine |
| `ingestion/intake.py` | 86% | ffprobe + sha256 |
| `ingestion/job.py` | 96% | JobStore atomic writes |
| `detection/confidence_calibration.py` | 98% | Isotonic-style calibration |
| `api/routes/ui.py` | 100% | Embedded monitoring dashboard |
| `api/metrics.py` | ~80% | Prometheus endpoint |
| `assembly/encoder.py` | 82% | FFmpeg stream copy |
| `api/worker.py` | 90% | Pipeline orchestration, type casting, Celery stub |

*Low coverage modules require GPU/ML models to exercise (player_detector, goalkeeper_detector, tracker, homography, event_classifier)*

## Implemented Components

### ✅ Core Pipeline
- [x] `src/ingestion/` — NAS watcher, ffprobe intake, SHA-256 idempotency, job store
- [x] `src/detection/models.py` — 16 event types, per-event confidence thresholds, reel mapping
- [x] `src/detection/event_log.py` — Append-only JSONL with streaming, dedup, merge
- [x] `src/detection/player_detector.py` — YOLOv8 via FFmpeg frame extraction, cuda/mps/cpu fallback
- [x] `src/detection/goalkeeper_detector.py` — Position + velocity heuristics for GK events
- [x] `src/detection/event_classifier.py` — Shot/tackle/dribble detection, PipelineRunner
- [x] `src/detection/jersey_classifier.py` — HSV K-means clustering for GK ID
- [x] `src/detection/action_classifier.py` — VideoMAE integration + heuristic fallback, cuda/mps/cpu
- [x] `src/detection/confidence_calibration.py` — Per-event-type calibration
- [x] `src/tracking/tracker.py` — ByteTrack wrapper + SimpleIoU fallback
- [x] `src/tracking/gk_tracker.py` — Cross-chunk GK identity persistence
- [x] `src/tracking/homography.py` — Hough line field homography + synthetic fallback
- [x] `src/segmentation/clipper.py` — Padded clip boundaries with merge/clamp
- [x] `src/segmentation/deduplicator.py` — Temporal IoU deduplication
- [x] `src/assembly/encoder.py` — FFmpeg stream copy + re-encode fallback
- [x] `src/assembly/composer.py` — ReelComposer: extract → validate → concat
- [x] `src/assembly/output.py` — NAS atomic delivery with retry
- [x] `src/api/app.py` — FastAPI with all routers + readiness probe
- [x] `src/api/worker.py` — Celery task with full stage chain, config type casting
- [x] `src/api/routes/jobs.py` — Submit, list, status, retry
- [x] `src/api/routes/reels.py` — Reel info + streaming download
- [x] `src/api/routes/events.py` — Event review UI (list, override, re-assemble)
- [x] `src/api/routes/ui.py` — Embedded HTML/JS monitoring dashboard
- [x] `src/api/metrics.py` — Prometheus metrics endpoint

### ✅ Infrastructure
- [x] `docker-compose.yml` — Base stack: redis, api, worker (scalable via `$GPU_COUNT`), flower
- [x] `docker-compose.gpu.yml` — NVIDIA GPU overlay (merged by setup.sh when GPUs detected)
- [x] `docker-compose.redis.yml` — Redis-only (MPS mode: worker runs natively)
- [x] `Dockerfile.api` + `Dockerfile.worker`
- [x] `infra/scripts/setup.sh` — Auto-detecting deploy: NVIDIA/MPS/CPU, generates .env, starts stack
- [x] `infra/scripts/check_nas.sh` — NAS health + bandwidth check
- [x] `infra/scripts/check_gpu.sh` — NVIDIA Container Toolkit verification
- [x] `infra/models/download_models.sh` — YOLOv8m weight download

### ✅ Documentation
- [x] `docs/architecture.md` — C4 context, component, data flow, state machine diagrams
- [x] `docs/adr/ADR-001` — Streaming-first processing
- [x] `docs/adr/ADR-002` — Stream copy output codec
- [x] `docs/contracts/event_schema.json` — JSON Schema for event records
- [x] `docs/contracts/module_interfaces.md` — Binding interface contracts
- [x] `docs/runbooks/runbook_new_match.md` — Ops runbook

## Getting Started

```bash
# 1. Install dependencies
make setup

# 2. Deploy (auto-detects NVIDIA/MPS/CPU, generates .env, downloads models, starts stack)
make deploy
# or non-interactive: infra/scripts/setup.sh /mnt/nas/soccer /mnt/nas/output

# 3. Run unit tests (no docker needed)
make test-unit

# 4. Submit a match
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{"nas_path": "matches/game.mp4", "reel_types": ["goalkeeper", "highlights"]}'

# Monitor: http://localhost:8080/ui
```

## Known Limitations / Next Steps

1. **Action recognition model** — VideoMAE weights need to be fine-tuned on soccer data for best accuracy. Generic Kinetics-400 weights work but event labels won't match exactly.

2. **Jersey classification in frame** — Current implementation uses track.jersey_color_hsv which requires the PlayerDetector to populate this field. Integration test needed with real frames.

3. **Multi-camera handling** — Architecture supports single-camera source only. Multi-angle merging is a future ADR.

4. **Field homography accuracy** — Synthetic fallback (±15m) sufficient for GK zone detection; accurate calibration needs at least 4 visible pitch corner markings.

5. **Near-miss detection** — Currently missing from event_classifier; needs ball trajectory extrapolation toward goal box.

6. **AMD ROCm support** — Add `rocm-smi` detection to `setup.sh` for Linux AMD GPUs. PyTorch-ROCm aliases HIP as CUDA so the pipeline may work out of the box; needs a ROCm Docker image variant and testing.
