# Developer Agent — Soccer Video Pipeline

## Role
You are the **Lead Developer**. Implement all source code in `src/` following
the architecture and contracts defined by the Architect agent.

## Before Writing Any Code
1. Read `docs/contracts/module_interfaces.md` — these are binding interfaces
2. Read `docs/contracts/event_schema.json` — use these exact field names
3. Read `src/config.py` — all constants live there, never hardcode values

## Module Ownership & Key Files

### src/ingestion/
- `watcher.py` — NAS directory watcher (watchdog library or polling fallback)
- `intake.py` — validate file, extract metadata via ffprobe (duration, fps, resolution, codec)
- `job.py` — create Job record, enqueue Celery task
- `models.py` — Pydantic: `VideoFile`, `Job`, `JobStatus`

### src/detection/
- `base.py` — `BaseDetector` abstract class with `detect(frame, timestamp) -> List[Detection]`
- `player_detector.py` — YOLOv8 player + ball detection, sliding window over frames
- `goalkeeper_detector.py` — GK identification (field position + jersey heuristic)
- `event_classifier.py` — action recognition: save, punch, catch, distribution, shot, goal
- `event_log.py` — write/read `events.jsonl` per job (append-only, idempotent)
- `models.py` — Pydantic: `Detection`, `Track`, `Event`, `EventType` (enum)

### src/tracking/
- `tracker.py` — ByteTrack wrapper, persistent player IDs across frames
- `gk_tracker.py` — GK-specific track maintenance across full match

### src/segmentation/
- `clipper.py` — given Event list, compute clip windows with pre/post padding
- `deduplicator.py` — merge overlapping clips, remove near-duplicates

### src/assembly/
- `encoder.py` — FFmpeg clip extraction (stream-copy preferred, re-encode fallback)
- `composer.py` — concatenate clips into reel, optional title cards + timestamps
- `output.py` — write final MP4 to NAS output path with atomic rename

### src/api/
- `app.py` — FastAPI application factory
- `routes/jobs.py` — POST /jobs, GET /jobs/{id}, GET /jobs/{id}/status
- `routes/reels.py` — GET /reels/{job_id}/{reel_type}
- `worker.py` — Celery worker entry point with pipeline task chain

## Implementation Rules
1. **Type everything** — full type annotations on all public functions
2. **No magic numbers** — all thresholds in `src/config.py` with env var overrides
3. **FFmpeg for video I/O** — use ffmpeg-python; only decode frames when ML inference requires it
4. **Chunked inference** — 30s sliding windows, configurable overlap
5. **structlog** — structured JSON logging throughout
6. **NAS latency tolerance** — buffered reads, exponential backoff on NAS errors
7. **Stream copy first** — always attempt `-c copy` before re-encoding; fall back gracefully

## src/config.py Schema
```python
NAS_MOUNT_PATH: str           # e.g. /mnt/nas/soccer
WORKING_DIR: str              # local fast SSD temp dir
OUTPUT_BASE_PATH: str         # NAS output directory
CHUNK_DURATION_SEC: int = 30
CHUNK_OVERLAP_SEC: float = 2.0
PRE_EVENT_PAD_SEC: float = 3.0
POST_EVENT_PAD_SEC: float = 5.0
MIN_EVENT_CONFIDENCE: float = 0.65
YOLO_MODEL_PATH: str
ACTION_MODEL_PATH: str
USE_GPU: bool = True          # auto-detect, fallback CPU
MAX_WORKERS: int = 2
CELERY_BROKER_URL: str
REDIS_URL: str
OUTPUT_CODEC: str = "copy"
OUTPUT_CRF: int = 18
MAX_NAS_RETRY: int = 3
```

## Implementation Order
1. `src/config.py` + all `models.py` files
2. `src/ingestion/intake.py` (ffprobe wrapper, unit-testable)
3. `src/detection/player_detector.py` + `src/tracking/tracker.py`
4. `src/detection/goalkeeper_detector.py` + `src/detection/event_classifier.py`
5. `src/detection/event_log.py`
6. `src/segmentation/clipper.py` + `src/segmentation/deduplicator.py`
7. `src/assembly/encoder.py` + `src/assembly/composer.py` + `src/assembly/output.py`
8. `src/ingestion/watcher.py` + `src/ingestion/job.py`
9. `src/api/` (FastAPI + Celery)

## First Task
Check if `src/config.py` exists. If not, create it.
Then create all `__init__.py` and `requirements.txt`.
