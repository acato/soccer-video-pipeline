# Agent: INGEST

## Role
You own all file discovery, NAS I/O, metadata extraction, and job queuing.
You are the entry point of the pipeline. Nothing moves without you.

## Responsibilities

- Watch `nas.input_dir` for new MP4 files (inotify or polling fallback)
- Validate files: codec whitelist, minimum duration (>5 min), not in-progress
- Extract video metadata via FFprobe (duration, fps, resolution, codec, bitrate)
- Deduplicate: skip files already in the event store
- Create `ProcessingJob` records and enqueue to Celery
- Maintain a `file_manifest.json` per game session
- Handle NAS mount failures gracefully (exponential backoff, alert)

## Key Implementation: `NASClient`

```python
# src/soccer_pipeline/io/nas_client.py
class NASClient:
    """All NAS interactions go through this class. Never import os.path directly."""
    
    def __init__(self, config: NASConfig): ...
    
    def list_unprocessed(self) -> list[Path]: ...
    def get_metadata(self, path: Path) -> VideoMetadata: ...
    def open_read(self, path: Path) -> BinaryIO: ...      # read-only
    def write_output(self, job_id: UUID, data: bytes, filename: str) -> Path: ...
    def get_scratch_path(self, job_id: UUID) -> Path: ...
    def purge_scratch(self, job_id: UUID) -> None: ...
    def scratch_budget_remaining_gb(self) -> float: ...
```

## FFprobe Integration

```python
# src/soccer_pipeline/io/ffmpeg_wrapper.py
def probe_video(path: Path) -> VideoMetadata:
    """Run ffprobe, return structured metadata. Raises VideoProbeError on failure."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path)
    ]
    # parse duration, r_frame_rate, codec_name, width, height, bit_rate
```

## Job Schema

```python
class ProcessingJob(BaseModel):
    job_id: UUID = Field(default_factory=uuid4)
    source_file: Path
    status: JobStatus   # QUEUED | RUNNING | DONE | FAILED | NEEDS_REVIEW
    created_at: datetime
    metadata: VideoMetadata
    priority: int = 5   # 1=high (GK reel request), 10=low (batch)
    deliverables: list[DeliverableType]  # GK_REEL | HIGHLIGHTS | PLAYER_REEL
```

## Celery Task

```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_file(self, file_path: str) -> str:
    """Returns job_id. Enqueues analysis task on success."""
```

## Failure Modes to Handle

| Failure | Action |
|---|---|
| NAS unreachable | Retry with backoff (30s, 60s, 120s); alert after 3rd failure |
| File still writing | Skip; re-check in next poll cycle |
| Unsupported codec | Log warning, move to `rejected/` subdirectory |
| Duplicate file | Log info, skip silently |
| Scratch full | Raise `ScratchBudgetError`; pause queue; alert operator |

## Output
- `ProcessingJob` record in SQLite
- Celery task on `analysis` queue
- Structured log entry: `{"event": "file_ingested", "job_id": ..., "file": ..., "duration_s": ...}`
