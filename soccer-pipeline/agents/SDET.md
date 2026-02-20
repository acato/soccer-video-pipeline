# Agent: SDET

## Role
You own test strategy, test implementation, CI configuration, and quality gates.
You write tests before or alongside feature code (TDD preferred). No feature is
complete without passing tests at the appropriate coverage level.

## Test Pyramid

```
         /\
        /  \  E2E (2–3 full pipeline runs on synthetic match)
       /────\
      /      \  Integration (agent ↔ agent, Celery tasks, DB)
     /────────\
    /          \  Unit (pure functions, models, detectors)
   /────────────\
```

## Fixtures & Test Data

### Synthetic Video Fixtures
Real 4K footage is too large for CI. Create synthetic test fixtures:

```python
# tests/fixtures/video_factory.py
def make_synthetic_clip(
    duration_s: float = 10.0,
    fps: int = 30,
    resolution: tuple = (1920, 1080),  # 1080p for CI speed
    codec: str = "h264",
    scenario: Literal["empty", "gk_save", "goal", "distribution"] = "empty",
) -> Path:
    """
    Generate a synthetic MP4 using FFmpeg lavfi.
    For 'gk_save': overlay a white circle (ball) moving toward a rectangle (goal).
    Returns path to temp file.
    """
```

### Event Fixtures
```python
# tests/fixtures/events.py
SAMPLE_GK_SAVE = VideoEvent(
    event_id=uuid4(),
    source_file=Path("test_match.mp4"),
    start_frame=450, end_frame=630,
    start_ts=15.0, end_ts=21.0,
    event_type=EventType.GK_SAVE_DIVING,
    confidence=0.85,
    ...
)
```

## Unit Tests

### Event Detection
```python
def test_gk_save_detection_high_confidence():
    """Ball trajectory enters GK zone, GK track nearby → save detected."""

def test_event_below_confidence_threshold_discarded():
    """confidence=0.35 → event not written to store."""

def test_deduplication_merges_close_events():
    """Two GK events 5s apart → merged into one extended clip."""
```

### Render
```python
def test_clip_extraction_duration_accurate():
    """Extracted clip duration within 0.1s of requested."""

def test_concat_produces_monotonic_timestamps():
    """Concatenated reel has no PTS discontinuities."""
```

### NAS Client
```python
def test_nas_client_read_only_on_input_dir(tmp_path, monkeypatch):
    """Attempt to write to input_dir raises PermissionError."""

def test_scratch_budget_check_blocks_at_threshold():
    """If scratch <5GB remaining, ScratchBudgetError raised."""
```

## Integration Tests

```python
@pytest.mark.integration
async def test_ingest_to_event_detection_pipeline(celery_app, sqlite_db, tmp_nas):
    """
    Place synthetic MP4 in tmp_nas/input → trigger ingest → 
    verify events written to DB within 60s.
    """

@pytest.mark.integration  
def test_gk_reel_manifest_from_events(event_store, sample_gk_events):
    """Given 8 sample GK events, GK reel manifest has correct ordering and durations."""
```

## E2E Tests

```python
@pytest.mark.e2e
@pytest.mark.slow
def test_full_pipeline_synthetic_match(tmp_nas, pipeline_config):
    """
    Feed a 5-minute synthetic match video.
    Assert: GK reel MP4 exists in output dir.
    Assert: Highlights reel MP4 exists in output dir.
    Assert: All temp files purged from scratch.
    Assert: Job status = DONE.
    """
```

## CI Configuration (GitHub Actions)

```yaml
# .github/workflows/ci.yml
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - pytest tests/unit -x --cov=src --cov-fail-under=80
  
  integration:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7
    steps:
      - pytest tests/integration -x -m integration
  
  e2e:
    runs-on: ubuntu-latest       # GPU runner for full CV stack
    if: github.ref == 'refs/heads/main'
    steps:
      - pytest tests/e2e -x -m e2e --timeout=300
```

## Quality Gates (must pass before merge)

- Unit test coverage ≥ 80% on `src/soccer_pipeline/`
- Zero test failures
- All `VideoEvent` model fields have validation tests
- Render agent: clip duration accuracy test passes
- No `time.sleep()` in tests (use `pytest-asyncio` / Celery eager mode)

## Performance Benchmarks (track, don't gate)

```python
@pytest.mark.benchmark
def test_yolov8_throughput(benchmark, gpu_device):
    """YOLOv8x on 1280px batch=4 should process ≥30 fps."""
    result = benchmark(run_detection_batch, frames=test_frames)
    assert result.stats.mean < 0.033   # 30fps = 33ms/frame
```
