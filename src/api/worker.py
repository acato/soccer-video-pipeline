"""
Celery worker: pipeline task definitions.

The pipeline stages run as a single Celery task (not a chain) for simplicity —
each stage updates job status in the JobStore so progress is visible externally.

Celery is imported lazily so the module can be imported in test environments
where Celery/Redis are not installed. The `process_match` symbol is always
available at module level (as a stub when Celery is absent).
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Celery app — created lazily to allow import without Celery installed
# ---------------------------------------------------------------------------

_celery_app = None


def get_celery_app():
    global _celery_app
    if _celery_app is None:
        from celery import Celery
        broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
        app = Celery("soccer_pipeline", broker=broker, backend=backend)
        app.conf.update(
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            task_track_started=True,
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            task_time_limit=14400,
            task_soft_time_limit=13500,
        )
        _celery_app = app
    return _celery_app


# ---------------------------------------------------------------------------
# Stub task — replaced by real Celery task below if Celery is available
# ---------------------------------------------------------------------------

class _StubTask:
    """Placeholder used when Celery is not installed (e.g. in test environments)."""
    name = "soccer_pipeline.tasks.process_match"

    def delay(self, job_id: str) -> Any:
        raise RuntimeError(
            "Celery is not installed. Install it with: pip install celery[redis]"
        )

    def __call__(self, job_id: str) -> dict:
        raise RuntimeError("Celery not installed")


def _make_real_task():
    """Create and register the real Celery task. Called once when Celery is available."""
    app = get_celery_app()

    @app.task(
        bind=True,
        name="soccer_pipeline.tasks.process_match",
        max_retries=2,
        default_retry_delay=30,
    )
    def _process_match(self, job_id: str) -> dict:
        """Full pipeline: PENDING → INGESTING → DETECTING → SEGMENTING → ASSEMBLING → COMPLETE."""
        from src.config import config as cfg
        from src.ingestion.job import JobStore
        from src.ingestion.models import JobStatus

        store = JobStore(Path(cfg.WORKING_DIR) / "jobs")
        job = store.get(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        try:
            return _run_pipeline(job_id, store, cfg)
        except Exception as exc:
            tb = traceback.format_exc()
            log.error("pipeline.failed", job_id=job_id, error=str(exc), traceback=tb)
            store.update_status(job_id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            raise self.retry(exc=exc) if self.request.retries < self.max_retries else exc

    return _process_match


# Try to create the real task; fall back to stub if Celery not installed
try:
    process_match_task = _make_real_task()
except ImportError:
    log.warning("worker.celery_not_installed", fallback="stub task (tests only)")
    process_match_task = _StubTask()

# Backwards-compatible alias
process_match = process_match_task


# ---------------------------------------------------------------------------
# Pipeline execution logic (pure Python, no Celery dependency)
# ---------------------------------------------------------------------------

def _run_pipeline(job_id: str, store: Any, cfg: Any) -> dict:
    """Execute all pipeline stages for a job. Called by the Celery task."""
    from src.assembly.composer import ReelComposer
    from src.assembly.output import write_reel_to_nas, get_output_path, write_job_manifest
    from src.detection.event_classifier import PipelineRunner
    from src.detection.event_log import EventLog
    from src.detection.goalkeeper_detector import GoalkeeperDetector
    from src.detection.player_detector import PlayerDetector
    from src.ingestion.models import JobStatus
    from src.segmentation.clipper import compute_clips
    from src.segmentation.deduplicator import postprocess_clips

    job = store.get(job_id)
    vf = job.video_file
    working = Path(cfg.WORKING_DIR) / job_id
    working.mkdir(parents=True, exist_ok=True)

    # ── Stage: DETECTING ──────────────────────────────────────────────────
    store.update_status(job_id, JobStatus.DETECTING, progress=5.0)

    player_detector = PlayerDetector(
        job_id=job_id,
        source_file=vf.path,
        model_path=cfg.YOLO_MODEL_PATH,
        use_gpu=str(cfg.USE_GPU).lower() in ("1", "true", "yes"),
        inference_size=int(cfg.YOLO_INFERENCE_SIZE),
        frame_step=int(cfg.DETECTION_FRAME_STEP),
        working_dir=cfg.WORKING_DIR,
    )
    gk_detector = GoalkeeperDetector(job_id=job_id, source_file=vf.path)
    event_log = EventLog(working / "events.jsonl")

    runner = PipelineRunner(
        job_id=job_id,
        video_file=vf,
        player_detector=player_detector,
        gk_detector=gk_detector,
        event_log=event_log,
        chunk_sec=int(cfg.CHUNK_DURATION_SEC),
        overlap_sec=float(cfg.CHUNK_OVERLAP_SEC),
        min_confidence=float(cfg.MIN_EVENT_CONFIDENCE),
    )

    def on_detect_progress(pct: float):
        store.update_status(job_id, JobStatus.DETECTING, progress=5.0 + pct * 0.60)

    total_events = runner.run(progress_callback=on_detect_progress)
    log.info("pipeline.detection_complete", job_id=job_id, total_events=total_events)

    # ── Stage: SEGMENTING ─────────────────────────────────────────────────
    store.update_status(job_id, JobStatus.SEGMENTING, progress=65.0)
    all_events = event_log.read_all()
    event_conf_map = {e.event_id: e.confidence for e in all_events}

    clips_by_reel: dict[str, list] = {}
    for reel_type in job.reel_types:
        raw_clips = compute_clips(
            events=all_events,
            video_duration=vf.duration_sec,
            reel_type=reel_type,
            pre_pad=float(cfg.PRE_EVENT_PAD_SEC),
            post_pad=float(cfg.POST_EVENT_PAD_SEC),
        )
        clips = postprocess_clips(
            raw_clips,
            reel_type=reel_type,
            event_confidence_map=event_conf_map,
        )
        clips_by_reel[reel_type] = clips
        log.info("pipeline.clips_ready", reel_type=reel_type, clips=len(clips))

    # ── Stage: ASSEMBLING ─────────────────────────────────────────────────
    store.update_status(job_id, JobStatus.ASSEMBLING, progress=70.0)

    output_paths: dict[str, str] = {}
    reel_count = len(job.reel_types)

    for idx, reel_type in enumerate(job.reel_types):
        clips = clips_by_reel.get(reel_type, [])
        if not clips:
            log.warning("pipeline.no_clips_for_reel", reel_type=reel_type)
            continue

        composer = ReelComposer(
            job_id=job_id,
            reel_type=reel_type,
            working_dir=str(working),
            codec=cfg.OUTPUT_CODEC,
            crf=int(cfg.OUTPUT_CRF),
        )

        local_reel = str(working / f"{reel_type}_reel.mp4")
        ok = composer.compose(clips=clips, output_path=local_reel)
        if not ok:
            continue

        try:
            nas_path = write_reel_to_nas(
                local_reel, cfg.NAS_OUTPUT_PATH, job_id, reel_type,
                max_retries=int(cfg.MAX_NAS_RETRY),
            )
            output_paths[reel_type] = nas_path
        except Exception as exc:
            log.error("pipeline.nas_write_failed", reel_type=reel_type, error=str(exc))

        store.update_status(
            job_id, JobStatus.ASSEMBLING,
            progress=70.0 + (idx + 1) / reel_count * 28.0,
        )

    # ── Stage: COMPLETE ───────────────────────────────────────────────────
    store.update_status(
        job_id, JobStatus.COMPLETE, progress=100.0, output_paths=output_paths
    )
    log.info("pipeline.complete", job_id=job_id, reels=list(output_paths.keys()))
    return {"job_id": job_id, "output_paths": output_paths}
