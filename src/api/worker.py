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


class PipelinePaused(Exception):
    """Raised when a pause request is detected between chunks."""


class PipelineCancelled(Exception):
    """Raised when a cancel request is detected between chunks."""


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
            from src.utils.sleep_inhibitor import SleepInhibitor
            prevent = str(cfg.PREVENT_SLEEP).lower() in ("1", "true", "yes")
            with SleepInhibitor(job_id=job_id, enabled=prevent):
                return _run_pipeline(job_id, store, cfg)
        except PipelinePaused:
            log.info("pipeline.paused", job_id=job_id)
            return {"job_id": job_id, "status": "paused"}
        except PipelineCancelled:
            log.info("pipeline.cancelled", job_id=job_id)
            store.update_status(job_id, JobStatus.CANCELLED, error="Cancelled by user")
            return {"job_id": job_id, "status": "cancelled"}
        except Exception as exc:
            tb = traceback.format_exc()
            log.error("pipeline.failed", job_id=job_id, error=str(exc), traceback=tb)
            store.update_status(job_id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            raise self.retry(exc=exc) if self.request.retries < self.max_retries else exc

    return _process_match


# Try to create the real task; fall back to stub if Celery not installed
try:
    process_match_task = _make_real_task()
    # Expose Celery app at module level for `celery -A src.api.worker worker`
    app = _celery_app
except ImportError:
    log.warning("worker.celery_not_installed", fallback="stub task (tests only)")
    process_match_task = _StubTask()
    app = None

# Backwards-compatible alias
process_match = process_match_task


# ---------------------------------------------------------------------------
# Pipeline execution logic (pure Python, no Celery dependency)
# ---------------------------------------------------------------------------

def _run_pipeline(job_id: str, store: Any, cfg: Any) -> dict:
    """Execute all pipeline stages for a job. Called by the Celery task."""
    from src.assembly.composer import ReelComposer
    from src.assembly.output import write_reel_to_nas, get_output_path, write_job_manifest
    from src.detection.base import NullDetector
    from src.detection.event_classifier import PipelineRunner
    from src.detection.event_log import EventLog
    from src.detection.goalkeeper_detector import GoalkeeperDetector
    from src.detection.models import EventType, is_gk_event_type
    from src.detection.player_detector import PlayerDetector
    from src.ingestion.models import Job, JobStatus
    from src.segmentation.clipper import compute_clips_v2
    from src.segmentation.deduplicator import postprocess_clips
    from src.segmentation.spatial_filter import filter_wrong_side_events, passes_sim_gate

    job = store.get(job_id)
    vf = job.video_file
    working = Path(cfg.WORKING_DIR) / job_id
    working.mkdir(parents=True, exist_ok=True)

    # ── Pre-flight: verify source file is accessible ────────────────────
    if not Path(vf.path).exists():
        error_msg = f"Source file not found: {vf.path}"
        log.error("pipeline.source_not_found", job_id=job_id, path=vf.path)
        store.update_status(job_id, JobStatus.FAILED, progress=0.0, error=error_msg)
        return {"job_id": job_id, "output_paths": {}, "error": error_msg}

    # ── Stage: DETECTING ──────────────────────────────────────────────────
    store.update_status(job_id, JobStatus.DETECTING, progress=5.0)

    if str(cfg.USE_NULL_DETECTOR).lower() in ("1", "true", "yes"):
        log.info("worker.null_detector_enabled", job_id=job_id)
        player_detector = NullDetector(job_id=job_id, source_file=vf.path)
    else:
        player_detector = PlayerDetector(
            job_id=job_id,
            source_file=vf.path,
            model_path=cfg.YOLO_MODEL_PATH,
            use_gpu=str(cfg.USE_GPU).lower() in ("1", "true", "yes"),
            inference_size=int(cfg.YOLO_INFERENCE_SIZE),
            frame_step=int(cfg.DETECTION_FRAME_STEP),
            working_dir=cfg.WORKING_DIR,
        )
    gk_detector = GoalkeeperDetector(job_id=job_id, source_file=vf.path, match_config=job.match_config)
    event_log = EventLog(working / "events.jsonl")
    # Only clear event log for fresh runs — preserve events on resume
    resume_from_chunk = 0
    if job.last_processed_chunk >= 0:
        resume_from_chunk = job.last_processed_chunk + 1
        log.info("worker.resuming", job_id=job_id, from_chunk=resume_from_chunk)
    else:
        event_log.clear()  # Remove stale events from previous failed runs

    # Optionally use ball-first touch detector instead of GK-first path
    ball_touch_detector = None
    use_ball_touch = str(cfg.USE_BALL_TOUCH_DETECTOR).lower() in ("1", "true", "yes")
    if use_ball_touch and job.match_config:
        from src.detection.ball_touch_detector import BallTouchDetector
        ball_touch_detector = BallTouchDetector(
            job_id=job_id,
            source_file=vf.path,
            match_config=job.match_config,
        )
        log.info("worker.ball_touch_detector_enabled", job_id=job_id)

    runner = PipelineRunner(
        job_id=job_id,
        video_file=vf,
        player_detector=player_detector,
        gk_detector=gk_detector,
        event_log=event_log,
        chunk_sec=int(cfg.CHUNK_DURATION_SEC),
        overlap_sec=float(cfg.CHUNK_OVERLAP_SEC),
        min_confidence=float(cfg.MIN_EVENT_CONFIDENCE),
        ball_touch_detector=ball_touch_detector,
        game_start_sec=job.game_start_sec,
        resume_from_chunk=resume_from_chunk,
    )

    # interrupted tracks whether the user paused/cancelled mid-detection.
    # When set, we still finish segmenting + assembling with partial events,
    # then set the final status to PAUSED or CANCELLED instead of COMPLETE.
    interrupted: str | None = None  # "paused" | "cancelled" | None

    def on_detect_progress(pct: float, chunk_idx: int = -1):
        job = store.get(job_id)
        if job and job.cancel_requested:
            raise PipelineCancelled(f"Job {job_id} cancelled by user")
        if job and job.pause_requested:
            raise PipelinePaused(f"Job {job_id} paused by user")
        store.update_status(job_id, JobStatus.DETECTING, progress=5.0 + pct * 0.60)
        # Save chunk checkpoint for resume support
        if chunk_idx >= 0:
            j = store.get(job_id)
            if j is not None:
                data = j.model_dump()
                data["last_processed_chunk"] = chunk_idx
                store.save(Job(**data))

    try:
        total_events = runner.run(progress_callback=on_detect_progress)
    except PipelinePaused:
        interrupted = "paused"
        log.info("pipeline.paused_partial", job_id=job_id)
    except PipelineCancelled:
        interrupted = "cancelled"
        log.info("pipeline.cancelled_partial", job_id=job_id)

    if not interrupted:
        log.info("pipeline.detection_complete", job_id=job_id, total_events=total_events)

    # ── Stage: SEGMENTING ─────────────────────────────────────────────────
    store.update_status(job_id, JobStatus.SEGMENTING, progress=65.0)
    all_events = event_log.read_all()
    event_conf_map = {e.event_id: e.confidence for e in all_events}

    # Apply spatial filter once to all GK events
    gk_events = [e for e in all_events if e.is_goalkeeper_event]
    non_gk_events = [e for e in all_events if not e.is_goalkeeper_event]
    filtered_gk = filter_wrong_side_events(gk_events, all_events, vf.duration_sec)
    # Apply sim gate to GK events
    filtered_gk = [e for e in filtered_gk if passes_sim_gate(e)]
    all_events = filtered_gk + non_gk_events

    # Get reel specs from job (handles legacy reel_types → ReelSpec conversion)
    reel_specs = job.get_reel_specs()

    clips_by_reel: dict[str, list] = {}
    for spec in reel_specs:
        wanted = set()
        for t in spec.event_types:
            try:
                wanted.add(EventType(t))
            except ValueError:
                log.warning("pipeline.unknown_event_type", event_type=t, reel=spec.name)

        filtered = [
            e for e in all_events
            if e.event_type in wanted and e.should_include()
        ]
        if not filtered:
            clips_by_reel[spec.name] = []
            log.info("pipeline.clips_ready", reel_type=spec.name, clips=0)
            continue

        clips = compute_clips_v2(
            events=filtered,
            video_duration=vf.duration_sec,
            reel_name=spec.name,
        )
        clips = postprocess_clips(
            clips,
            reel_type=spec.name,
            max_reel_duration_sec=spec.max_reel_duration_sec,
            event_confidence_map=event_conf_map,
        )
        clips_by_reel[spec.name] = clips
        log.info("pipeline.clips_ready", reel_type=spec.name, clips=len(clips))

    # ── Stage: ASSEMBLING ─────────────────────────────────────────────────
    store.update_status(job_id, JobStatus.ASSEMBLING, progress=70.0)

    output_paths: dict[str, str] = {}
    nonempty_reels = [r for r in clips_by_reel if clips_by_reel[r]]
    reel_count = max(len(nonempty_reels), 1)

    for idx, reel_name in enumerate(nonempty_reels):
        clips = clips_by_reel[reel_name]

        composer = ReelComposer(
            job_id=job_id,
            reel_type=reel_name,
            working_dir=str(working),
            codec=cfg.OUTPUT_CODEC,
            crf=int(cfg.OUTPUT_CRF),
        )

        local_reel = str(working / f"{reel_name}_reel.mp4")
        ok = composer.compose(clips=clips, output_path=local_reel)
        if not ok:
            continue

        try:
            nas_path = write_reel_to_nas(
                local_reel, cfg.NAS_OUTPUT_PATH, job_id, reel_name,
                max_retries=int(cfg.MAX_NAS_RETRY),
            )
            output_paths[reel_name] = nas_path
        except Exception as exc:
            log.error("pipeline.nas_write_failed", reel_type=reel_name, error=str(exc))

        store.update_status(
            job_id, JobStatus.ASSEMBLING,
            progress=70.0 + (idx + 1) / reel_count * 28.0,
        )

    # ── Final status ────────────────────────────────────────────────────
    if interrupted == "paused":
        store.update_status(
            job_id, JobStatus.PAUSED, progress=100.0, output_paths=output_paths,
        )
        log.info("pipeline.paused", job_id=job_id, reels=list(output_paths.keys()))
        return {"job_id": job_id, "status": "paused", "output_paths": output_paths}

    if interrupted == "cancelled":
        store.update_status(
            job_id, JobStatus.CANCELLED, progress=100.0, output_paths=output_paths,
            error="Cancelled by user",
        )
        log.info("pipeline.cancelled", job_id=job_id, reels=list(output_paths.keys()))
        return {"job_id": job_id, "status": "cancelled", "output_paths": output_paths}

    null_mode = str(cfg.USE_NULL_DETECTOR).lower() in ("1", "true", "yes")
    reel_names = [s.name for s in reel_specs]
    if not output_paths and reel_specs and not null_mode:
        # Collect diagnostic info for the error message
        total_events = len(all_events)
        event_types_found = list({e.event_type for e in all_events})
        gk_event_count = sum(1 for e in all_events if e.is_goalkeeper_event)
        error_msg = (
            f"No reels produced for {reel_names}. "
            f"Detection found {total_events} events ({event_types_found}) "
            f"but {gk_event_count} were GK events. "
            f"Clips per reel: {', '.join(f'{r}={len(clips_by_reel.get(r, []))}' for r in reel_names)}"
        )
        store.update_status(
            job_id, JobStatus.FAILED, progress=100.0, error=error_msg,
        )
        log.warning("pipeline.no_reels_produced", job_id=job_id, reel_names=reel_names,
                     total_events=total_events, gk_events=gk_event_count)
        return {"job_id": job_id, "output_paths": {}}

    store.update_status(
        job_id, JobStatus.COMPLETE, progress=100.0, output_paths=output_paths
    )
    log.info("pipeline.complete", job_id=job_id, reels=list(output_paths.keys()))
    return {"job_id": job_id, "output_paths": output_paths}
