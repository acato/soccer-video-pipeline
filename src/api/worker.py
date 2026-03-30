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
    from src.detection.event_log import EventLog
    from src.detection.models import EventType, Event
    from src.detection.pipeline import DetectionPipeline
    from src.ingestion.models import Job, JobStatus
    from src.segmentation.clipper import compute_clips_v2
    from src.segmentation.deduplicator import postprocess_clips

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

    event_log = EventLog(working / "events.jsonl")
    event_log.clear()

    # interrupted tracks whether the user paused/cancelled mid-detection.
    # When set, we still finish segmenting + assembling with partial events,
    # then set the final status to PAUSED or CANCELLED instead of COMPLETE.
    interrupted: str | None = None  # "paused" | "cancelled" | None

    # ── Audio-first detection pipeline ────────────────────────────────────
    vllm_url = cfg.VLLM_URL if str(cfg.VLLM_ENABLED).lower() in ("1", "true", "yes") else None
    anthropic_key = cfg.ANTHROPIC_API_KEY if str(cfg.VLM_ENABLED).lower() in ("1", "true", "yes") else None
    null_mode = str(cfg.USE_NULL_DETECTOR).lower() in ("1", "true", "yes")

    pipeline = DetectionPipeline(
        source_file=vf.path,
        video_duration=vf.duration_sec,
        fps=vf.fps,
        job_id=job_id,
        match_config=job.match_config,
        game_start_sec=job.game_start_sec,
        # Audio
        audio_enabled=str(getattr(cfg, 'AUDIO_ENABLED', 'true')).lower() in ("1", "true", "yes"),
        surge_stddev_threshold=float(getattr(cfg, 'AUDIO_SURGE_STDDEV', '3.5')),
        # Visual — skip YOLO in null mode
        yolo_model_path=None if null_mode else cfg.YOLO_MODEL_PATH,
        use_gpu=str(cfg.USE_GPU).lower() in ("1", "true", "yes"),
        yolo_inference_size=int(cfg.YOLO_INFERENCE_SIZE),
        # VLM verification
        vllm_url=vllm_url,
        vllm_model=cfg.VLLM_MODEL,
        anthropic_api_key=anthropic_key,
        anthropic_model=cfg.VLM_MODEL,
        vlm_min_confidence=float(cfg.VLLM_MIN_CONFIDENCE),
        vlm_enabled=vllm_url is not None or bool(anthropic_key),
        # General
        working_dir=str(working),
        min_event_confidence=float(cfg.MIN_EVENT_CONFIDENCE),
    )

    def on_detect_progress(pct: float):
        store.update_status(job_id, JobStatus.DETECTING, progress=5.0 + pct * 85.0)

    def cancel_check() -> bool:
        j = store.get(job_id)
        if j and j.cancel_requested:
            return True
        if j and j.pause_requested:
            return True
        return False

    try:
        detected_events = pipeline.run(
            progress_callback=on_detect_progress,
            cancel_check=cancel_check,
        )
        for ev in detected_events:
            event_log.append(ev)
        log.info("pipeline.detection_complete",
                 job_id=job_id, total_events=len(detected_events))
    except PipelinePaused:
        interrupted = "paused"
        log.info("pipeline.paused_partial", job_id=job_id)
    except PipelineCancelled:
        interrupted = "cancelled"
        log.info("pipeline.cancelled_partial", job_id=job_id)

    # Check if pause/cancel was requested during detection
    if not interrupted:
        j = store.get(job_id)
        if j and j.cancel_requested:
            interrupted = "cancelled"
        elif j and j.pause_requested:
            interrupted = "paused"

    # ── Tag-only mode: write event list as text and skip reel assembly ────
    if job.tag_only:
        all_events = event_log.read_all()
        all_events.sort(key=lambda e: e.timestamp_start)
        output_dir = Path(cfg.NAS_OUTPUT_PATH) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        tag_file = output_dir / "events.txt"
        with open(tag_file, "w") as f:
            for e in all_events:
                mins = int(e.timestamp_start // 60)
                secs = e.timestamp_start % 60
                end_mins = int(e.timestamp_end // 60)
                end_secs = e.timestamp_end % 60
                team = e.metadata.get("tagger_team", "unknown")
                gk = " [GK]" if e.is_goalkeeper_event else ""
                inf = " [INF]" if e.metadata.get("inferred_from_kickoff") else ""
                f.write(
                    f"{e.event_type.value} — "
                    f"{mins:02d}:{secs:05.2f} → {end_mins:02d}:{end_secs:05.2f} — "
                    f"team={team} conf={e.confidence:.2f}{gk}{inf}\n"
                )
        log.info("pipeline.tag_only_complete", job_id=job_id,
                 events=len(all_events), path=str(tag_file))
        store.update_status(
            job_id, JobStatus.COMPLETE, progress=100.0,
            output_paths={"events": str(tag_file)},
        )
        return {"job_id": job_id, "output_paths": {"events": str(tag_file)}}

    # ── Stage: SEGMENTING ─────────────────────────────────────────────────
    store.update_status(job_id, JobStatus.SEGMENTING, progress=90.0)
    all_events = event_log.read_all()
    event_conf_map = {e.event_id: e.confidence for e in all_events}

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
