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
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# Repo root is two levels above this file: src/api/worker.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _stamp_worker_commit() -> None:
    """Write worker_commit.txt stamped with the git HEAD of the repo the worker
    just loaded from. Called on Celery worker_ready so every worker restart
    refreshes the file automatically — no need to run restart_pipeline.sh
    to keep the poller's mismatch guardrail honest.

    Writes to $SOCCER_STATE_DIR/worker_commit.txt (default ~/soccer-runs/state).
    Format matches scripts/restart_pipeline.sh so downstream readers are
    unchanged: short SHA, full SHA, UTC timestamp, pid lines.
    """
    state_dir = Path(os.getenv("SOCCER_STATE_DIR", str(Path.home() / "soccer-runs" / "state")))
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        short_sha = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        full_sha = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        log.warning("worker.commit_stamp_failed", error=str(exc), repo_root=str(_REPO_ROOT))
        return

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    target = state_dir / "worker_commit.txt"
    tmp = target.with_suffix(".txt.tmp")
    body = f"{short_sha}\n{full_sha}\n{ts}\ncelery_pid={os.getpid()}\n"
    try:
        tmp.write_text(body)
        os.replace(tmp, target)
    except OSError as exc:
        log.warning("worker.commit_stamp_write_failed", error=str(exc), target=str(target))
        return
    log.info("worker.commit_stamped", short_sha=short_sha, target=str(target))


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
            task_time_limit=86400,       # 24h — uncapped candidates can run long
            task_soft_time_limit=82800,  # 23h — soft limit before hard kill
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

    # Stamp worker_commit.txt whenever a worker boots, so the poller's
    # mismatch guardrail reflects the actual loaded code — not whatever
    # restart_pipeline.sh last wrote.
    from celery.signals import worker_ready

    @worker_ready.connect
    def _on_worker_ready(sender=None, **_kwargs):
        _stamp_worker_commit()
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
    from src.ingestion.models import Job, JobStatus

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

    # ── Route: dual-pass, VLM-first, or heuristic pipeline ──────────────
    dual_pass = str(cfg.DUAL_PASS_ENABLED).lower() in ("1", "true", "yes")
    if dual_pass:
        return _run_dual_pass_pipeline(job_id, job, store, cfg, working)
    use_vlm = str(cfg.USE_VLM_DETECTION).lower() in ("1", "true", "yes")
    if use_vlm:
        return _run_vlm_pipeline(job_id, job, store, cfg, working)
    return _run_heuristic_pipeline(job_id, job, store, cfg, working)


def _run_vlm_pipeline(job_id: str, job: Any, store: Any, cfg: Any, working: Path) -> dict:
    """VLM-first two-pass detection pipeline."""
    from src.assembly.composer import ReelComposer
    from src.assembly.output import write_reel_to_nas
    from src.detection.vlm_event_detector import VLMEventDetector
    from src.ingestion.models import JobStatus
    from src.segmentation.clipper import clips_from_boundaries

    vf = job.video_file

    # ── Stage: DETECTING (VLM scan) ──────────────────────────────────
    store.update_status(job_id, JobStatus.DETECTING, progress=5.0)

    reel_specs = job.get_reel_specs()
    # Collect all event types requested across reel specs
    all_event_types: list[str] = []
    for spec in reel_specs:
        all_event_types.extend(spec.event_types)
    all_event_types = list(set(all_event_types))

    detector = VLMEventDetector(
        api_key=cfg.ANTHROPIC_API_KEY,
        model=cfg.VLM_MODEL,
        source_file=vf.path,
        video_duration=vf.duration_sec,
        job_id=job_id,
        event_types=all_event_types,
        frame_interval=float(cfg.VLM_FRAME_INTERVAL),
        frame_width=int(cfg.VLM_DETECT_FRAME_WIDTH),
    )

    def on_detect_progress(pct: float):
        store.update_status(job_id, JobStatus.DETECTING, progress=5.0 + pct * 55.0)

    all_events = detector.detect(progress_callback=on_detect_progress)
    log.info("pipeline.vlm_detection_complete", job_id=job_id, events=len(all_events))

    # ── Stage: SEGMENTING ─────────────────────────────────────────────
    store.update_status(job_id, JobStatus.SEGMENTING, progress=60.0)

    # Group events by reel and convert to clip boundaries
    clips_by_reel: dict[str, list] = {}
    for spec in reel_specs:
        wanted_types = set(spec.event_types)
        reel_events = [e for e in all_events if e.event_type.value in wanted_types]
        if not reel_events:
            clips_by_reel[spec.name] = []
            continue

        # Build EventBoundary objects from the VLM-detected events
        from src.detection.models import EventBoundary
        boundaries = [
            EventBoundary(
                event_type=e.event_type.value,
                clip_start_sec=e.timestamp_start,
                clip_end_sec=e.timestamp_end,
                confirmed=True,
                reasoning=e.metadata.get("vlm_reasoning", ""),
            )
            for e in reel_events
        ]

        clips = clips_from_boundaries(
            boundaries=boundaries,
            source_file=vf.path,
            reel_type=spec.name,
        )
        clips_by_reel[spec.name] = clips
        log.info("pipeline.clips_ready", reel_type=spec.name, clips=len(clips))

    # ── Stage: ASSEMBLING ─────────────────────────────────────────────
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

    # ── Final status ────────────────────────────────────────────────
    reel_names = [s.name for s in reel_specs]
    if not output_paths and reel_specs:
        error_msg = (
            f"No reels produced for {reel_names}. "
            f"VLM detection found {len(all_events)} events."
        )
        store.update_status(job_id, JobStatus.FAILED, progress=100.0, error=error_msg)
        log.warning("pipeline.no_reels_produced", job_id=job_id, reel_names=reel_names)
        return {"job_id": job_id, "output_paths": {}}

    store.update_status(job_id, JobStatus.COMPLETE, progress=100.0, output_paths=output_paths)
    log.info("pipeline.complete", job_id=job_id, reels=list(output_paths.keys()))
    return {"job_id": job_id, "output_paths": output_paths}


def _run_dual_pass_pipeline(job_id: str, job: Any, store: Any, cfg: Any, working: Path) -> dict:
    """Dual-pass VLM pipeline: 8B triage → model swap → 32B classification."""
    from src.assembly.composer import ReelComposer
    from src.assembly.output import write_reel_to_nas, write_job_manifest
    from src.detection.dual_pass_detector import DualPassConfig, DualPassDetector
    from src.detection.event_log import EventLog
    from src.ingestion.models import JobStatus
    from src.segmentation.clipper import compute_clips_v2
    from src.segmentation.deduplicator import postprocess_clips

    vf = job.video_file

    # ── Stage: DETECTING (dual-pass VLM) ─────────────────────────────────
    store.update_status(job_id, JobStatus.DETECTING, progress=5.0)

    single_pass = str(getattr(cfg, 'SINGLE_PASS_32B', 'false')).lower() in ("1", "true", "yes")

    _truthy = lambda v: str(v).lower() in ("1", "true", "yes")
    def _parse_int_csv(v) -> tuple[int, ...]:
        s = str(v or "").strip()
        if not s:
            return ()
        return tuple(int(x) for x in s.split(",") if x.strip())
    yolo_grounding = _truthy(getattr(cfg, 'YOLO_GROUNDING_ENABLED', 'false'))

    dp_config = DualPassConfig(
        vllm_url=cfg.VLLM_URL,
        single_pass=single_pass,
        single_pass_step_sec=float(getattr(cfg, 'SINGLE_PASS_STEP_SEC', 10.0)),
        single_pass_window_sec=float(getattr(cfg, 'SINGLE_PASS_WINDOW_SEC', 15.0)),
        single_pass_frames=int(getattr(cfg, 'SINGLE_PASS_FRAMES', 5)),
        yolo_crop_enabled=_truthy(getattr(cfg, 'YOLO_CROP_ENABLED', 'false')),
        field_crop_enabled=_truthy(getattr(cfg, 'FIELD_CROP_ENABLED', 'false')),
        # ball_crop is per-job: job.ball_crop_enabled overrides the env var
        # when set, so operators can opt in for zoomed-out cameras
        # (sporting_ac-style) and out for close cameras (Rush-style).
        field_crop_upscale_long_edge=int(getattr(cfg, 'FIELD_CROP_UPSCALE_LONG_EDGE', 0)),
        ball_crop_enabled=(job.ball_crop_enabled
                           if job.ball_crop_enabled is not None
                           else _truthy(getattr(cfg, 'BALL_CROP_ENABLED', 'false'))),
        # QL1 Pass 2 refinement — per-job override mirrors ball_crop pattern
        refinement_enabled=(job.refinement_enabled
                            if getattr(job, 'refinement_enabled', None) is not None
                            else _truthy(getattr(cfg, 'REFINEMENT_ENABLED', 'false'))),
        # QL2 audio fusion — per-job override
        audio_fusion_enabled=(job.audio_fusion_enabled
                              if getattr(job, 'audio_fusion_enabled', None) is not None
                              else _truthy(getattr(cfg, 'AUDIO_FUSION_ENABLED', 'false'))),
        audio_cache_dir=str(Path(cfg.WORKING_DIR) / "audio_cache"),
        tier1_model_name=cfg.DUAL_PASS_TIER1_NAME,
        tier1_model_path=cfg.DUAL_PASS_TIER1_PATH,
        tier2_model_name=cfg.DUAL_PASS_TIER2_NAME,
        tier2_model_path=cfg.DUAL_PASS_TIER2_PATH,
        step_sec=float(cfg.DUAL_PASS_TRIAGE_STEP),
        swap_script=cfg.DUAL_PASS_SWAP_SCRIPT or "",
        yolo_grounding_enabled=yolo_grounding,
        yolo_grounding_fail_open=_truthy(getattr(cfg, 'YOLO_GROUNDING_FAIL_OPEN', 'true')),
        yolo_grounding_frames=int(getattr(cfg, 'YOLO_GROUNDING_FRAMES', 5)),
        yolo_grounding_frame_span_sec=float(getattr(cfg, 'YOLO_GROUNDING_FRAME_SPAN_SEC', 2.0)),
        yolo_grounding_inference_size=int(getattr(cfg, 'YOLO_GROUNDING_INFERENCE_SIZE', 640)),
        yolo_grounding_ball_conf=float(getattr(cfg, 'YOLO_GROUNDING_BALL_CONF', 0.15)),
        yolo_model_path=str(getattr(cfg, 'YOLO_MODEL_PATH', '')),
        yolo_use_gpu=_truthy(getattr(cfg, 'USE_GPU', 'false')),
        yolo_ball_class_id=int(getattr(cfg, 'YOLO_BALL_CLASS_ID', 32)),
        yolo_person_class_ids=_parse_int_csv(getattr(cfg, 'YOLO_PERSON_CLASS_IDS', '0')),
        yolo_gk_class_ids=_parse_int_csv(getattr(cfg, 'YOLO_GK_CLASS_IDS', '')),
        yolo_gk_proximity_threshold=float(getattr(cfg, 'YOLO_GK_PROXIMITY_THRESHOLD', 0.20)),
        yolo_gk_frames=int(getattr(cfg, 'YOLO_GK_FRAMES', 10)),
        yolo_gk_min_span_sec=float(getattr(cfg, 'YOLO_GK_MIN_SPAN_SEC', 6.0)),
        yolo_gk_inference_size=int(getattr(cfg, 'YOLO_GK_INFERENCE_SIZE', 1280)),
        yolo_trajectory_enabled=_truthy(getattr(cfg, 'YOLO_TRAJECTORY_ENABLED', 'true')),
        yolo_parry_angle_threshold=float(getattr(cfg, 'YOLO_PARRY_ANGLE_THRESHOLD', 90.0)),
        yolo_deflection_angle_threshold=float(getattr(cfg, 'YOLO_DEFLECTION_ANGLE_THRESHOLD', 30.0)),
        yolo_catch_speed_ratio_threshold=float(getattr(cfg, 'YOLO_CATCH_SPEED_RATIO_THRESHOLD', 0.3)),
        yolo_missed_speed_ratio_threshold=float(getattr(cfg, 'YOLO_MISSED_SPEED_RATIO_THRESHOLD', 0.8)),
        yolo_fks_lookback_sec=float(getattr(cfg, 'YOLO_FKS_LOOKBACK_SEC', 5.0)),
        yolo_fks_n_frames=int(getattr(cfg, 'YOLO_FKS_N_FRAMES', 4)),
        yolo_fks_stillness_std_threshold=float(getattr(cfg, 'YOLO_FKS_STILLNESS_STD_THRESHOLD', 0.04)),
        yolo_fks_motion_std_threshold=float(getattr(cfg, 'YOLO_FKS_MOTION_STD_THRESHOLD', 0.08)),
        yolo_ball_chain_enabled=_truthy(getattr(cfg, 'YOLO_BALL_CHAIN_ENABLED', 'true')),
        yolo_ball_max_speed_per_sec=float(getattr(cfg, 'YOLO_BALL_MAX_SPEED_PER_SEC', 0.3)),
    )

    detector = DualPassDetector(
        config=dp_config,
        source_file=vf.path,
        video_duration=vf.duration_sec,
        job_id=job_id,
        working_dir=str(working),
    )

    def on_detect_progress(pct: float):
        store.update_status(job_id, JobStatus.DETECTING, progress=5.0 + pct * 55.0)

    from src.detection.dual_pass_detector import CanaryFailure
    try:
        all_events = detector.detect(progress_callback=on_detect_progress)
    except CanaryFailure as exc:
        log.critical("pipeline.canary_failure", job_id=job_id, error=str(exc))
        store.update_status(job_id, JobStatus.FAILED, error=str(exc))
        raise

    # Write events to event log
    event_log = EventLog(working / "events.jsonl")
    event_log.clear()
    for ev in all_events:
        event_log.append(ev)

    log.info("pipeline.dual_pass_detection_complete",
             job_id=job_id, events=len(all_events))

    # ── Stage: SEGMENTING ────────────────────────────────────────────────
    store.update_status(job_id, JobStatus.SEGMENTING, progress=60.0)

    reel_specs = job.get_reel_specs()
    event_conf_map = {e.event_id: e.confidence for e in all_events}

    from src.detection.models import EventType as ET
    clips_by_reel: dict[str, list] = {}
    for spec in reel_specs:
        wanted = set()
        for t in spec.event_types:
            try:
                wanted.add(ET(t))
            except ValueError:
                log.warning("pipeline.unknown_event_type", event_type=t, reel=spec.name)

        filtered = [e for e in all_events if e.event_type in wanted]
        if not filtered:
            clips_by_reel[spec.name] = []
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

    # ── Stage: ASSEMBLING ────────────────────────────────────────────────
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

    # ── Final status ─────────────────────────────────────────────────────
    reel_names = [s.name for s in reel_specs]
    if not output_paths and reel_specs:
        error_msg = (
            f"No reels produced for {reel_names}. "
            f"Dual-pass detection found {len(all_events)} events."
        )
        store.update_status(job_id, JobStatus.FAILED, progress=100.0, error=error_msg)
        log.warning("pipeline.no_reels_produced", job_id=job_id, reel_names=reel_names)
        return {"job_id": job_id, "output_paths": {}}

    # Write a per-job manifest alongside the reels so downstream tools
    # (review UI, analysis scripts) can inspect the events that fed
    # each reel — including the trajectory_signature tag on GK events
    # (parry / catch / deflection) surfaced by the YOLO grounding gate.
    try:
        def _event_to_dict(e):
            return {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "timestamp_start": e.timestamp_start,
                "timestamp_end": e.timestamp_end,
                "confidence": e.confidence,
                "reel_targets": list(e.reel_targets),
                "trajectory_signature": (
                    e.metadata.get("trajectory_signature")
                    if isinstance(e.metadata, dict) else None
                ),
                "trajectory": (
                    e.metadata.get("trajectory")
                    if isinstance(e.metadata, dict) else None
                ),
            }

        events_by_reel = {
            reel_name: [
                _event_to_dict(e)
                for e in all_events
                if reel_name in e.reel_targets
            ]
            for reel_name in reel_names
        }
        clips_by_reel_meta = {
            reel_name: [
                {
                    "start_sec": round(c.start_sec, 2),
                    "end_sec": round(c.end_sec, 2),
                    "duration_sec": round(c.end_sec - c.start_sec, 2),
                    "primary_event_type": c.primary_event_type,
                    "primary_signature": c.primary_signature,
                    "event_count": len(c.events),
                }
                for c in clips_by_reel.get(reel_name, [])
            ]
            for reel_name in reel_names
        }
        write_job_manifest(
            nas_output_base=cfg.NAS_OUTPUT_PATH,
            job_id=job_id,
            output_paths=output_paths,
            metadata={
                "source_file": vf.filename,
                "video_duration_sec": vf.duration_sec,
                "pipeline": "dual_pass_single_pass_32b" if single_pass else "dual_pass",
                "total_events": len(all_events),
                "events_by_reel": events_by_reel,
                "clips_by_reel": clips_by_reel_meta,
            },
        )
    except Exception as exc:
        log.warning("pipeline.manifest_write_failed", job_id=job_id, error=str(exc))

    store.update_status(job_id, JobStatus.COMPLETE, progress=100.0, output_paths=output_paths)
    log.info("pipeline.complete", job_id=job_id, reels=list(output_paths.keys()))
    return {"job_id": job_id, "output_paths": output_paths}


def _run_heuristic_pipeline(job_id: str, job: Any, store: Any, cfg: Any, working: Path) -> dict:
    """Original heuristic detection pipeline (YOLO + tracking + GK identification)."""
    from src.assembly.composer import ReelComposer
    from src.assembly.output import write_reel_to_nas, get_output_path, write_job_manifest
    from src.detection.event_log import EventLog
    from src.detection.models import EventType, Event
    from src.detection.pipeline import DetectionPipeline
    from src.ingestion.models import Job, JobStatus
    from src.segmentation.clipper import compute_clips_v2
    from src.segmentation.deduplicator import postprocess_clips

    vf = job.video_file

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

    # Two-tier VLM config
    tiered_vlm = str(getattr(cfg, 'TIERED_VLM_ENABLED', 'false')).lower() in ("1", "true", "yes")

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
        # Two-tier VLM
        tiered_vlm=tiered_vlm,
        tier1_model=str(getattr(cfg, 'TIER1_MODEL', '')) if tiered_vlm else "",
        tier1_model_path=str(getattr(cfg, 'TIER1_MODEL_PATH', '')) if tiered_vlm else "",
        tier1_lora_path=str(getattr(cfg, 'TIER1_LORA_PATH', '')) if tiered_vlm else "",
        tier2_model=str(getattr(cfg, 'TIER2_MODEL', '')) if tiered_vlm else "",
        tier2_model_path=str(getattr(cfg, 'TIER2_MODEL_PATH', '')) if tiered_vlm else "",
        tier1_min_confidence=float(getattr(cfg, 'TIER1_MIN_CONFIDENCE', '0.6')),
        tier1_broken_threshold=float(getattr(cfg, 'TIER1_BROKEN_THRESHOLD', '3.0')),
        tier2_spot_check_rate=float(getattr(cfg, 'TIER2_SPOT_CHECK_RATE', '0.10')),
        tier2_escalation_cap=float(getattr(cfg, 'TIER2_ESCALATION_CAP', '0.50')),
        model_swap_script=str(getattr(cfg, 'MODEL_SWAP_SCRIPT', '')),
        model_swap_timeout_sec=int(getattr(cfg, 'MODEL_SWAP_TIMEOUT_SEC', '120')),
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
