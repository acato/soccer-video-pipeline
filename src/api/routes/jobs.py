"""
Job management endpoints.

POST   /jobs                  Submit a new processing job
GET    /jobs                  List all jobs
GET    /jobs/{job_id}         Get full job record
GET    /jobs/{job_id}/status  Get lightweight status + progress
POST   /jobs/{job_id}/retry   Re-queue a failed job
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import at module level so tests can patch src.api.routes.jobs.process_match_task
from src.api.worker import process_match_task
from src.detection.jersey_classifier import JERSEY_COLOR_PALETTE
from src.ingestion.intake import extract_metadata
from src.ingestion.models import MatchConfig

log = structlog.get_logger(__name__)
router = APIRouter()


class SubmitJobRequest(BaseModel):
    nas_path: str
    match_config: Optional[MatchConfig] = None
    kit_name: Optional[str] = None
    reel_types: list[str] = ["keeper", "highlights"]
    game_start_sec: float = 0.0


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_pct: float
    error: Optional[str] = None


def _get_store():
    from src.config import config as dyn_cfg
    from src.ingestion.job import JobStore
    return JobStore(Path(dyn_cfg.WORKING_DIR) / "jobs")


def _build_match_config_from_team(kit_name: str) -> MatchConfig:
    """Build a MatchConfig from the saved team config + selected kit name."""
    from src.api.routes.team import load_team_config
    from src.ingestion.models import KitConfig

    team_cfg = load_team_config()
    if team_cfg is None:
        raise HTTPException(400, "No team configured. Run ./setup-team.sh first.")
    kits = team_cfg.get("kits", {})
    if kit_name not in kits:
        available = ", ".join(kits.keys()) if kits else "(none)"
        raise HTTPException(400, f"Kit '{kit_name}' not found. Available kits: {available}")
    kit = kits[kit_name]
    team_name = team_cfg.get("team_name", "My Team")
    return MatchConfig(
        team=KitConfig(team_name=team_name, outfield_color=kit["outfield_color"], gk_color=kit["gk_color"]),
        opponent=KitConfig(team_name="Opponent", outfield_color="white", gk_color="neon_yellow"),
    )


@router.post("", status_code=201)
@router.post("/", status_code=201, include_in_schema=False)
def submit_job(request: SubmitJobRequest):
    """Submit a match video for processing. Idempotent: same file returns existing job."""
    import src.config as cfg
    from src.config import config as dyn_cfg
    from src.ingestion.job import JobStore, create_job

    valid_reels = {"keeper", "highlights", "player"}
    invalid = [r for r in request.reel_types if r not in valid_reels]
    if invalid:
        raise HTTPException(400, f"Invalid reel types: {invalid}. Valid: {sorted(valid_reels)}")

    # Build match_config from team config if not provided directly
    match_config = request.match_config
    if match_config is None:
        kit_name = request.kit_name or "Home"
        match_config = _build_match_config_from_team(kit_name)

    invalid_colors = []
    for role, kit in [("team", match_config.team), ("opponent", match_config.opponent)]:
        for field, color in [("outfield_color", kit.outfield_color), ("gk_color", kit.gk_color)]:
            key = color.lower().replace(" ", "_").replace("-", "_")
            if key not in JERSEY_COLOR_PALETTE:
                invalid_colors.append(f"{role}.{field}={color!r}")
    if invalid_colors:
        valid = ", ".join(sorted(JERSEY_COLOR_PALETTE))
        raise HTTPException(400, f"Unknown jersey color(s): {invalid_colors}. Valid: {valid}")

    full_path = str(Path(dyn_cfg.NAS_MOUNT_PATH) / request.nas_path.lstrip("/"))
    if not Path(full_path).exists():
        raise HTTPException(404, f"Video file not found: {full_path}")

    try:
        video_file = extract_metadata(full_path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, f"Failed to read video metadata: {exc}")

    store = _get_store()

    # Idempotency check — same SHA-256 = same file
    existing = _find_by_hash(store, video_file.sha256)
    if existing:
        from src.ingestion.models import JobStatus
        if existing.status == JobStatus.FAILED:
            # Re-queue failed jobs so callers don't get stuck on a stale failure
            store.update_status(existing.job_id, JobStatus.PENDING, progress=0.0, error=None)
            process_match_task.delay(existing.job_id)
            log.info("jobs.requeued_failed", job_id=existing.job_id)
            return store.get(existing.job_id)
        log.info("jobs.idempotent", job_id=existing.job_id)
        return existing

    job = create_job(video_file, request.reel_types, store, match_config, game_start_sec=request.game_start_sec)
    process_match_task.delay(job.job_id)
    log.info("jobs.submitted", job_id=job.job_id, filename=video_file.filename)
    return job


@router.get("")
@router.get("/", include_in_schema=False)
def list_jobs(limit: int = 50):
    return _get_store().list_all()[:limit]


@router.get("/{job_id}")
def get_job(job_id: str):
    job = _get_store().get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


@router.get("/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = _get_store().get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        error=job.error,
    )


@router.post("/{job_id}/pause")
def pause_job(job_id: str):
    """Request pause of an active job. Takes effect at next chunk boundary."""
    from src.ingestion.models import JobStatus
    store = _get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    active = {JobStatus.PENDING, JobStatus.INGESTING, JobStatus.DETECTING,
              JobStatus.SEGMENTING, JobStatus.ASSEMBLING}
    if job.status not in active:
        raise HTTPException(400, f"Only active jobs can be paused (current: {job.status})")
    if job.status == JobStatus.PENDING:
        # Not yet picked up by a worker — go directly to PAUSED
        store.update_status(job_id, JobStatus.PAUSED, progress=0.0)
    else:
        store.request_pause(job_id)
    return store.get(job_id)


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str):
    """Request cancellation. Takes effect at next chunk boundary."""
    from src.ingestion.models import JobStatus
    store = _get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    terminal = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED}
    if job.status in terminal:
        raise HTTPException(400, f"Cannot cancel a {job.status} job")
    if job.status in (JobStatus.PAUSED, JobStatus.PENDING):
        # No worker running — set CANCELLED directly
        store.update_status(job_id, JobStatus.CANCELLED, error="Cancelled by user")
    else:
        store.request_cancel(job_id)
    return store.get(job_id)


@router.post("/{job_id}/resume")
def resume_job(job_id: str):
    """Resume a paused job. Re-queues from the beginning."""
    from src.ingestion.models import JobStatus
    store = _get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status != JobStatus.PAUSED:
        raise HTTPException(400, f"Only paused jobs can be resumed (current: {job.status})")
    store.update_status(job_id, JobStatus.PENDING, progress=0.0, error=None)
    process_match_task.delay(job_id)
    log.info("jobs.resumed", job_id=job_id)
    return store.get(job_id)


@router.post("/{job_id}/retry")
def retry_job(job_id: str):
    from src.ingestion.models import JobStatus
    store = _get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(400, f"Job must be FAILED or CANCELLED to retry (current: {job.status})")
    updated = store.update_status(job_id, JobStatus.PENDING, progress=0.0, error=None)
    process_match_task.delay(job_id)
    log.info("jobs.retry", job_id=job_id)
    return updated


@router.delete("/{job_id}")
def delete_job(job_id: str):
    """Delete a completed, failed, paused, or cancelled job."""
    from src.ingestion.models import JobStatus
    store = _get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    deletable = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.PAUSED, JobStatus.CANCELLED}
    if job.status not in deletable:
        raise HTTPException(
            400, f"Only completed, failed, paused, or cancelled jobs can be deleted (current: {job.status})"
        )
    store.delete(job_id)
    log.info("jobs.deleted", job_id=job_id)
    return {"deleted": True, "job_id": job_id}


def _find_by_hash(store, sha256: str):
    for job in store.list_all():
        if job.video_file.sha256 == sha256:
            return job
    return None
