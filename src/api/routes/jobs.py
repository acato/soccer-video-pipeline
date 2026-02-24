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
    match_config: MatchConfig
    reel_types: list[str] = ["keeper", "highlights"]


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_pct: float
    error: Optional[str] = None


def _get_store():
    from src.config import config as dyn_cfg
    from src.ingestion.job import JobStore
    return JobStore(Path(dyn_cfg.WORKING_DIR) / "jobs")


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

    invalid_colors = []
    for role, kit in [("team", request.match_config.team), ("opponent", request.match_config.opponent)]:
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

    job = create_job(video_file, request.reel_types, store, request.match_config)
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


@router.post("/{job_id}/retry")
def retry_job(job_id: str):
    from src.ingestion.models import JobStatus
    store = _get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status not in (JobStatus.FAILED,):
        raise HTTPException(400, f"Job must be FAILED to retry (current: {job.status})")
    updated = store.update_status(job_id, JobStatus.PENDING, progress=0.0, error=None)
    process_match_task.delay(job_id)
    log.info("jobs.retry", job_id=job_id)
    return updated


@router.delete("/{job_id}")
def delete_job(job_id: str):
    """Delete a completed or failed job."""
    from src.ingestion.models import JobStatus
    store = _get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status not in (JobStatus.COMPLETE, JobStatus.FAILED):
        raise HTTPException(
            400, f"Only completed or failed jobs can be deleted (current: {job.status})"
        )
    store.delete(job_id)
    log.info("jobs.deleted", job_id=job_id)
    return {"deleted": True, "job_id": job_id}


def _find_by_hash(store, sha256: str):
    for job in store.list_all():
        if job.video_file.sha256 == sha256:
            return job
    return None
