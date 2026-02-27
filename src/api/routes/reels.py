"""
Reel and event management endpoints.

GET  /reels/{job_id}/{reel_type}                    Reel metadata
GET  /reels/{job_id}/{reel_type}/download           Stream reel file
GET  /reels/{job_id}/{reel_type}/events             List events for this reel
PUT  /reels/{job_id}/{reel_type}/events/{event_id}  Override event include/exclude
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

log = structlog.get_logger(__name__)
router = APIRouter()


class ReelInfo(BaseModel):
    job_id: str
    reel_type: str
    path: str
    size_bytes: int
    size_mb: float
    clip_count: int = 0


class EventOverrideRequest(BaseModel):
    include: bool   # True = force include, False = force exclude


def _get_store():
    import src.config as cfg
    from src.ingestion.job import JobStore
    return JobStore(Path(cfg.WORKING_DIR) / "jobs")


def _get_job_or_404(job_id: str):
    job = _get_store().get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


@router.get("/{job_id}/{reel_type}", response_model=ReelInfo)
def get_reel_info(job_id: str, reel_type: str):
    """Get reel metadata. Returns 409 if job not yet complete."""
    from src.ingestion.models import JobStatus
    job = _get_job_or_404(job_id)

    has_reels = {JobStatus.COMPLETE, JobStatus.PAUSED, JobStatus.CANCELLED}
    if job.status not in has_reels:
        raise HTTPException(
            409,
            f"Job not complete yet (status: {job.status}, {job.progress_pct:.0f}%)"
        )

    reel_path = job.output_paths.get(reel_type)
    if not reel_path or not Path(reel_path).exists():
        raise HTTPException(404, f"Reel '{reel_type}' not found for job {job_id}")

    size = Path(reel_path).stat().st_size
    return ReelInfo(
        job_id=job_id,
        reel_type=reel_type,
        path=reel_path,
        size_bytes=size,
        size_mb=round(size / 1024 / 1024, 1),
    )


@router.get("/{job_id}/{reel_type}/download")
def download_reel(job_id: str, reel_type: str):
    """Stream the reel MP4 file."""
    from src.ingestion.models import JobStatus
    job = _get_job_or_404(job_id)
    has_reels = {JobStatus.COMPLETE, JobStatus.PAUSED, JobStatus.CANCELLED}
    if job.status not in has_reels:
        raise HTTPException(409, "Job not complete")

    reel_path = job.output_paths.get(reel_type)
    if not reel_path or not Path(reel_path).exists():
        raise HTTPException(404, f"Reel '{reel_type}' not found")

    filename = Path(reel_path).name
    return FileResponse(
        reel_path, media_type="video/mp4", filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{job_id}/{reel_type}/events")
def list_reel_events(job_id: str, reel_type: str, include_excluded: bool = False):
    """
    List detected events for a specific reel type.
    Useful for reviewing/overriding detection quality.
    """
    import src.config as cfg
    from src.detection.event_log import EventLog

    _get_job_or_404(job_id)

    event_log = EventLog(Path(cfg.WORKING_DIR) / job_id / "events.jsonl")
    if not event_log.exists():
        return {"events": [], "total": 0}

    events = event_log.filter_by_reel(reel_type)
    if not include_excluded:
        events = [e for e in events if e.should_include()]

    return {
        "events": [e.model_dump() for e in events],
        "total": len(events),
        "reel_type": reel_type,
        "job_id": job_id,
    }


@router.put("/{job_id}/{reel_type}/events/{event_id}")
def override_event(job_id: str, reel_type: str, event_id: str, body: EventOverrideRequest):
    """
    Manually override whether an event is included in the reel.
    Sets review_override=True/False on the event record.

    After overriding, re-run assembly via POST /jobs/{job_id}/retry to apply.
    """
    import src.config as cfg
    from src.detection.event_log import EventLog
    from src.detection.models import Event

    _get_job_or_404(job_id)

    log_path = Path(cfg.WORKING_DIR) / job_id / "events.jsonl"
    event_log = EventLog(log_path)

    events = event_log.read_all()
    target = next((e for e in events if e.event_id == event_id), None)
    if target is None:
        raise HTTPException(404, f"Event {event_id} not found in job {job_id}")

    # Update override
    updated_data = target.model_dump()
    updated_data["review_override"] = body.include
    updated_data["reviewed"] = True
    updated = Event(**updated_data)

    # Rewrite log with updated event (dedup-by-id will keep latest version)
    event_log.append(updated)

    log.info(
        "event.override_set",
        job_id=job_id, event_id=event_id,
        include=body.include, event_type=updated.event_type
    )
    return {"event_id": event_id, "review_override": body.include, "status": "updated"}
