"""
Events review API — allows manual review of detected events before reel assembly.

This enables a workflow where:
  1. Pipeline runs detection
  2. Operator reviews event list, removes false positives, adds missed events
  3. Operator triggers assembly from reviewed event list

Endpoints:
  GET  /events/{job_id}                List all events for a job
  GET  /events/{job_id}/{event_id}     Get single event detail
  PUT  /events/{job_id}/{event_id}     Update event (include/exclude override)
  POST /events/{job_id}/assemble       Trigger assembly from current event list
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = structlog.get_logger(__name__)
router = APIRouter()


class EventUpdateRequest(BaseModel):
    review_override: Optional[bool] = None   # True=force include, False=force exclude, None=auto
    confidence: Optional[float] = None       # Manual confidence override


class EventSummary(BaseModel):
    total: int
    goalkeeper: int
    highlights: int
    by_type: dict[str, int]
    auto_include: int
    manually_excluded: int
    manually_included: int


def _get_store():
    import src.config as cfg
    from src.ingestion.job import JobStore
    return JobStore(Path(cfg.WORKING_DIR) / "jobs")


def _get_event_log(job_id: str):
    import src.config as cfg
    from src.detection.event_log import EventLog
    return EventLog(Path(cfg.WORKING_DIR) / job_id / "events.jsonl")


@router.get("/{job_id}", response_model=EventSummary)
async def list_events_summary(job_id: str):
    """Summary statistics for all detected events."""
    store = _get_store()
    if store.get(job_id) is None:
        raise HTTPException(404, f"Job not found: {job_id}")

    event_log = _get_event_log(job_id)
    events = event_log.read_all()

    by_type: dict[str, int] = {}
    gk_count = hl_count = 0
    auto_include = manually_exc = manually_inc = 0

    for ev in events:
        by_type[ev.event_type.value] = by_type.get(ev.event_type.value, 0) + 1
        if "goalkeeper" in ev.reel_targets:
            gk_count += 1
        if "highlights" in ev.reel_targets:
            hl_count += 1
        if ev.review_override is True:
            manually_inc += 1
        elif ev.review_override is False:
            manually_exc += 1
        else:
            if ev.should_include():
                auto_include += 1

    return EventSummary(
        total=len(events),
        goalkeeper=gk_count,
        highlights=hl_count,
        by_type=by_type,
        auto_include=auto_include,
        manually_excluded=manually_exc,
        manually_included=manually_inc,
    )


@router.get("/{job_id}/list")
async def list_events(
    job_id: str,
    reel_type: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 200,
):
    """List events for a job with optional filtering."""
    store = _get_store()
    if store.get(job_id) is None:
        raise HTTPException(404, f"Job not found: {job_id}")

    event_log = _get_event_log(job_id)
    events = event_log.read_all()

    if reel_type:
        events = [e for e in events if reel_type in e.reel_targets]
    if min_confidence > 0:
        events = [e for e in events if e.confidence >= min_confidence]

    return events[:limit]


@router.put("/{job_id}/{event_id}")
async def update_event(job_id: str, event_id: str, update: EventUpdateRequest):
    """
    Apply manual review override to a single event.
    The update is written back to the event log.
    """
    event_log = _get_event_log(job_id)
    events = event_log.read_all()

    target = next((e for e in events if e.event_id == event_id), None)
    if target is None:
        raise HTTPException(404, f"Event not found: {event_id}")

    # Build updated event
    data = target.model_dump()
    if update.review_override is not None:
        data["review_override"] = update.review_override
        data["reviewed"] = True
    if update.confidence is not None:
        data["confidence"] = max(0.0, min(1.0, update.confidence))

    from src.detection.models import Event
    updated_event = Event(**data)

    # Rewrite log (replace the event in-place)
    # For append-only semantics, append the update; deduplication handles it
    event_log.append(updated_event)

    log.info(
        "events.updated",
        job_id=job_id,
        event_id=event_id,
        review_override=update.review_override,
    )
    return updated_event


@router.post("/{job_id}/assemble")
async def trigger_assembly(job_id: str, reel_types: Optional[list[str]] = None):
    """
    Trigger reel assembly from the current (post-review) event list.
    Useful for re-assembling after manual event edits.
    """
    from src.api.worker import process_match_task
    from src.ingestion.models import JobStatus

    store = _get_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")

    if job.status not in (JobStatus.COMPLETE, JobStatus.FAILED):
        raise HTTPException(409, f"Job must be COMPLETE or FAILED to re-assemble (status: {job.status})")

    # Update job to trigger assembly-only path
    if reel_types:
        data = job.model_dump()
        data["reel_types"] = reel_types
        from src.ingestion.models import Job
        job = Job(**data)
        store.save(job)

    # Skip detection and go straight to segmentation/assembly
    # This is handled by the worker detecting that events.jsonl already exists
    updated = store.update_status(job_id, JobStatus.SEGMENTING, progress=65.0)
    process_match_task.delay(job_id)

    return {"status": "assembly_triggered", "job_id": job_id}
