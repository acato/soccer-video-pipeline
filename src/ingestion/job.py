"""
Job store and Celery task enqueuing.

Jobs are persisted as JSON files in WORKING_DIR/jobs/{job_id}.json
so they survive worker restarts without a database dependency.
Redis is used only for the Celery task queue, not for job state.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from src.ingestion.models import Job, JobStatus, MatchConfig, VideoFile

log = structlog.get_logger(__name__)


class JobStore:
    """
    Simple file-backed job registry.
    Thread-safe for concurrent reads; writes use atomic rename.
    """

    def __init__(self, jobs_dir: str | Path):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def save(self, job: Job) -> Job:
        """Persist job to disk atomically."""
        path = self._path(job.job_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(job.model_dump_json(indent=2))
        tmp.replace(path)
        log.debug("job.saved", job_id=job.job_id, status=job.status)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Load a job by ID, or None if not found."""
        path = self._path(job_id)
        if not path.exists():
            return None
        return Job.model_validate_json(path.read_text())

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: float = None,
        error: str = None,
        output_paths: dict[str, str] = None,
    ) -> Optional[Job]:
        """Load, update status, and persist. Returns updated job or None if not found."""
        job = self.get(job_id)
        if job is None:
            log.warning("job.not_found", job_id=job_id)
            return None
        job = job.with_status(status, progress, error)
        if output_paths:
            data = job.model_dump()
            data["output_paths"].update(output_paths)
            from src.ingestion.models import Job as JobModel
            job = JobModel(**data)
        return self.save(job)

    def list_all(self) -> list[Job]:
        """Return all jobs sorted by creation time (newest first)."""
        jobs = []
        for p in self.jobs_dir.glob("*.json"):
            try:
                jobs.append(Job.model_validate_json(p.read_text()))
            except Exception as exc:
                log.warning("job.corrupt_file", path=str(p), error=str(exc))
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def request_pause(self, job_id: str) -> Optional[Job]:
        """Set pause_requested flag on a job."""
        job = self.get(job_id)
        if job is None:
            return None
        data = job.model_dump()
        data["pause_requested"] = True
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated = Job(**data)
        return self.save(updated)

    def request_cancel(self, job_id: str) -> Optional[Job]:
        """Set cancel_requested flag on a job."""
        job = self.get(job_id)
        if job is None:
            return None
        data = job.model_dump()
        data["cancel_requested"] = True
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated = Job(**data)
        return self.save(updated)

    def delete(self, job_id: str) -> bool:
        """Remove a job file from disk. Returns True if deleted, False if not found."""
        path = self._path(job_id)
        if not path.exists():
            return False
        path.unlink()
        log.info("job.deleted", job_id=job_id)
        return True

    def _path(self, job_id: str) -> Path:
        # Sanitize job_id to prevent path traversal
        safe_id = job_id.replace("/", "").replace("..", "")
        return self.jobs_dir / f"{safe_id}.json"


def create_job(
    video_file: VideoFile,
    reel_types: list[str],
    store: JobStore,
    match_config: Optional[MatchConfig] = None,
) -> Job:
    """
    Create a new Job record, persist it, and enqueue for processing.
    Returns the newly created Job.
    """
    job = Job(video_file=video_file, reel_types=reel_types, match_config=match_config)
    store.save(job)
    log.info(
        "job.created",
        job_id=job.job_id,
        filename=video_file.filename,
        reel_types=reel_types,
        duration_sec=video_file.duration_sec,
    )
    # Celery enqueue is done by the API layer to avoid circular imports
    return job
