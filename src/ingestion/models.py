"""
Data models for the ingestion layer.
These are the canonical Job and VideoFile types used throughout the pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    DETECTING = "detecting"
    SEGMENTING = "segmenting"
    ASSEMBLING = "assembling"
    COMPLETE = "complete"
    FAILED = "failed"


class VideoFile(BaseModel):
    path: str                       # Absolute path on NAS mount (source, read-only)
    filename: str
    duration_sec: float
    fps: float
    width: int
    height: int
    codec: str                      # "h264" or "hevc"
    size_bytes: int
    sha256: str                     # For idempotency — same hash = same job result


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    video_file: VideoFile
    status: JobStatus = JobStatus.PENDING
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reel_types: list[str] = Field(default_factory=lambda: ["goalkeeper", "highlights"])
    output_paths: dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
    progress_pct: float = 0.0

    def with_status(self, status: JobStatus, progress: float = None, error: str = None) -> "Job":
        """Return a new Job with updated status (immutable update pattern)."""
        data = self.model_dump()
        data["status"] = status
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        if progress is not None:
            data["progress_pct"] = progress
        if error is not None:
            data["error"] = error
        elif status != JobStatus.FAILED:
            data["error"] = None          # clear stale error on non-failure transitions
        return Job(**data)
