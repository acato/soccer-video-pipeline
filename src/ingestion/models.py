"""
Data models for the ingestion layer.
These are the canonical Job and VideoFile types used throughout the pipeline.
"""
from __future__ import annotations

import re
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
    PAUSED = "paused"
    CANCELLED = "cancelled"


class KitConfig(BaseModel):
    """Jersey colors worn by one team in a specific match."""
    team_name: str          # e.g. "Foobar FC"
    outfield_color: str     # e.g. "dark_blue" — see JERSEY_COLOR_PALETTE
    gk_color: str           # e.g. "neon_yellow"

    @property
    def team_slug(self) -> str:
        """Filename/reel-safe identifier: 'Foobar FC' → 'foobar_fc_gk'."""
        slug = re.sub(r"[^a-z0-9]+", "_", self.team_name.lower()).strip("_")
        return f"{slug}_gk"


class MatchConfig(BaseModel):
    """Per-match kit context for a single team.

    `team`     — the team this job is generating reels for.
    `opponent` — the opposing team; needed so the pipeline can distinguish
                 the two GKs by jersey color, but no reels are produced for them.

    Colors are per-match, not per-team, because teams switch between
    home/away/third kits depending on the fixture.
    """
    team: KitConfig
    opponent: KitConfig


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
    reel_types: list[str] = Field(default_factory=lambda: ["keeper", "highlights"])
    match_config: Optional[MatchConfig] = None
    output_paths: dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None
    progress_pct: float = 0.0
    pause_requested: bool = False
    cancel_requested: bool = False

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
        # Clear flags when resuming (PENDING) or cancelling
        if status == JobStatus.PENDING:
            data["pause_requested"] = False
            data["cancel_requested"] = False
        elif status == JobStatus.CANCELLED:
            data["pause_requested"] = False
        return Job(**data)
