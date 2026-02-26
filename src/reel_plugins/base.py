"""
Core abstractions for the reel plugin system.

A ReelPlugin selects events from the detection output and declares how
they should be clipped.  Multiple plugins can target the same reel —
their clips are merged before assembly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.detection.models import Event
    from src.ingestion.models import MatchConfig
    from src.segmentation.clipper import ClipBoundary


@dataclass(frozen=True)
class ClipParams:
    """How a plugin wants its clips cut."""

    pre_pad_sec: float = 3.0
    post_pad_sec: float = 3.0
    merge_gap_sec: float = 2.0
    max_clip_duration_sec: float = 60.0
    max_reel_duration_sec: float = 1200.0  # 20 min
    min_clip_duration_sec: float = 2.0

    def __post_init__(self) -> None:
        if self.pre_pad_sec < 0:
            raise ValueError(f"pre_pad_sec must be >= 0, got {self.pre_pad_sec}")
        if self.post_pad_sec < 0:
            raise ValueError(f"post_pad_sec must be >= 0, got {self.post_pad_sec}")
        if self.max_clip_duration_sec <= 0:
            raise ValueError(f"max_clip_duration_sec must be > 0, got {self.max_clip_duration_sec}")
        if self.max_reel_duration_sec <= 0:
            raise ValueError(f"max_reel_duration_sec must be > 0, got {self.max_reel_duration_sec}")
        if self.min_clip_duration_sec < 0:
            raise ValueError(f"min_clip_duration_sec must be >= 0, got {self.min_clip_duration_sec}")


@dataclass(frozen=True)
class PipelineContext:
    """Read-only metadata available to plugins during event selection."""

    video_duration_sec: float
    match_config: MatchConfig | None
    keeper_track_ids: dict[str, int | None]  # e.g. {"keeper_a": 42, "keeper_b": None}
    job_id: str


class ReelPlugin(ABC):
    """A plugin that contributes clips to a named reel.

    Lifecycle:
        1. ``select_events`` filters the full event list to this plugin's events.
        2. The orchestrator runs ``compute_clips`` with this plugin's ``clip_params``.
        3. ``post_filter_clips`` gets a chance to reorder / drop clips.
        4. Clips from all plugins sharing the same ``reel_name`` are merged.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique human-readable plugin name, e.g. ``'keeper_saves'``."""

    @property
    @abstractmethod
    def reel_name(self) -> str:
        """Target reel identifier.

        Multiple plugins may share a ``reel_name``; their clips are merged
        into a single output reel.
        """

    @property
    def clip_params(self) -> ClipParams:
        """Override to customize padding, merge gap, and duration caps."""
        return ClipParams()

    @abstractmethod
    def select_events(
        self, events: list[Event], ctx: PipelineContext
    ) -> list[Event]:
        """Filter the full event log to events this plugin wants.

        This **must** be a pure function: no side effects, no I/O.
        Return a subset (or empty list) of the input events.
        """

    def post_filter_clips(
        self, clips: list[ClipBoundary]
    ) -> list[ClipBoundary]:
        """Optional hook to reorder, drop, or annotate clips after clipping.

        Default implementation returns clips unchanged.
        """
        return clips
