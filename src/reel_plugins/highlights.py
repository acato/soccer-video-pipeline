"""
Built-in highlights reel plugin.

Replicates the pre-refactor highlights reel behavior: shots, goals,
near-misses, penalties, and free-kick shots.  DRIBBLE_SEQUENCE and
TACKLE are excluded (they were removed from highlights in Feb 2026
because they flooded the reel with routine play).
"""
from __future__ import annotations

from src.detection.models import Event, EventType
from src.reel_plugins.base import ClipParams, PipelineContext, ReelPlugin

_HIGHLIGHTS_TYPES = frozenset({
    EventType.SHOT_ON_TARGET,
    EventType.SHOT_OFF_TARGET,
    EventType.GOAL,
    EventType.NEAR_MISS,
    EventType.PENALTY,
    EventType.FREE_KICK_SHOT,
})


class HighlightsShotsPlugin(ReelPlugin):
    """Shot/goal events for the highlights reel."""

    @property
    def name(self) -> str:
        return "highlights_shots"

    @property
    def reel_name(self) -> str:
        return "highlights"

    @property
    def clip_params(self) -> ClipParams:
        return ClipParams(
            pre_pad_sec=3.0,
            post_pad_sec=5.0,
            max_clip_duration_sec=90.0,
            max_reel_duration_sec=15 * 60,
        )

    def select_events(
        self, events: list[Event], ctx: PipelineContext
    ) -> list[Event]:
        return [
            e for e in events
            if e.event_type in _HIGHLIGHTS_TYPES
            and "highlights" in e.reel_targets
            and e.should_include()
        ]
