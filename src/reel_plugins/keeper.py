"""
Built-in keeper reel plugins.

These replicate the pre-refactor keeper reel behavior:
  - KeeperSavesPlugin: diving/standing stops, punches, catches
  - KeeperDistributionPlugin: goal kicks, short/long distribution
"""
from __future__ import annotations

from src.detection.models import Event, EventType
from src.reel_plugins.base import ClipParams, PipelineContext, ReelPlugin

# Event types that represent a GK save action.
_SAVE_TYPES = frozenset({
    EventType.SHOT_STOP_DIVING,
    EventType.SHOT_STOP_STANDING,
    EventType.PUNCH,
    EventType.CATCH,
    EventType.ONE_ON_ONE,
})

# Event types that represent GK distribution.
_DISTRIBUTION_TYPES = frozenset({
    EventType.GOAL_KICK,
    EventType.DISTRIBUTION_SHORT,
    EventType.DISTRIBUTION_LONG,
})


def _targets_keeper_reel(reel_targets: list[str]) -> bool:
    """Return True if any reel target starts with 'keeper'."""
    return any(rt == "keeper" or rt.startswith("keeper_") for rt in reel_targets)


class KeeperSavesPlugin(ReelPlugin):
    """GK save events (dives, standing stops, punches, catches, 1-on-1s)."""

    @property
    def name(self) -> str:
        return "keeper_saves"

    @property
    def reel_name(self) -> str:
        return "keeper"

    @property
    def clip_params(self) -> ClipParams:
        return ClipParams(
            pre_pad_sec=2.0,
            post_pad_sec=1.5,
            max_clip_duration_sec=15.0,
            max_reel_duration_sec=20 * 60,
        )

    def select_events(
        self, events: list[Event], ctx: PipelineContext
    ) -> list[Event]:
        return [
            e for e in events
            if e.event_type in _SAVE_TYPES
            and e.is_goalkeeper_event
            and _targets_keeper_reel(e.reel_targets)
            and e.should_include()
        ]


class KeeperDistributionPlugin(ReelPlugin):
    """GK distribution events (goal kicks, short/long distribution)."""

    @property
    def name(self) -> str:
        return "keeper_distribution"

    @property
    def reel_name(self) -> str:
        return "keeper"

    @property
    def clip_params(self) -> ClipParams:
        return ClipParams(
            pre_pad_sec=1.0,
            post_pad_sec=3.0,
            max_clip_duration_sec=10.0,
            max_reel_duration_sec=20 * 60,
        )

    def select_events(
        self, events: list[Event], ctx: PipelineContext
    ) -> list[Event]:
        return [
            e for e in events
            if e.event_type in _DISTRIBUTION_TYPES
            and e.is_goalkeeper_event
            and _targets_keeper_reel(e.reel_targets)
            and e.should_include()
        ]
