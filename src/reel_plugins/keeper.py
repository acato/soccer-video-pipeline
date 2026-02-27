"""
Built-in keeper reel plugins.

Four plugins cover all GK event types with tailored clip padding:
  - KeeperSavesPlugin: diving/standing stops, punches, catches, penalties
  - KeeperGoalKickPlugin: goal kicks (short pre, long post to see ball land)
  - KeeperDistributionPlugin: short/long distribution
  - KeeperOneOnOnePlugin: 1-on-1 situations (generous post to see outcome)

A majority-vote spatial filter removes false positives from the wrong
side of the pitch (opponent GK misidentified as ours).
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
    EventType.PENALTY,
})

# Event types that represent GK distribution.
_DISTRIBUTION_TYPES = frozenset({
    EventType.DISTRIBUTION_SHORT,
    EventType.DISTRIBUTION_LONG,
})


def _targets_keeper_reel(reel_targets: list[str]) -> bool:
    """Return True if any reel target starts with 'keeper'."""
    return any(rt == "keeper" or rt.startswith("keeper_") for rt in reel_targets)


# ---------------------------------------------------------------------------
# Spatial false-positive filter
# ---------------------------------------------------------------------------

_MIDFIELD_LO = 0.35
_MIDFIELD_HI = 0.65


def _filter_wrong_side_events(
    selected_events: list[Event],
    all_events: list[Event],
    video_duration_sec: float,
) -> list[Event]:
    """Remove keeper events that are on the wrong side of the pitch.

    Two-stage filter:
      1. **Midfield gate** — reject any event whose bounding-box center_x
         falls in the middle band (0.35–0.65).  Goalkeepers very rarely
         operate there, so these are almost always false positives.
      2. **Majority-vote side filter** — among the remaining outer-third
         events, count left vs right per game half.  If one side dominates
         (>= 60 %, >= 2 events) that's "our" keeper's side and events on
         the opposite side are removed.

    Events without a bounding_box are always kept (safe default).
    """
    half_time = video_duration_sec / 2

    # Collect ALL keeper events with a bounding box that are in the
    # outer thirds — these are the only reliable voters.
    voters: list[Event] = [
        e for e in all_events
        if e.is_goalkeeper_event
        and e.bounding_box is not None
        and not (_MIDFIELD_LO < e.bounding_box.center_x < _MIDFIELD_HI)
    ]

    # Split voters into first / second half.
    first_half_voters = [e for e in voters if e.timestamp_start < half_time]
    second_half_voters = [e for e in voters if e.timestamp_start >= half_time]

    def _majority_side(events: list[Event]) -> str | None:
        """Return 'left' or 'right' if one side dominates, else None."""
        if len(events) < 2:
            return None
        left = sum(1 for e in events if e.bounding_box.center_x < 0.5)
        right = len(events) - left
        if left / len(events) >= 0.6:
            return "left"
        if right / len(events) >= 0.6:
            return "right"
        return None

    first_side = _majority_side(first_half_voters)
    second_side = _majority_side(second_half_voters)

    def _keep(event: Event) -> bool:
        if event.bounding_box is None:
            return True
        cx = event.bounding_box.center_x

        # Stage 1: reject midfield events outright.
        if _MIDFIELD_LO < cx < _MIDFIELD_HI:
            return False

        # Stage 2: reject events on the wrong side.
        if event.timestamp_start < half_time:
            if first_side == "left":
                return cx < 0.5
            if first_side == "right":
                return cx >= 0.5
        else:
            if second_side == "left":
                return cx < 0.5
            if second_side == "right":
                return cx >= 0.5
        return True  # ambiguous — keep

    return [e for e in selected_events if _keep(e)]


class KeeperSavesPlugin(ReelPlugin):
    """GK save events (dives, standing stops, punches, catches, penalties)."""

    @property
    def name(self) -> str:
        return "keeper_saves"

    @property
    def reel_name(self) -> str:
        return "keeper"

    @property
    def clip_params(self) -> ClipParams:
        return ClipParams(
            pre_pad_sec=8.0,
            post_pad_sec=4.0,
            max_clip_duration_sec=25.0,
            max_reel_duration_sec=20 * 60,
        )

    def select_events(
        self, events: list[Event], ctx: PipelineContext
    ) -> list[Event]:
        selected = [
            e for e in events
            if e.event_type in _SAVE_TYPES
            and (e.is_goalkeeper_event or e.event_type == EventType.PENALTY)
            and _targets_keeper_reel(e.reel_targets)
            and e.should_include()
        ]
        return _filter_wrong_side_events(selected, events, ctx.video_duration_sec)


class KeeperGoalKickPlugin(ReelPlugin):
    """GK goal kick events (short pre, long post to see ball land and be received)."""

    @property
    def name(self) -> str:
        return "keeper_goal_kick"

    @property
    def reel_name(self) -> str:
        return "keeper"

    @property
    def clip_params(self) -> ClipParams:
        return ClipParams(
            pre_pad_sec=1.0,
            post_pad_sec=10.0,
            max_clip_duration_sec=20.0,
            max_reel_duration_sec=20 * 60,
        )

    def select_events(
        self, events: list[Event], ctx: PipelineContext
    ) -> list[Event]:
        selected = [
            e for e in events
            if e.event_type == EventType.GOAL_KICK
            and e.is_goalkeeper_event
            and _targets_keeper_reel(e.reel_targets)
            and e.should_include()
        ]
        return _filter_wrong_side_events(selected, events, ctx.video_duration_sec)


class KeeperDistributionPlugin(ReelPlugin):
    """GK distribution events (short/long distribution, hand throws)."""

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
            post_pad_sec=8.0,
            max_clip_duration_sec=20.0,
            max_reel_duration_sec=20 * 60,
        )

    def select_events(
        self, events: list[Event], ctx: PipelineContext
    ) -> list[Event]:
        selected = [
            e for e in events
            if e.event_type in _DISTRIBUTION_TYPES
            and e.is_goalkeeper_event
            and _targets_keeper_reel(e.reel_targets)
            and e.should_include()
        ]
        return _filter_wrong_side_events(selected, events, ctx.video_duration_sec)


class KeeperOneOnOnePlugin(ReelPlugin):
    """GK 1-on-1 situations (generous padding to capture full play)."""

    @property
    def name(self) -> str:
        return "keeper_one_on_one"

    @property
    def reel_name(self) -> str:
        return "keeper"

    @property
    def clip_params(self) -> ClipParams:
        return ClipParams(
            pre_pad_sec=3.0,
            post_pad_sec=6.0,
            max_clip_duration_sec=30.0,
            max_reel_duration_sec=20 * 60,
        )

    def select_events(
        self, events: list[Event], ctx: PipelineContext
    ) -> list[Event]:
        selected = [
            e for e in events
            if e.event_type == EventType.ONE_ON_ONE
            and e.is_goalkeeper_event
            and _targets_keeper_reel(e.reel_targets)
            and e.should_include()
        ]
        return _filter_wrong_side_events(selected, events, ctx.video_duration_sec)
