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


def _passes_sim_gate(event: Event) -> bool:
    """Return True if event has sufficient jersey-color similarity.

    Penalties and corner kicks are exempt — penalties are ML-detected and
    corner kicks are detected from ball trajectory patterns, neither has
    sim_team_gk from jersey color classification.

    Save events (dives, standing stops, punches, catches) use a lower
    threshold because they're already curated by ball trajectory analysis
    (direction_change/speed_drop) + goal-area position gate.
    """
    if event.event_type in (EventType.PENALTY, EventType.CORNER_KICK):
        return True
    if event.event_type in _SAVE_TYPES:
        return event.metadata.get("sim_team_gk", 0) >= _MIN_REEL_SIM_SAVE
    return event.metadata.get("sim_team_gk", 0) >= _MIN_REEL_SIM_TEAM_GK


# ---------------------------------------------------------------------------
# Reel-level quality gate
# ---------------------------------------------------------------------------

# Minimum sim_team_gk to include an event in the keeper reel.
# Events below this threshold are excluded regardless of spatial position.
# This catches FPs from pre-game footage and weak jersey matches where the
# color margin check passed but the absolute similarity is too low.
_MIN_REEL_SIM_TEAM_GK = 0.75

# Lower threshold for save events — already curated by trajectory analysis
# (direction_change/speed_drop) + goal-area GK override.  Spatial filter
# still removes wrong-side events.
_MIN_REEL_SIM_SAVE = 0.55

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

    # Only high-confidence, high-similarity keeper events vote.
    # Goal kicks and corner kicks are excluded from both voting AND filtering:
    # goal kicks inherit bounding_box from the original save/catch event,
    # and corner kicks have bounding_box at ball position (unreliable with
    # auto-panning cameras). Neither should influence side determination.
    # sim_team_gk >= 0.77 ensures only strong jersey-color matches vote,
    # preventing blue-vs-teal FPs from flipping the side determination.
    _VOTER_MIN_CONFIDENCE = 0.75
    _VOTER_MIN_SIM_TEAM_GK = 0.77
    _EXEMPT_FROM_SIDE_FILTER = frozenset({EventType.GOAL_KICK, EventType.CORNER_KICK})

    voters: list[Event] = [
        e for e in all_events
        if e.is_goalkeeper_event
        and e.bounding_box is not None
        and not (_MIDFIELD_LO < e.bounding_box.center_x < _MIDFIELD_HI)
        and e.confidence >= _VOTER_MIN_CONFIDENCE
        and e.metadata.get("sim_team_gk", 0) >= _VOTER_MIN_SIM_TEAM_GK
        and e.event_type not in _EXEMPT_FROM_SIDE_FILTER
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
        if left / len(events) >= 0.55:
            return "left"
        if right / len(events) >= 0.55:
            return "right"
        return None

    first_side = _majority_side(first_half_voters)
    second_side = _majority_side(second_half_voters)

    def _keep(event: Event) -> bool:
        if event.bounding_box is None:
            return True
        # Goal kicks and corner kicks are exempt from side filtering
        if event.event_type in _EXEMPT_FROM_SIDE_FILTER:
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
            post_pad_sec=2.0,
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

            and e.should_include()
            and _passes_sim_gate(e)
        ]
        return _filter_wrong_side_events(selected, events, ctx.video_duration_sec)


class KeeperGoalKickPlugin(ReelPlugin):
    """GK goal kick events (short pre, moderate post to see ball land)."""

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
            post_pad_sec=2.0,
            max_clip_duration_sec=15.0,
            max_reel_duration_sec=20 * 60,
        )

    def select_events(
        self, events: list[Event], ctx: PipelineContext
    ) -> list[Event]:
        selected = [
            e for e in events
            if e.event_type == EventType.GOAL_KICK
            and e.is_goalkeeper_event

            and e.should_include()
            and _passes_sim_gate(e)
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
            post_pad_sec=2.0,
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

            and e.should_include()
            and _passes_sim_gate(e)
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
            post_pad_sec=2.0,
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

            and e.should_include()
            and _passes_sim_gate(e)
        ]
        return _filter_wrong_side_events(selected, events, ctx.video_duration_sec)


class KeeperCornerKickPlugin(ReelPlugin):
    """Corner kick events defended by our GK."""

    @property
    def name(self) -> str:
        return "keeper_corner_kick"

    @property
    def reel_name(self) -> str:
        return "keeper"

    @property
    def clip_params(self) -> ClipParams:
        return ClipParams(
            pre_pad_sec=3.0,
            post_pad_sec=2.0,
            max_clip_duration_sec=25.0,
            max_reel_duration_sec=20 * 60,
        )

    def select_events(
        self, events: list[Event], ctx: PipelineContext
    ) -> list[Event]:
        selected = [
            e for e in events
            if e.event_type == EventType.CORNER_KICK
            and e.is_goalkeeper_event

            and e.should_include()
            and _passes_sim_gate(e)
        ]
        return _filter_wrong_side_events(selected, events, ctx.video_duration_sec)
