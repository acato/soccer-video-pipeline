"""
Spatial false-positive filter for goalkeeper events.

Removes keeper events that are on the wrong side of the pitch using a
two-stage approach:
  1. Midfield gate — reject events in the middle band (0.35–0.65)
  2. Majority-vote side filter — determine "our" keeper's side per half

Also provides a reel-level sim_team_gk quality gate.

Extracted from src/reel_plugins/keeper.py for reuse in the composable-reels
pipeline (which no longer goes through plugins).
"""
from __future__ import annotations

from src.detection.models import Event, EventType

# ---------------------------------------------------------------------------
# Reel-level quality gate
# ---------------------------------------------------------------------------

# Minimum sim_team_gk to include an event in the keeper reel.
_MIN_REEL_SIM_TEAM_GK = 0.75

# Lower threshold for save events — already curated by trajectory analysis.
_MIN_REEL_SIM_SAVE = 0.55

_SAVE_TYPES = frozenset({
    EventType.SHOT_STOP_DIVING,
    EventType.SHOT_STOP_STANDING,
    EventType.PUNCH,
    EventType.CATCH,
    EventType.PENALTY,
})


def passes_sim_gate(event: Event) -> bool:
    """Return True if event has sufficient jersey-color similarity.

    Penalties and corner kicks are exempt — penalties are ML-detected and
    corner kicks are detected from ball trajectory patterns.

    Save events use a lower threshold because they're already curated by
    ball trajectory analysis (direction_change/speed_drop) + goal-area gate.
    """
    if event.event_type in (EventType.PENALTY, EventType.CORNER_KICK):
        return True
    if event.event_type in _SAVE_TYPES:
        return event.metadata.get("sim_team_gk", 0) >= _MIN_REEL_SIM_SAVE
    return event.metadata.get("sim_team_gk", 0) >= _MIN_REEL_SIM_TEAM_GK


# ---------------------------------------------------------------------------
# Spatial false-positive filter
# ---------------------------------------------------------------------------

_MIDFIELD_LO = 0.35
_MIDFIELD_HI = 0.65


def filter_wrong_side_events(
    selected_events: list[Event],
    all_events: list[Event],
    video_duration_sec: float,
) -> list[Event]:
    """Remove keeper events that are on the wrong side of the pitch.

    Two-stage filter:
      1. **Midfield gate** — reject any event whose bounding-box center_x
         falls in the middle band (0.35–0.65).
      2. **Majority-vote side filter** — among the remaining outer-third
         events, count left vs right per game half.  If one side dominates
         (>= 55%, >= 2 events) that's "our" keeper's side and events on
         the opposite side are removed.

    Events without a bounding_box are always kept (safe default).
    """
    half_time = video_duration_sec / 2

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

    first_half_voters = [e for e in voters if e.timestamp_start < half_time]
    second_half_voters = [e for e in voters if e.timestamp_start >= half_time]

    def _majority_side(events: list[Event]) -> str | None:
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
