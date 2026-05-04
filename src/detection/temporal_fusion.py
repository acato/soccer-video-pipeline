"""Temporal fusion: shot_on_target → goal promotion via dead-time pattern.

Complements audio_fusion. Audio fusion catches ~50% of GT goals (the loud ones).
Temporal fusion targets the silent goals: a real goal is followed by an extended
celebration pause where the VLM detects nothing — no catch, no distribution, no
shot, no restart. A saved shot in contrast is followed by GK distribution
activity within ~10s.

Heuristic (intentionally conservative — favor precision):
  For each shot_on_target NOT already covered by a goal:
    Within the lookahead window after the shot, count VLM events of any kind.
    If the window is empty for at least min_dead_time_sec, promote.

The dead-time threshold (default 25s) is chosen so that:
  - Real goals (celebration ~10-30s + kickoff setup) typically clear the bar.
  - Saved shots (GK distribution within 5-10s) typically fail it.
  - Missed-and-out-of-bounds shots (goal_kick within 10-20s) typically fail it
    when the goal_kick is detected; they leak through when it isn't.
"""
from __future__ import annotations

from typing import Optional

import structlog

log = structlog.get_logger(__name__)


# Event types that count as "activity" in the post-shot window. Any of these
# firing within the lookahead cancels the goal hypothesis.
_ACTIVITY_TYPES = frozenset({
    "shot_on_target", "shot_off_target", "goal", "near_miss",
    "shot_stop_diving", "shot_stop_standing", "catch", "punch",
    "throw_in", "goal_kick", "free_kick_shot", "corner_kick",
    "distribution_short", "distribution_long", "kickoff", "set_piece",
})


def apply_temporal_fusion(
    events: list,
    min_dead_time_sec: float = 25.0,
    lookahead_sec: float = 45.0,
    promoted_confidence: float = 0.55,
    job_id: Optional[str] = None,
) -> tuple[list, dict]:
    """Promote shot_on_target → goal when followed by extended dead time.

    Args:
        events: detector output (list[Event]). Modified in place for goal
            interval tracking; returned list contains originals + new goals.
        min_dead_time_sec: required gap (no VLM events) after the shot.
        lookahead_sec: post-shot window we scan for activity.
        promoted_confidence: confidence assigned to promoted goals. Tuned below
            audio fusion's 0.5-0.9 since the temporal signal is noisier.
        job_id: log correlation.

    Returns:
        (events + new_goals, stats_dict)
    """
    from src.detection.models import Event, EventType  # local import to avoid cycle

    stats = {
        "shots_examined": 0,
        "shots_promoted": 0,
        "promotion_blocked_existing_goal": 0,
        "promotion_blocked_activity": 0,
    }

    def _etype(e):
        return e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type)

    # Coverage intervals from existing goals (incl. any audio-promoted ones).
    existing_goal_intervals = [
        (e.timestamp_start - 5.0, e.timestamp_end + 25.0)
        for e in events if _etype(e) == "goal"
    ]

    events_sorted = sorted(events, key=lambda e: e.timestamp_start)

    new_goals = []
    for e in events:
        if _etype(e) != "shot_on_target":
            continue
        stats["shots_examined"] += 1

        if any(s <= e.timestamp_start <= en for s, en in existing_goal_intervals):
            stats["promotion_blocked_existing_goal"] += 1
            continue

        win_start = e.timestamp_end
        win_end = e.timestamp_start + lookahead_sec
        if win_end - win_start < min_dead_time_sec:
            continue

        has_activity = False
        for other in events_sorted:
            if other is e:
                continue
            if _etype(other) not in _ACTIVITY_TYPES:
                continue
            if other.timestamp_start >= win_end:
                break
            if other.timestamp_start > win_start:
                has_activity = True
                break

        if has_activity:
            stats["promotion_blocked_activity"] += 1
            continue

        new_goal = Event(
            job_id=e.job_id,
            source_file=e.source_file,
            event_type=EventType.GOAL,
            timestamp_start=e.timestamp_start,
            timestamp_end=max(e.timestamp_end, e.timestamp_start + 15.0),
            confidence=promoted_confidence,
            reel_targets=list(e.reel_targets),
            frame_start=e.frame_start,
            frame_end=e.frame_end,
            metadata={
                "promoted_from_shot": True,
                "source_event_type": "shot_on_target",
                "promotion_method": "temporal_dead_time",
                "min_dead_time_sec": min_dead_time_sec,
                "lookahead_sec": lookahead_sec,
                "vlm_reasoning": (
                    f"Temporal fusion: shot_on_target at {e.timestamp_start:.1f}s "
                    f"with no VLM activity in [{win_start:.1f}, {win_end:.1f}]s "
                    f"({win_end - win_start:.1f}s of dead time) — matches post-goal "
                    f"celebration pattern."
                ),
            },
        )
        new_goals.append(new_goal)
        existing_goal_intervals.append(
            (new_goal.timestamp_start - 5.0, new_goal.timestamp_end + 25.0)
        )
        stats["shots_promoted"] += 1

    log.info("temporal_fusion.summary", job_id=job_id, **stats,
             total_events_in=len(events), new_goals=len(new_goals))

    return events + new_goals, stats
