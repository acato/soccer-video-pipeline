"""Offline temporal-fusion diagnostic: measure ceiling of shot→goal promotion.

The pipeline ships temporal_fusion.py but it's currently disabled in eval
(multi_game_eval.sh sets temporal_fusion_enabled: False). This script
re-applies the logic offline against existing dual_pass_events.jsonl files
and reports how many new goal TPs would have been recovered, without
needing a fresh multi-game eval (~6h saved).

Use case: validate the "post-goal dead-time" hypothesis cheaply. If
temporal_fusion would recover several goals on existing data, re-enable
it. If it would recover ~0 (because the upstream shot_on_target events are
in the wrong places), we know the fix requires a kickoff detector (which
runs independently of shot calls), not just enabling temporal_fusion.

Usage:
    python scripts/diag_temporal_fusion.py \\
        --events /tmp/soccer-pipeline/<job>/diagnostics/dual_pass_events.jsonl \\
        --out /tmp/fused_events.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Mirror temporal_fusion.py's _ACTIVITY_TYPES
ACTIVITY_TYPES = frozenset({
    "shot_on_target", "shot_off_target", "goal", "near_miss",
    "shot_stop_diving", "shot_stop_standing", "catch", "punch",
    "throw_in", "goal_kick", "free_kick_shot", "corner_kick",
    "distribution_short", "distribution_long", "kickoff", "set_piece",
})


def apply_temporal_fusion_to_jsonl(
    events: list[dict],
    min_dead_time_sec: float = 25.0,
    lookahead_sec: float = 45.0,
    promoted_confidence: float = 0.55,
) -> tuple[list[dict], dict]:
    """Pure-dict reimplementation of src.detection.temporal_fusion.

    Operates on the dual_pass_events.jsonl schema:
      {event_type, start_sec, end_sec, confidence, reasoning, triage_labels}
    """
    stats = {
        "shots_examined": 0,
        "shots_promoted": 0,
        "promotion_blocked_existing_goal": 0,
        "promotion_blocked_activity": 0,
    }

    existing_goal_intervals = [
        (e["start_sec"] - 5.0, e["end_sec"] + 25.0)
        for e in events if e.get("event_type") == "goal"
    ]
    events_sorted = sorted(events, key=lambda e: e["start_sec"])

    new_goals = []
    for e in events:
        if e.get("event_type") != "shot_on_target":
            continue
        stats["shots_examined"] += 1

        if any(s <= e["start_sec"] <= en for s, en in existing_goal_intervals):
            stats["promotion_blocked_existing_goal"] += 1
            continue

        win_start = e["end_sec"]
        win_end = e["start_sec"] + lookahead_sec
        if win_end - win_start < min_dead_time_sec:
            continue

        has_activity = False
        for other in events_sorted:
            if other is e:
                continue
            if other.get("event_type") not in ACTIVITY_TYPES:
                continue
            if other["start_sec"] >= win_end:
                break
            if other["start_sec"] > win_start:
                has_activity = True
                break

        if has_activity:
            stats["promotion_blocked_activity"] += 1
            continue

        new_goal = {
            "event_type": "goal",
            "start_sec": e["start_sec"],
            "end_sec": max(e["end_sec"], e["start_sec"] + 15.0),
            "confidence": promoted_confidence,
            "reasoning": (
                f"temporal_fusion: shot_on_target at {e['start_sec']:.1f}s with "
                f"{win_end - win_start:.1f}s of dead time (promoted from shot)"
            ),
            "triage_labels": ["temporal_fusion"],
            "_promoted_from_shot": True,
        }
        new_goals.append(new_goal)
        existing_goal_intervals.append(
            (new_goal["start_sec"] - 5.0, new_goal["end_sec"] + 25.0)
        )
        stats["shots_promoted"] += 1

    out = events + new_goals
    out.sort(key=lambda e: e["start_sec"])
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-dead-time", type=float, default=25.0)
    ap.add_argument("--lookahead", type=float, default=45.0)
    args = ap.parse_args()

    events = []
    with open(args.events) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    fused, stats = apply_temporal_fusion_to_jsonl(
        events,
        min_dead_time_sec=args.min_dead_time,
        lookahead_sec=args.lookahead,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for e in fused:
            f.write(json.dumps(e) + "\n")

    print(f"input: {len(events)} events from {args.events}", file=sys.stderr)
    print(f"output: {len(fused)} events to {args.out}", file=sys.stderr)
    print(f"stats:", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k}: {v}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
