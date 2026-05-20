"""Trace which TPs are lost under expanded negative-evidence filter."""
import json
from pathlib import Path

TOL = 90.0
SAVE_WINDOW = 10.0

NEGATIVE_EVENT_TYPES = {
    "catch", "shot_stop_diving", "shot_stop_standing",
    "throw_in", "corner_kick", "free_kick_shot",
}

GAMES = {
    "game_22": (
        "/tmp/kickoff_game_22_tiered_relaxed.jsonl",
        "/tmp/soccer-pipeline/031ee71a-7a4f-4605-a642-9bff003e4804/events.jsonl",
        [g + 195 for g in [1559.7, 2195.4]] +
        [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]],
    ),
    "game_21": (
        "/tmp/kickoff_game_21_tiered_relaxed.jsonl",
        "/tmp/soccer-pipeline/60ed91b3-ebcb-4ddb-ac18-768923a17419/events.jsonl",
        [g + 65 for g in [1578.6, 2008.6]],
    ),
}


def _t(e):
    return e.get("start_sec", e.get("timestamp_start"))


def is_real(e):
    return e.get("metadata", {}).get("detection_method", "") in {"dual_pass", "shot_outcome"}


for game, (tiered_path, events_path, gts) in GAMES.items():
    print(f"\n=== {game} ===")
    tiered = [json.loads(l) for l in Path(tiered_path).read_text().splitlines() if l.strip()]
    candidate_goals = [r for r in tiered
                       if r.get("event_type") == "goal"
                       and r.get("metadata", {}).get("goal_tier") == "candidate"]

    all_events = [json.loads(l) for l in Path(events_path).read_text().splitlines() if l.strip()]
    negs = sorted([(_t(e), e.get("event_type"))
                   for e in all_events
                   if e.get("event_type") in NEGATIVE_EVENT_TYPES and is_real(e)])

    for c in candidate_goals:
        t = _t(c)
        nearest_gt = min(((abs(t - g), g) for g in gts), default=(99999, None))
        is_tp = nearest_gt[0] <= TOL
        nearest = min((abs(t - st), st, kind) for st, kind in negs) if negs else (99999, None, None)
        blocked = nearest[0] <= SAVE_WINDOW
        if is_tp:
            label = "TP-LOST" if blocked else "TP-kept"
            print(f"  {label}  cand t={t:.0f}  GT={nearest_gt[1]} (diff {nearest_gt[0]:.0f}s)  "
                  f"nearest_neg={nearest[2]} at {nearest[1]} (diff {nearest[0]:.0f}s)")
