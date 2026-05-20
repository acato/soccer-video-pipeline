"""Negative-evidence filter for candidate goals.

Logic: if a dual_pass detected event implies no goal at time T, drop
any candidate goal in T's exclusion window.

Strong negative signals (drop candidate within ±30s):
  catch              — keeper caught the ball, shot was saved
  shot_stop_diving   — diving save (parry), shot was stopped
  shot_stop_standing — standing save, shot was stopped

Directional negative signals:
  goal_kick — keeper's restart after the ball went out behind goal line
              WITHOUT scoring. The shot that led to it was in the seconds
              before. Drop candidates in [goal_kick - 30s, goal_kick - 0s].
  throw_in — ball out across sideline; can't be a goal. But too common
             to be useful; skip.

Ambiguous (NOT a disqualifier on its own):
  corner_kick — can lead to a goal 5-15s later (header from corner)
  free_kick_shot — could be a direct-goal kick
  shot_on_target — shot may or may not have scored
"""
import json
from pathlib import Path

# ±10s for catch / shot_stop events. Tighter window required because rebound
# goals exist: keeper parries shot at T, ball deflected in at T+10-15s. A 30s
# window would block the rebound goal as well as the save itself.
SAVE_WINDOW = 10.0

# Goal kick = no goal in the [-30, -0] preceding window
GOAL_KICK_PRE_WINDOW = (30.0, 0.0)

# Strong negative: the moment of these events cannot also be a goal.
# Excluded: free_kick_shot — v11 LoRA sometimes labels the kickoff restart
# itself as a free kick, blocking real-goal candidates 5-10s after the goal.
NEGATIVE_EVENT_TYPES = {
    "catch", "shot_stop_diving", "shot_stop_standing",
    "throw_in", "corner_kick",
}

TOL = 90.0

GAMES = {
    "game_20": (
        "/tmp/kickoff_game_20_tiered_relaxed.jsonl",
        "/tmp/soccer-pipeline/a0f8f93c-8611-466b-934c-8cd48a2aee00/events.jsonl",
        [g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
        [g + 775 for g in [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]],
    ),
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
    "rush": (
        "/tmp/kickoff_rush_tiered_relaxed.jsonl",
        "/tmp/soccer-pipeline/47d1358c-268c-4577-b4e1-f9943f51be6a/events.jsonl",
        [g + 418 for g in [383.1, 647.0]] +
        [g + 418 + 770 for g in [3038.4, 3263.5]],
    ),
}


def _t(e):
    return e.get("start_sec", e.get("timestamp_start"))


def load_events(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def is_real_detection(ev):
    """True if event has a detector source (not a GT label imported for eval)."""
    md = ev.get("metadata", {})
    method = md.get("detection_method", "")
    # The dual_pass detector tags with detection_method "dual_pass" or
    # "shot_outcome". GT-only events have detection_method missing.
    return method in {"dual_pass", "shot_outcome"}


def candidate_blocked(cand_t, negs, goal_kicks):
    """Returns (blocked: bool, reason: str)."""
    for nt in negs:
        if abs(cand_t - nt) <= SAVE_WINDOW:
            return True, f"save within ±{SAVE_WINDOW:.0f}s (at {nt:.0f})"
    for gkt in goal_kicks:
        # Reject if candidate is BEFORE the goal kick by up to 30s
        # (shot before that became the goal kick)
        if 0 <= (gkt - cand_t) <= GOAL_KICK_PRE_WINDOW[0]:
            return True, f"goal_kick {gkt:.0f}s (this is the post-shot restart)"
    return False, ""


def score(dets, gts, tol=TOL):
    used = set()
    tp = 0
    for d in sorted(dets):
        for i, g in enumerate(gts):
            if i in used: continue
            if abs(d - g) <= tol:
                used.add(i); tp += 1; break
    return tp, len(dets) - tp, len(gts) - tp


print(f"{'game':<10} {'mode':<20} {'#cand':>5} {'TP':>3} {'FP':>3} {'FN':>3}")
print("-" * 65)
totals = {"before": [0, 0, 0], "neg_save": [0, 0, 0], "neg_save_gk": [0, 0, 0]}

for game, (tiered_path, events_path, gts) in GAMES.items():
    rows = load_events(tiered_path)
    candidate_goals = [r for r in rows
                       if r.get("event_type") == "goal"
                       and r.get("metadata", {}).get("goal_tier") == "candidate"]

    all_events = load_events(events_path)
    # Extract save events and goal_kick times — but only REAL detector events
    # (skip GT-label imports). dual_pass events.jsonl mixes both.
    save_times = [_t(e) for e in all_events
                  if e.get("event_type") in NEGATIVE_EVENT_TYPES
                  and is_real_detection(e)]
    goal_kick_times = [_t(e) for e in all_events
                       if e.get("event_type") == "goal_kick"
                       and is_real_detection(e)]

    # Mode: before (all candidates)
    cand_t = [_t(c) for c in candidate_goals]
    tp, fp, fn = score(cand_t, gts)
    totals["before"][0] += tp; totals["before"][1] += fp; totals["before"][2] += fn
    print(f"{game:<10} {'before filter':<20} {len(cand_t):>5} {tp:>3} {fp:>3} {fn:>3}")

    # Mode: filter on saves only
    kept_save = [t for t in cand_t
                 if not any(abs(t - st) <= SAVE_WINDOW for st in save_times)]
    tp, fp, fn = score(kept_save, gts)
    totals["neg_save"][0] += tp; totals["neg_save"][1] += fp; totals["neg_save"][2] += fn
    print(f"{game:<10} {'-save (±30s)':<20} {len(kept_save):>5} {tp:>3} {fp:>3} {fn:>3}")

    # Mode: filter on saves + goal_kicks
    kept_both = []
    for t in cand_t:
        blocked, _ = candidate_blocked(t, save_times, goal_kick_times)
        if not blocked:
            kept_both.append(t)
    tp, fp, fn = score(kept_both, gts)
    totals["neg_save_gk"][0] += tp; totals["neg_save_gk"][1] += fp; totals["neg_save_gk"][2] += fn
    print(f"{game:<10} {'-save-goalkick':<20} {len(kept_both):>5} {tp:>3} {fp:>3} {fn:>3}")

    # Info: how many saves and goal_kicks detected
    print(f"           ({len(save_times)} saves, {len(goal_kick_times)} goal_kicks "
          f"in events)")
    print()

print("=" * 65)
print(f"{'AGG':<10} {'mode':<20} {'':>5} {'TP':>3} {'FP':>3} {'FN':>3} {'recall':>6} {'prec':>6}")
for mode in ("before", "neg_save", "neg_save_gk"):
    tp, fp, fn = totals[mode]
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    print(f"{'AGG':<10} {mode:<20} {'':>5} {tp:>3} {fp:>3} {fn:>3} {rec:>6.2f} {prec:>6.2f}")
