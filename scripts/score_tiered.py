"""Score tiered event files per tier (confirmed vs candidate) per game."""
import json
from pathlib import Path

TOL = 90.0

GAMES = {
    "game_20": ("/tmp/kickoff_game_20_tiered_relaxed.jsonl",
                [g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
                [g + 775 for g in [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]]),
    "game_22": ("/tmp/kickoff_game_22_tiered_relaxed.jsonl",
                [g + 195 for g in [1559.7, 2195.4]] +
                [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]]),
    "game_21": ("/tmp/kickoff_game_21_tiered_relaxed.jsonl",
                [g + 65 for g in [1578.6, 2008.6]]),
    "rush": ("/tmp/kickoff_rush_tiered_relaxed.jsonl",
             [g + 418 for g in [383.1, 647.0]] +
             [g + 418 + 770 for g in [3038.4, 3263.5]]),
}


def score(dets, gts, tol=TOL):
    used = set()
    tp = 0
    for d in sorted(dets):
        for i, g in enumerate(gts):
            if i in used: continue
            if abs(d - g) <= tol:
                used.add(i); tp += 1; break
    return tp, len(dets) - tp, len(gts) - tp


def _t(e):
    return e.get("start_sec", e.get("timestamp_start"))


print(f"{'game':<10} {'tier':<12} {'#':>3} {'TP':>3} {'FP':>3} {'FN':>3}")
print("-" * 50)
totals = {"confirmed": [0, 0, 0], "candidate": [0, 0, 0], "either": [0, 0, 0]}
for game, (path, gts) in GAMES.items():
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    goals = [r for r in rows if r.get("event_type") == "goal"]
    confirmed = [_t(r) for r in goals if r.get("metadata", {}).get("goal_tier") == "confirmed"]
    # filter confirmed to only those that aren't GT-import artifacts; we accept all here
    candidate = [_t(r) for r in goals if r.get("metadata", {}).get("goal_tier") == "candidate"]

    for tier_name, dets in (("confirmed", confirmed),
                            ("candidate", candidate),
                            ("either (union)", sorted(set(confirmed) | set(candidate)))):
        tp, fp, fn = score(dets, gts)
        print(f"{game:<10} {tier_name:<12} {len(dets):>3} {tp:>3} {fp:>3} {fn:>3}")
        key = tier_name.split()[0]  # "confirmed", "candidate", "either"
        if key in totals:
            totals[key][0] += tp
            totals[key][1] += fp
            totals[key][2] += fn
    print()

print("=" * 50)
print(f"{'AGG':<10} {'tier':<12} {'':>3} {'TP':>3} {'FP':>3} {'FN':>3} {'recall':>6} {'prec':>6}")
for k in ("confirmed", "candidate", "either"):
    tp, fp, fn = totals[k]
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    print(f"{'AGG':<10} {k:<12} {'':>3} {tp:>3} {fp:>3} {fn:>3} {rec:>6.2f} {prec:>6.2f}")
