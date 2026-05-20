"""Score paired-tier saves in tiered events files against GT.

Tiered events are produced by merge_ensemble_into_events.py --save-tiers and
have metadata.save_tier in {confirmed, candidate} on the relevant events.

Confirmed tier = catch + shot_stop_*  (high precision baseline)
Candidate tier = shot_on_target + free_kick_shot (recall booster)
Union          = either tier (for keeper-reel inclusion)
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from score_gk_actions import GAMES, load_gt_saves, score


def _t(e):
    return e.get("start_sec", e.get("timestamp_start"))


def load_tiered(path):
    """Returns (confirmed_times, candidate_times)."""
    confirmed, candidate = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        tier = e.get("metadata", {}).get("save_tier")
        if tier == "confirmed":
            confirmed.append(_t(e))
        elif tier == "candidate":
            candidate.append(_t(e))
    return sorted(confirmed), sorted(candidate)


def main():
    print(f"{'game':<10} {'tol':>4} {'tier':<11} {'#det':>4} {'#gt':>3} "
          f"{'TP':>3} {'FP':>3} {'FN':>3} {'rec':>5} {'prec':>5}")
    print("-" * 75)
    aggs = {tol: {"confirmed": [0,0,0], "candidate": [0,0,0], "union": [0,0,0]}
            for tol in (30, 60, 90)}
    for game, cfg in GAMES.items():
        gts = load_gt_saves(cfg["gt"], cfg["off_1h"], cfg["off_2h"])
        tiered_path = f"/tmp/kickoff_{game}_tiered_events.jsonl"
        confirmed, candidate = load_tiered(tiered_path)
        union = sorted(set(confirmed) | set(candidate))
        for tol in (30, 60, 90):
            for name, dets in (("confirmed", confirmed),
                               ("candidate", candidate),
                               ("union",     union)):
                matched, fps, fns = score(dets, gts, tol)
                tp, fp, fn = len(matched), len(fps), len(fns)
                rec = tp / max(1, tp + fn)
                prec = tp / max(1, tp + fp)
                print(f"{game:<10} {tol:>3}s {name:<11} {len(dets):>4} {len(gts):>3} "
                      f"{tp:>3} {fp:>3} {fn:>3} {rec:>5.2f} {prec:>5.2f}")
                aggs[tol][name][0] += tp
                aggs[tol][name][1] += fp
                aggs[tol][name][2] += fn
            print()
    print("=" * 75)
    for tol in (30, 60, 90):
        for name in ("confirmed", "candidate", "union"):
            tp, fp, fn = aggs[tol][name]
            rec = tp / max(1, tp + fn)
            prec = tp / max(1, tp + fp)
            print(f"AGG        {tol:>3}s {name:<11} {'':>4} {'':>3} "
                  f"{tp:>3} {fp:>3} {fn:>3} {rec:>5.2f} {prec:>5.2f}")
        print()


if __name__ == "__main__":
    main()
