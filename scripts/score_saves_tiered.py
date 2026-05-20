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
    """Returns (confirmed_times, candidate_times, inferred_times)."""
    confirmed, candidate, inferred = [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        tier = e.get("metadata", {}).get("save_tier")
        if tier == "confirmed":
            confirmed.append(_t(e))
        elif tier == "candidate":
            candidate.append(_t(e))
        elif tier == "inferred":
            inferred.append(_t(e))
    return sorted(confirmed), sorted(candidate), sorted(inferred)


def coverage_score(dets, gts, tol):
    """Unique-coverage metric: a GT is covered if ANY det is within ±tol.
    A det is 'used' (TP-contributing) if it covers at least one uncovered GT.
    This is the right metric for reel construction — overlapping detections
    each generate clips, and any covering clip is a TP."""
    covered_gts = set()
    used_dets = set()
    for i, d in enumerate(dets):
        for j, (g, *_) in enumerate(gts):
            if abs(d - g) <= tol:
                covered_gts.add(j)
                used_dets.add(i)
    tp = len(covered_gts)
    fp = len(dets) - len(used_dets)
    fn = len(gts) - tp
    return tp, fp, fn


def main():
    print(f"{'game':<10} {'tol':>4} {'tier':<11} {'#det':>4} {'#gt':>3} "
          f"{'TP':>3} {'FP':>3} {'FN':>3} {'rec':>5} {'prec':>5}")
    print("-- coverage metric (any-det-in-window per GT) " + "-" * 30)
    aggs = {tol: {"confirmed": [0,0,0], "candidate": [0,0,0],
                  "inferred": [0,0,0], "union": [0,0,0]}
            for tol in (30, 60, 90)}
    for game, cfg in GAMES.items():
        gts = load_gt_saves(cfg["gt"], cfg["off_1h"], cfg["off_2h"])
        tiered_path = f"/tmp/kickoff_{game}_tiered_events.jsonl"
        confirmed, candidate, inferred = load_tiered(tiered_path)
        union = sorted(set(confirmed) | set(candidate) | set(inferred))
        for tol in (30, 60, 90):
            for name, dets in (("confirmed", confirmed),
                               ("candidate", candidate),
                               ("inferred", inferred),
                               ("union",     union)):
                tp, fp, fn = coverage_score(dets, gts, tol)
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
        for name in ("confirmed", "candidate", "inferred", "union"):
            tp, fp, fn = aggs[tol][name]
            rec = tp / max(1, tp + fn)
            prec = tp / max(1, tp + fp)
            print(f"AGG        {tol:>3}s {name:<11} {'':>4} {'':>3} "
                  f"{tp:>3} {fp:>3} {fn:>3} {rec:>5.2f} {prec:>5.2f}")
        print()


if __name__ == "__main__":
    main()
