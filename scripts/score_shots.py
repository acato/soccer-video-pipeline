"""Score shot detection against GT (Shots & Goals events).

GT includes On Target, Off Target, Blocked, Goals, and a few odd outcomes.
For a highlights reel ("FP-tolerant, all shots") we want recall over all.
"""
import json
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, "scripts")
from score_gk_actions import GAMES, _t, is_real, find_gt_files


def load_gt_shots(gt_dir, off_1h, off_2h):
    """Return list of (video_time, outcome) for every Shots & Goals event."""
    out = []
    h1_path, h2_path = find_gt_files(gt_dir)
    for path, off in [(h1_path, off_1h), (h2_path, off_2h)]:
        if path is None: continue
        data = json.loads(path.read_text())
        for r in data["data"]:
            for e in r["events"]:
                if e["event_name"] == "Shots & Goals":
                    outcome = e.get("property", {}).get("Outcome", "")
                    out.append((r["event_time"] / 1000.0 + off, outcome))
    return sorted(out)


def coverage(dets, gts, tol):
    covered = set()
    used = set()
    for i, d in enumerate(dets):
        for j, (g, *_) in enumerate(gts):
            if abs(d - g) <= tol:
                covered.add(j)
                used.add(i)
    tp = len(covered)
    return tp, len(dets) - len(used), len(gts) - tp


def load_tiered_shots(path):
    """Read shot_tier-tagged events from a tiered events file.
    Returns (confirmed_times, candidate_times)."""
    confirmed, candidate = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        e = json.loads(line)
        tier = e.get("metadata", {}).get("shot_tier")
        if tier == "confirmed":
            confirmed.append(_t(e))
        elif tier == "candidate":
            candidate.append(_t(e))
    return sorted(confirmed), sorted(candidate)


def main():
    print(f"\n{'game':<10} {'tol':>4} {'tier':<11} {'#det':>5} {'#gt':>4} {'TP':>3} {'FP':>4} {'FN':>3} {'rec':>5} {'prec':>5}")
    print("-" * 80)
    aggs = {tol: {"confirmed":[0,0,0], "candidate":[0,0,0], "union":[0,0,0]}
            for tol in (30, 60, 90)}
    by_outcome = {tol: Counter() for tol in (30, 60, 90)}
    by_outcome_fn = {tol: Counter() for tol in (30, 60, 90)}
    for game, cfg in GAMES.items():
        gts = load_gt_shots(cfg["gt"], cfg["off_1h"], cfg["off_2h"])
        tiered_path = f"/tmp/kickoff_{game}_tiered_events.jsonl"
        confirmed, candidate = load_tiered_shots(tiered_path)
        union = sorted(set(confirmed) | set(candidate))
        for tol in (30, 60, 90):
            for name, dets in (("confirmed", confirmed),
                               ("candidate", candidate),
                               ("union",     union)):
                tp, fp, fn = coverage(dets, gts, tol)
                rec = tp / max(1, tp + fn)
                prec = tp / max(1, tp + fp)
                print(f"{game:<10} {tol:>3}s {name:<11} {len(dets):>5} {len(gts):>4} "
                      f"{tp:>3} {fp:>4} {fn:>3} {rec:>5.2f} {prec:>5.2f}")
                aggs[tol][name][0] += tp
                aggs[tol][name][1] += fp
                aggs[tol][name][2] += fn
            # Per-outcome breakdown using union dets
            for j, (g, outcome) in enumerate(gts):
                if any(abs(d - g) <= tol for d in union):
                    by_outcome[tol][outcome] += 1
                else:
                    by_outcome_fn[tol][outcome] += 1
        print()
    print("=" * 80)
    for tol in (30, 60, 90):
        for name in ("confirmed", "candidate", "union"):
            tp, fp, fn = aggs[tol][name]
            rec = tp / max(1, tp + fn)
            prec = tp / max(1, tp + fp)
            print(f"AGG        {tol:>3}s {name:<11} {'':>5} {'':>4} "
                  f"{tp:>3} {fp:>4} {fn:>3} {rec:>5.2f} {prec:>5.2f}")
        print()
    print("Per-outcome coverage at ±60s (union):")
    outcomes = sorted(by_outcome[60].keys() | by_outcome_fn[60].keys())
    for o in outcomes:
        cov = by_outcome[60].get(o, 0)
        miss = by_outcome_fn[60].get(o, 0)
        total = cov + miss
        rec = cov / total if total else 0
        print(f"  {o!r:<22} covered {cov:>3} / {total:>3} ({rec:.2f})")


if __name__ == "__main__":
    main()
