"""Test temporal grouping filters: how many FPs do we cut by enforcing
minimum-time-between-detections rules at 60s, 120s, 180s, 300s windows?
"""
import json
from pathlib import Path
from collections import Counter

TOL = 90.0


def aggregate_relaxed(labels):
    labs = sorted(labels, key=lambda x: x[0])
    if any(l == "celebration" for _, l in labs):
        return "GOAL"
    for i, (_, l) in enumerate(labs):
        if l == "goal":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL"
    for i, (_, l) in enumerate(labs):
        if l == "kickoff_restart":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL"
            for _, before in labs[:i]:
                if before in ("goal", "celebration", "set_piece"):
                    return "GOAL"
    return "NO"


GAMES = {
    "game_22": {
        "formations": ["/tmp/kickoff_game_22_formation_v2_base.jsonl"],
        "pattern": None,
        "gts": sorted([g + 195 for g in [1559.7, 2195.4]] +
                      [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]]),
    },
    "game_21": {
        "formations": ["/tmp/kickoff_game_21_formation_v2_base.jsonl"],
        "pattern": None,
        "gts": sorted([g + 65 for g in [1578.6, 2008.6]]),
    },
    "game_20": {
        "formations": ["/tmp/kickoff_game20_1H_formation_base.jsonl",
                       "/tmp/kickoff_game20_2H_formation_base.jsonl"],
        "pattern": None,
        "gts": sorted([g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
                      [g + 775 for g in [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]]),
    },
    "rush": {
        "formations": ["/tmp/kickoff_rush_formation_v2_base.jsonl"],
        "pattern": "/tmp/kickoff_rush_pattern_v11_0191.jsonl",
        "gts": sorted([g + 418 for g in [383.1, 647.0]] +
                      [g + 418 + 770 for g in [3038.4, 3263.5]]),
    },
}

# Print GT spacing per game to understand minimum natural distance
print("=== GT goal spacing per game ===")
for game, cfg in GAMES.items():
    gts = cfg["gts"]
    if len(gts) < 2:
        continue
    deltas = [gts[i+1] - gts[i] for i in range(len(gts)-1)]
    print(f"  {game}: GT count={len(gts)}, min spacing={min(deltas):.0f}s, "
          f"max={max(deltas):.0f}s, mean={sum(deltas)/len(deltas):.0f}s")
print()


def load_all_goals(cfg):
    out = []
    for fp in cfg["formations"]:
        if not Path(fp).exists():
            continue
        for line in Path(fp).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if aggregate_relaxed(r.get("_vlm_labels", [])) == "GOAL":
                out.append(r)
    if cfg.get("pattern") and Path(cfg["pattern"]).exists():
        for line in Path(cfg["pattern"]).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_vlm_verdict") == "GOAL":
                out.append(r)
    return out


def dedup_with_window(dets, window_sec, score_fn=None):
    """Greedy: sort by time, keep det if no kept det is within window."""
    if score_fn:
        dets = sorted(dets, key=lambda d: (-score_fn(d), d["start_sec"]))
    else:
        dets = sorted(dets, key=lambda d: d["start_sec"])
    kept = []
    for d in dets:
        if any(abs(d["start_sec"] - k["start_sec"]) < window_sec for k in kept):
            continue
        kept.append(d)
    return kept


def gt_level_score(dets, gts, tol=TOL):
    tp = 0
    used = set()
    for g in gts:
        for i, d in enumerate(dets):
            if i in used:
                continue
            if abs(d["start_sec"] - g) <= tol:
                tp += 1
                used.add(i)
                break
    fp = len(dets) - len(used)
    return tp, fp


# Score function: more kickoff_restart labels = stronger candidate
def kr_score(d):
    return sum(1 for _, l in d.get("_vlm_labels", []) if l == "kickoff_restart")


WINDOWS = [60, 90, 120, 180, 240, 300]
print(f"{'window':<12} {'method':<20} {'TP':>3} {'FP':>4} {'recall':>6} {'prec':>6}")
print("-" * 60)
for window in WINDOWS:
    for method, score_fn in (("by-time (earliest)", None),
                              ("by-kr-count (best)", kr_score)):
        total_tp = 0
        total_fp = 0
        total_gts = 0
        for game, cfg in GAMES.items():
            dets = load_all_goals(cfg)
            dets = dedup_with_window(dets, window, score_fn=score_fn)
            tp, fp = gt_level_score(dets, cfg["gts"])
            total_tp += tp
            total_fp += fp
            total_gts += len(cfg["gts"])
        prec = total_tp / max(1, total_tp + total_fp)
        rec = total_tp / max(1, total_gts)
        print(f"{window:<12} {method:<20} {total_tp:>3} {total_fp:>4} {rec:>6.2f} {prec:>6.2f}")
