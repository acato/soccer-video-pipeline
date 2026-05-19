"""Re-aggregate ensemble VLM labels under a relaxed kickoff_restart rule.

The current rule requires either celebration, goal+reset, kickoff_restart
preceded by another label, or ≥2 kickoff_restart. But on game_22 GT 2390
two candidates had a single kickoff_restart at -60 (boundary frame), with
subsequent active_play. The strict rule dropped them.

Try a looser rule: ANY kickoff_restart label, provided subsequent labels
include active_play/idle (consistent with kickoff → play resumption).

Re-scores existing formation_v2_base outputs against GT, reporting TP/FP
deltas vs the strict rule.
"""
import json
from pathlib import Path

GAMES = {
    "game_22": {
        "path": "/tmp/kickoff_game_22_formation_v2_base.jsonl",
        # offset_1H=195, halftime=690 (from prior calibration)
        "gt_video_1h": [g + 195 for g in [1559.7, 2195.4]],
        "gt_video_2h": [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]],
        "h1_dur": 2700,
    },
    "game_21": {
        "path": "/tmp/kickoff_game_21_formation_v2_base.jsonl",
        "gt_video_1h": [g + 65 for g in [1578.6, 2008.6]],
        "gt_video_2h": [],
        "h1_dur": 2700,
    },
    "rush": {
        "path": "/tmp/kickoff_rush_formation_v2_base.jsonl",
        "gt_video_1h": [g + 418 for g in [383.1, 647.0]],
        "gt_video_2h": [g + 418 + 770 for g in [3038.4, 3263.5]],
        "h1_dur": 2700,
    },
    "game_20_1H": {
        "path": "/tmp/kickoff_game20_1H_formation_base.jsonl",
        "gt_video_1h": [g + 189 for g in [1072.2, 1137.0, 1639.2, 2314.2]],
        # 1H scan covers 0-3550, so include early 2H goals that landed in scan
        "gt_video_2h": [g + 189 + 100 for g in [2442.0, 3106.4]],
        "h1_dur": 2400,
    },
    "game_20_2H": {
        "path": "/tmp/kickoff_game20_2H_formation_base.jsonl",
        # 2H scan covers 3554-7229 → late 2H GT goals
        "gt_video_1h": [],
        "gt_video_2h": [g + 189 + 100 for g in [3596.0, 3976.8, 4065.2]],
        "h1_dur": 2400,
    },
}

TOL = 90.0


def aggregate_strict(labels):
    """Current v3 rule."""
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
            for _, before in labs[:i]:
                if before in ("goal", "celebration", "set_piece"):
                    return "GOAL"
    kr = sum(1 for _, l in labs if l == "kickoff_restart")
    if kr >= 2:
        return "GOAL"
    return "NO"


def aggregate_relaxed(labels):
    """Relaxed: any kickoff_restart + at least one subsequent active_play/idle."""
    labs = sorted(labels, key=lambda x: x[0])
    if any(l == "celebration" for _, l in labs):
        return "GOAL"
    for i, (_, l) in enumerate(labs):
        if l == "goal":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL"
    # Any kickoff_restart followed by an active_play OR idle later
    for i, (_, l) in enumerate(labs):
        if l == "kickoff_restart":
            # Allow it if there's a subsequent active_play/idle (play resumption)
            # OR if preceded by goal/celebration/set_piece (original rule)
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL"
            for _, before in labs[:i]:
                if before in ("goal", "celebration", "set_piece"):
                    return "GOAL"
    return "NO"


def score(dets, gt_times, tol=TOL):
    used = set()
    tp = 0
    for d in sorted(dets, key=lambda x: x["start_sec"]):
        best_i, best_dt = None, float("inf")
        for i, g in enumerate(gt_times):
            if i in used:
                continue
            dt = abs(d["start_sec"] - g)
            if dt <= tol and dt < best_dt:
                best_i, best_dt = i, dt
        if best_i is not None:
            used.add(best_i)
            tp += 1
    return tp, len(dets) - tp, len(gt_times) - tp


def main():
    print(f"{'game':<13} {'rule':<8} {'kept':>5} {'TP':>3} {'FP':>3} {'FN':>3}")
    totals = {"strict": [0, 0, 0], "relaxed": [0, 0, 0]}
    for game, cfg in GAMES.items():
        if not Path(cfg["path"]).exists():
            continue
        rows = [json.loads(l) for l in Path(cfg["path"]).read_text().splitlines() if l.strip()]
        gt = cfg["gt_video_1h"] + cfg["gt_video_2h"]
        for rule_name, fn in (("strict", aggregate_strict), ("relaxed", aggregate_relaxed)):
            dets = [r for r in rows if fn(r.get("_vlm_labels", [])) == "GOAL"]
            tp, fp, fn_ = score(dets, gt)
            print(f"{game:<13} {rule_name:<8} {len(dets):>5} {tp:>3} {fp:>3} {fn_:>3}")
            totals[rule_name][0] += tp
            totals[rule_name][1] += fp
            totals[rule_name][2] += fn_
        print()
    print("=" * 50)
    for rule in ("strict", "relaxed"):
        tp, fp, fn_ = totals[rule]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn_)
        print(f"AGG {rule:<8}   TP={tp:>2} FP={fp:>3} FN={fn_:>2}  prec={prec:.2f}  rec={rec:.2f}")


if __name__ == "__main__":
    main()
