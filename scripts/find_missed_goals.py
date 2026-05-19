"""Identify which specific GT goals are missed by the relaxed-rule ensemble."""
import json
from pathlib import Path

GAMES = {
    "game_22": {
        "path": "/tmp/kickoff_game_22_formation_v2_base.jsonl",
        "gts": [("1H#1", 1559.7 + 195), ("1H#2", 2195.4 + 195),
                ("2H#1", 3734.7 + 195 + 690), ("2H#2", 4853.3 + 195 + 690),
                ("2H#3", 5066.6 + 195 + 690)],
    },
    "game_21": {
        "path": "/tmp/kickoff_game_21_formation_v2_base.jsonl",
        "gts": [("1H#1", 1578.6 + 65), ("1H#2", 2008.6 + 65)],
    },
    "rush": {
        "path": "/tmp/kickoff_rush_formation_v2_base.jsonl",
        "gts": [("1H#1", 383.1 + 418), ("1H#2", 647.0 + 418),
                ("2H#1", 3038.4 + 418 + 770), ("2H#2", 3263.5 + 418 + 770)],
    },
    # game_20 calibrated empirically: 1H scan shows game ends ~video 2500,
    # halftime from 2500-3175 (~11 min), 2H kickoff at video 3175.
    # offset_1H = 100, halftime = 675. game_clock 0 = video 100; 1H ends
    # game_clock 2400 = video 2500. 2H game_clock 2400 = video 3175.
    "game_20_1H_scan": {
        "path": "/tmp/kickoff_game20_1H_formation_base.jsonl",
        "gts": [("1H#1", 1072.2 + 100), ("1H#2", 1137.0 + 100),
                ("1H#3", 1639.2 + 100), ("1H#4", 2314.2 + 100),
                # 1H scan covers 0-3550, so it includes the 1st 2H goal
                # (at video 3217) but not later 2H goals
                ("2H#1", 2442.0 + 775)],
    },
    "game_20_2H_scan": {
        "path": "/tmp/kickoff_game20_2H_formation_base.jsonl",
        # 2H scan covers 3554-7229. All 2H goals AFTER 3554 land here.
        "gts": [("2H#2", 3106.4 + 775), ("2H#3", 3596.0 + 775),
                ("2H#4", 3976.8 + 775), ("2H#5", 4065.2 + 775)],
    },
}


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


TOL = 90.0

for game, cfg in GAMES.items():
    rows = [json.loads(l) for l in Path(cfg["path"]).read_text().splitlines() if l.strip()]
    goal_dets = sorted(
        r["start_sec"] for r in rows
        if aggregate_relaxed(r.get("_vlm_labels", [])) == "GOAL"
    )

    print(f"\n=== {game} ===")
    for name, gt_t in cfg["gts"]:
        close = [(d, abs(d - gt_t)) for d in goal_dets if abs(d - gt_t) <= TOL]
        if close:
            close.sort(key=lambda x: x[1])
            d, diff = close[0]
            print(f"  {name} GT {gt_t:.0f}: CAUGHT by det {d:.0f} (diff {diff:.0f}s)")
        else:
            # Find nearest candidate ANYWHERE in window even if NO
            nearest_any = sorted(
                (abs(r["start_sec"] - gt_t), r["start_sec"], r)
                for r in rows if abs(r["start_sec"] - gt_t) <= 120
            )
            if nearest_any:
                diff, near_t, near_r = nearest_any[0]
                labels = near_r.get("_vlm_labels", [])
                labels_str = " ".join(f"{o:+d}:{l[:5]}" for o, l in labels)
                print(f"  {name} GT {gt_t:.0f}: MISSED. Nearest candidate t={near_t:.0f} "
                      f"(diff {diff:.0f}s, verdict={aggregate_relaxed(labels)})")
                print(f"    labels: {labels_str}")
            else:
                print(f"  {name} GT {gt_t:.0f}: MISSED. NO candidate within 120s")
