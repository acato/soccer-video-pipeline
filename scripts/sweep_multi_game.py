"""Multi-game kickoff parameter sweep on cached YOLO data.

Adds MAX_KICKOFF_RUN_LENGTH filter (rejects detections where the
kickoff_setup run lasts more than N frames — those are continuous play,
not real restarts).

Uses video-time GT mapping with per-game offsets fitted from goals.
tol=90s matches sweep_kickoffs.py convention (accounts for halftime
uncertainty + analytics-vs-video timing jitter).

Usage:
    python scripts/sweep_multi_game.py --top 25 --tol 90
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_kickoffs as dk  # type: ignore  # noqa: E402

# Per-game video offsets in seconds (video_time = game_clock_sec + offset).
# offset_1h: fitted from goal-event matches.
# halftime: assumed 200s for offset_2h = offset_1h + halftime.
GAMES = {
    "game_20_1H_scan": {
        "frames": "/tmp/kickoff_game20_1H_frames.jsonl",
        "offset_1h": 124, "halftime": 200,
        "gt_1h_game": [1072.2, 1137.0, 1639.2, 2314.2],
        # The "1H" scan also overruns into early 2H (scan ended 3550s,
        # 2H starts ~video 2524+halftime=2724). Include early 2H GT goals
        # that fall inside the scan range.
        "gt_2h_game": [2442.0, 3106.4],   # GT 2H goals < video 3550 with hf=200
    },
    "game_20_2H_scan": {
        "frames": "/tmp/kickoff_game20_2H_frames.jsonl",
        "offset_1h": 124, "halftime": 200,
        "gt_1h_game": [],
        "gt_2h_game": [3596.0, 3976.8, 4065.2],   # 2H goals > video 3554
    },
    "game_22": {
        "frames": "/tmp/kickoff_game_22_frames.jsonl",
        "offset_1h": 460, "halftime": 200,
        "gt_1h_game": [1559.7, 2195.4],
        "gt_2h_game": [3734.7, 4853.3, 5066.6],
    },
    "game_21": {
        "frames": "/tmp/kickoff_game_21_frames.jsonl",
        "offset_1h": 24, "halftime": 200,
        "gt_1h_game": [1578.6, 2008.6],
        "gt_2h_game": [],  # GT 2H has 0 Goals Conceded
    },
    "rush": {
        "frames": "/tmp/kickoff_rush_frames.jsonl",
        "offset_1h": 418, "halftime": 200,
        "gt_1h_game": [383.1, 647.0],
        "gt_2h_game": [3038.4, 3263.5],
    },
}


def gt_video_for(cfg):
    """Map game-clock GT to video-time using offsets."""
    o1 = cfg["offset_1h"]
    o2 = o1 + cfg["halftime"]
    return ([g + o1 for g in cfg["gt_1h_game"]]
            + [g + o2 for g in cfg["gt_2h_game"]])


def load_raw(path):
    return [
        {
            "ball": r.get("ball"),
            "p_left": r.get("p_left", 0),
            "p_right": r.get("p_right", 0),
            "total_field": r.get("total_field", 0),
            "in_circle": r.get("in_circle", 0),
            "t": r["t"],
        }
        for r in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())
    ]


def run_cfg(raw, cfg, max_run_length=None):
    """Apply cfg to detector globals, run, filter by max kickoff_setup run length."""
    saved = {k: getattr(dk, k) for k in cfg}
    for k, v in cfg.items():
        setattr(dk, k, v)
    try:
        flags = dk.derive_flags(raw)
        ts = [r["t"] for r in raw]
        goals = dk.detect_goals(flags, ts, 5.0)
    finally:
        for k, v in saved.items():
            setattr(dk, k, v)
    if max_run_length is not None:
        # Drop goals whose kickoff_setup run lasted > max_run_length seconds
        goals = [g for g in goals
                 if (g["_kickoff_end"] - g["_kickoff_start"]) <= max_run_length]
    return goals


def score(detected, gt_video, tol):
    used = set()
    tp = 0
    for d in detected:
        best_i, best_dt = None, float("inf")
        for i, g in enumerate(gt_video):
            if i in used:
                continue
            dt = abs(d["start_sec"] - g)
            if dt <= tol and dt < best_dt:
                best_i, best_dt = i, dt
        if best_i is not None:
            used.add(best_i)
            tp += 1
    fp = len(detected) - tp
    fn = len(gt_video) - tp
    return tp, fp, fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--tol", type=float, default=90.0)
    args = p.parse_args()

    raw_cache = {g: load_raw(c["frames"]) for g, c in GAMES.items()
                 if Path(c["frames"]).exists()}
    gt_cache = {g: gt_video_for(c) for g, c in GAMES.items()
                if g in raw_cache}

    # Baseline (current code, no max_run filter)
    print(f"=== BASELINE (current thresholds, tol={args.tol}s, no run-length cap) ===")
    bl_total = (0, 0, 0)
    for g, raw in raw_cache.items():
        det = run_cfg(raw, {})
        tp, fp, fn = score(det, gt_cache[g], args.tol)
        bl_total = (bl_total[0]+tp, bl_total[1]+fp, bl_total[2]+fn)
        print(f"  {g}: TP={tp} FP={fp:>2} FN={fn}  (n_det={len(det)}, n_gt={len(gt_cache[g])})")
    tp, fp, fn = bl_total
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    print(f"  TOTAL: TP={tp} FP={fp} FN={fn}  prec={prec:.2f} rec={rec:.2f}\n")

    # Sweep axes — aggressive FP reduction; accept some recall loss
    center_xs = [(0.40, 0.60), (0.42, 0.58), (0.44, 0.56), (0.46, 0.54)]
    center_ys = [(0.35, 0.50), (0.36, 0.48), (0.37, 0.46)]
    cel_mins = [3, 4, 5, 6]                                # 15-30s
    trav_ys = [(0.30, 0.70), (0.35, 0.60), (0.40, 0.55)]
    max_runs = [10, 15, 20, 25]                            # seconds; tighter

    results = []
    combos = list(product(center_xs, center_ys, cel_mins, trav_ys, max_runs))
    print(f"sweeping {len(combos)} combos × {len(raw_cache)} games...\n")
    for (cx, cy, cm, ty, mr) in combos:
        cfg = {
            "CENTER_X_LO": cx[0], "CENTER_X_HI": cx[1],
            "CENTER_Y_LO": cy[0], "CENTER_Y_HI": cy[1],
            "MIN_CELEBRATION_SUSTAIN_FRAMES": cm,
            "TRAVERSAL_BALL_END_Y_LO": ty[0], "TRAVERSAL_BALL_END_Y_HI": ty[1],
        }
        per_game = {}
        for g, raw in raw_cache.items():
            det = run_cfg(raw, cfg, max_run_length=mr)
            per_game[g] = score(det, gt_cache[g], args.tol)
        total_tp = sum(s[0] for s in per_game.values())
        total_fp = sum(s[1] for s in per_game.values())
        total_fn = sum(s[2] for s in per_game.values())
        results.append((cfg, mr, per_game, total_tp, total_fp, total_fn))

    # Rank: maximize TP, then minimize FP, then minimize max-per-game-FP
    results.sort(key=lambda r: (-r[3], r[4], max(s[1] for s in r[2].values())))

    print(f"=== TOP {args.top} configs (rank by total TP desc, then total FP asc) ===")
    print(f"{'cx':<14}{'cy':<14}{'cMin':>5}{'trav_y':<14}{'maxR':>5}  "
          + "  ".join(f"{g[:11]}" for g in raw_cache.keys()) + "   TP/FP/FN")
    for cfg, mr, pg, ttp, tfp, tfn in results[:args.top]:
        cx = f"({cfg['CENTER_X_LO']:.2f},{cfg['CENTER_X_HI']:.2f})"
        cy = f"({cfg['CENTER_Y_LO']:.2f},{cfg['CENTER_Y_HI']:.2f})"
        ty = f"({cfg['TRAVERSAL_BALL_END_Y_LO']:.2f},{cfg['TRAVERSAL_BALL_END_Y_HI']:.2f})"
        mr_s = "-" if mr is None else f"{mr}s"
        per = "  ".join(f"{pg[g][0]}/{pg[g][1]:>2}" for g in raw_cache.keys())
        print(f"{cx:<14}{cy:<14}{cfg['MIN_CELEBRATION_SUSTAIN_FRAMES']:>5}"
              f"{ty:<14}{mr_s:>5}  {per}   {ttp}/{tfp}/{tfn}")


if __name__ == "__main__":
    main()
