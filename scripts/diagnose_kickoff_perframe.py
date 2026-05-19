"""Diagnose kickoff-pattern detector across games — compare per-frame
trigger rates and sample false-positive evidence chains.

Usage:
    python scripts/diagnose_kickoff_perframe.py game_22 game_21 rush game_20_1H game_20_2H
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

GAMES = {
    "game_22": {
        "frames": "/tmp/kickoff_game_22_frames.jsonl",
        "dets":   "/tmp/kickoff_game_22.jsonl",
        # GT video times (fit by hand earlier)
        "tp_gt_video": [(2655.0, 2625.4)],  # det → GT pairing in video time
        "fitted_offset_1h": 460,
        "fitted_offset_2h": 540,
        "gt_1h_video": [1989.7, 2625.4],
        "gt_2h_video": [4274.7, 5393.3, 5606.6],
    },
    "game_21": {
        "frames": "/tmp/kickoff_game_21_frames.jsonl",
        "dets":   "/tmp/kickoff_game_21.jsonl",
        "tp_gt_video": [(2005.0, 2004.6)],
        "gt_1h_video": [1574.6, 2004.6],
        "gt_2h_video": [],
    },
    "game_20_1H": {
        "frames": "/tmp/kickoff_game20_1H_frames.jsonl",
        "dets":   "/tmp/kickoff_game20_1H.jsonl",
        "tp_gt_video": [(865.0, 872.0), (2125.0, 2114.0)],
        "gt_1h_video": [872.0, 937.0, 1439.2, 2114.2],
        "gt_2h_video": [],
    },
    "game_20_2H": {
        "frames": "/tmp/kickoff_game20_2H_frames.jsonl",
        "dets":   "/tmp/kickoff_game20_2H.jsonl",
        "tp_gt_video": [],
        # Per scan_log, 2H scanned only 3554-7229s
        "gt_2h_video_estimate_lo": 2535,  # if halftime=300
        "gt_2h_video_estimate_hi": 4365,
        "gt_1h_video": [],
        "gt_2h_video": [],  # unknown without precise halftime
    },
    "rush": {
        "frames": "/tmp/kickoff_rush_frames.jsonl",
        "dets":   "/tmp/kickoff_rush.jsonl",
        "tp_gt_video": [(600.0, 601.1), (895.0, 865.0),
                        (2635.0, 2605.4), (2815.0, 2830.5)],
        "gt_1h_video": [601.1, 865.0],
        "gt_2h_video": [2605.4, 2830.5],
    },
}


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def per_frame_summary(frames):
    n = len(frames)
    wide = sum(1 for f in frames if f["wide_shot"])
    ball_c = sum(1 for f in frames if f["ball_at_center"])
    ko = sum(1 for f in frames if f["kickoff_setup"])
    ko_strong = sum(1 for f in frames if f.get("kickoff_setup_strong"))
    one_circle = sum(1 for f in frames if f.get("one_in_circle"))
    ball_seen = sum(1 for f in frames if f["ball"] is not None)
    avg_field = sum(f["total_field"] for f in frames) / n if n else 0
    return dict(n=n, wide=wide, ball_c=ball_c, ko=ko, ko_strong=ko_strong,
                one_circle=one_circle, ball_seen=ball_seen, avg_field=avg_field)


def label_frame(f):
    ws = "W" if f["wide_shot"] else "-"
    bc = "B" if f["ball_at_center"] else "-"
    oc = "C" if f.get("one_in_circle") else "-"
    ko = "K" if f["kickoff_setup"] else "-"
    ball = f["ball"]
    bxy = f"({ball[0]:.2f},{ball[1]:.2f})" if ball else "None"
    return (f"t={f['t']:5.0f}  [{ws}{bc}{oc}{ko}]  "
            f"field={f['total_field']:2d}  in_c={f['in_circle']}  ball={bxy}")


def show_chain(label, frames, det):
    """Print frames around a detection's evidence chain."""
    cs = det.get("_celebration_start")
    ce = det.get("_celebration_end")
    ks = det["_kickoff_start"]
    ke = det["_kickoff_end"]
    lo = (cs if cs is not None else det["start_sec"]) - 10
    hi = ke + 10
    nearby = [f for f in frames if lo <= f["t"] <= hi]
    print(f"\n  [{label}] det @ {det['start_sec']}s "
          f"(method={det['_method'].split('_')[-1]}, "
          f"cel={cs}-{ce}, kickoff={ks:.0f}-{ke:.0f})")
    for f in nearby:
        print(f"    {label_frame(f)}")


def main(games):
    print("=== Per-frame trigger rates ===")
    print(f"{'game':<14} {'n':>4}  {'wide%':>6} {'ball_c%':>7} {'ko%':>5} "
          f"{'koS%':>5} {'1in_c%':>7} {'ball_seen%':>10} {'avgF':>5}")
    summaries = {}
    for g in games:
        cfg = GAMES[g]
        if not Path(cfg["frames"]).exists():
            print(f"  {g}: frames file missing")
            continue
        frames = load_jsonl(cfg["frames"])
        s = per_frame_summary(frames)
        summaries[g] = (frames, s)
        n = s["n"]
        print(f"{g:<14} {n:>4}  {s['wide']/n*100:>5.1f}% "
              f"{s['ball_c']/n*100:>6.1f}% {s['ko']/n*100:>4.1f}% "
              f"{s['ko_strong']/n*100:>4.1f}% {s['one_circle']/n*100:>6.1f}% "
              f"{s['ball_seen']/n*100:>9.1f}% {s['avg_field']:>4.1f}")

    # Sample evidence chains: 1 TP + 3 FPs for rush, 1 each for game_22
    for g in ("rush", "game_22"):
        if g not in summaries:
            continue
        frames, _ = summaries[g]
        dets = load_jsonl(GAMES[g]["dets"])
        tp_times = {d_t for d_t, _ in GAMES[g]["tp_gt_video"]}
        tps = [d for d in dets if d["start_sec"] in tp_times]
        fps = [d for d in dets if d["start_sec"] not in tp_times]
        print(f"\n\n=== {g.upper()} — evidence chains ===")
        if tps:
            show_chain(f"{g} TP", frames, tps[0])
        for fp in fps[:3]:
            show_chain(f"{g} FP", frames, fp)


if __name__ == "__main__":
    main(sys.argv[1:] or list(GAMES.keys()))
