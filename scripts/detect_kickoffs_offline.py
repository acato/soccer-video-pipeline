"""Offline kickoff-detection iterator — uses cached per-frame YOLO data.

Reads the per_frame_out file from a previous detect_kickoffs.py run
(which contains raw YOLO outputs: ball, p_left, p_right, total_field,
in_circle, t). Re-applies derive_flags() and detect_goals() with the
current threshold settings from detect_kickoffs.py. This skips the
13-min YOLO pass entirely, letting us iterate on thresholds in seconds.

Usage:
    python scripts/detect_kickoffs_offline.py \\
        --per-frame /tmp/kickoff_game20_1H_frames.jsonl \\
        --out /tmp/kickoff_game20_offline.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import the live detection logic so we always reflect the latest thresholds
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_kickoffs import (  # type: ignore  # noqa: E402
    derive_flags,
    detect_goals,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-frame", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    rows = []
    with open(args.per_frame) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Strip cached flags — they reflect old thresholds.
    raw = []
    for r in rows:
        raw.append({
            "ball": r.get("ball"),
            "p_left": r.get("p_left", 0),
            "p_right": r.get("p_right", 0),
            "total_field": r.get("total_field", 0),
            "in_circle": r.get("in_circle", 0),
            "t": r["t"],
        })

    flags = derive_flags(raw)
    for r, f in zip(raw, flags):
        for k in ("wide_shot", "ball_at_center", "one_in_circle",
                  "kickoff_setup", "kickoff_setup_strong", "kickoff_scene"):
            if k in f:
                r[k] = f[k]

    timestamps = [r["t"] for r in raw]
    goals = detect_goals(flags, timestamps, args.interval)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for g in goals:
            f.write(json.dumps(g) + "\n")

    print(f"frames: {len(raw)}", file=sys.stderr)
    print(f"wide_shot=True: {sum(1 for f in flags if f['wide_shot'])}", file=sys.stderr)
    print(f"ball_at_center=True: {sum(1 for f in flags if f['ball_at_center'])}", file=sys.stderr)
    print(f"one_in_circle=True: {sum(1 for f in flags if f['one_in_circle'])}", file=sys.stderr)
    print(f"kickoff_setup=True: {sum(1 for f in flags if f['kickoff_setup'])}", file=sys.stderr)
    print(f"detected {len(goals)} goals → {args.out}", file=sys.stderr)
    for g in goals:
        cs = g.get("_celebration_start"); ce = g.get("_celebration_end")
        tr = g.get("_traversal_origin")
        cs_str = f"{cs:.1f}" if cs is not None else "-"
        ce_str = f"{ce:.1f}" if ce is not None else "-"
        tr_str = f"{tr:.1f}" if tr is not None else "-"
        print(f"  goal @ {g['start_sec']:.1f}s (cel {cs_str}-{ce_str}, "
              f"traversal {tr_str}, kickoff {g['_kickoff_start']:.1f}-{g['_kickoff_end']:.1f}s)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
