"""Dense YOLO re-sampling around each candidate.

For each formation candidate that fired GOAL under the relaxed rule,
extract frames at 1s intervals in ±10s window around the candidate.
Run the v9b ball detector + uisikdag player detector. Apply the STRICT
kickoff_setup_strong test (wide + ball_at_center + 1-3 in_circle).

KEEP if ANY of the 21 dense-sampled frames satisfies strict kickoff.

Outputs: per-game JSONL with each candidate's dense-frame analysis +
final verdict.

Usage:
    python scripts/dense_yolo_filter.py \\
        --candidates /tmp/kickoff_game_22_formation_v2_base.jsonl \\
        --video "/path/to/video.mp4" \\
        --out /tmp/kickoff_game_22_dense_yolo.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Re-use machinery from detect_kickoffs.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_kickoffs import (  # type: ignore  # noqa: E402
    analyze_batch, extract_frames_batch,
    CENTER_X_LO, CENTER_X_HI, CENTER_Y_LO, CENTER_Y_HI,
    WIDE_SHOT_MIN_PERSONS,
    BALL_MODEL_DEFAULT, PLAYER_MODEL_DEFAULT,
)


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


def is_strict_kickoff(per_frame_result, in_c_min=1, in_c_max=3):
    """wide_shot + ball_at_center + 1<=in_circle<=3."""
    if per_frame_result["total_field"] < WIDE_SHOT_MIN_PERSONS:
        return False
    ball = per_frame_result.get("ball")
    if not ball:
        return False
    bx, by = ball[0], ball[1]
    if not (CENTER_X_LO <= bx <= CENTER_X_HI and CENTER_Y_LO <= by <= CENTER_Y_HI):
        return False
    inc = per_frame_result.get("in_circle", 0)
    return in_c_min <= inc <= in_c_max


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True, type=Path,
                   help="formation jsonl from base FP8 ensemble")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--window", type=int, default=10,
                   help="±N seconds around candidate (default 10)")
    p.add_argument("--interval", type=float, default=1.0,
                   help="dense sampling interval (default 1.0s)")
    p.add_argument("--ball-model", default=BALL_MODEL_DEFAULT)
    p.add_argument("--player-model", default=PLAYER_MODEL_DEFAULT)
    p.add_argument("--workdir", default="/tmp/kickoff_dense_frames")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--in-c-max", type=int, default=3,
                   help="max in_circle players for strict kickoff (default 3)")
    args = p.parse_args()

    # Load candidates that fired GOAL under relaxed rule
    cands = []
    for line in args.candidates.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # If this is a base-FP8 formation file, use relaxed aggregator
        if "_vlm_labels" in r:
            if aggregate_relaxed(r["_vlm_labels"]) != "GOAL":
                continue
        else:
            # Pattern_v11 file: rely on existing _vlm_verdict
            if r.get("_vlm_verdict") != "GOAL":
                continue
        cands.append(r)

    print(f"loaded {len(cands)} GOAL candidates", file=sys.stderr)

    # Build timestamp list: for each candidate, ±window at interval
    timestamps_set = set()
    cand_timestamps = {}
    for c in cands:
        base = c["start_sec"]
        ts = []
        t = base - args.window
        while t <= base + args.window + 0.001:
            ts.append(round(t, 1))
            timestamps_set.add(round(t, 1))
            t += args.interval
        cand_timestamps[c["start_sec"]] = ts

    timestamps = sorted(t for t in timestamps_set if t > 0)
    print(f"unique timestamps to extract: {len(timestamps)}", file=sys.stderr)

    workdir = Path(args.workdir) / Path(args.video).stem

    from ultralytics import YOLO
    print(f"loading YOLO models...", file=sys.stderr)
    ball_model = YOLO(args.ball_model)
    player_model = YOLO(args.player_model)

    print(f"extracting + analyzing frames in batches of {args.batch_size}",
          file=sys.stderr)
    per_frame_all = {}
    t0 = time.time()
    for batch_start in range(0, len(timestamps), args.batch_size):
        batch_ts = timestamps[batch_start:batch_start + args.batch_size]
        paths = extract_frames_batch(args.video, batch_ts, workdir)
        results = analyze_batch(ball_model, player_model, paths)
        for ts, r in zip(batch_ts, results):
            per_frame_all[ts] = r
        elapsed = time.time() - t0
        rate = len(per_frame_all) / max(1, elapsed)
        eta = (len(timestamps) - len(per_frame_all)) / max(0.01, rate)
        print(f"  {len(per_frame_all)}/{len(timestamps)} ({rate:.1f}/s, ETA {eta/60:.1f}min)",
              file=sys.stderr)

    # Save ALL per-frame data for offline filter exploration.
    out_rows = []
    n_pass = 0
    for c in cands:
        base = c["start_sec"]
        cand_ts = cand_timestamps[base]
        all_frames = []
        passing_frames = []
        for ts in cand_ts:
            if ts not in per_frame_all:
                continue
            r = dict(per_frame_all[ts])
            r["t"] = ts
            all_frames.append(r)
            if is_strict_kickoff(r, in_c_max=args.in_c_max):
                passing_frames.append(r)
        verdict = "STRICT_KICKOFF" if passing_frames else "NO_STRICT_KICKOFF"
        if passing_frames:
            n_pass += 1
        out = dict(c)
        out["_dense_verdict"] = verdict
        out["_dense_passing"] = passing_frames
        out["_dense_frames"] = all_frames    # NEW: full per-frame data
        out_rows.append(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n=> {n_pass}/{len(cands)} passed strict kickoff test")
    print(f"   wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
