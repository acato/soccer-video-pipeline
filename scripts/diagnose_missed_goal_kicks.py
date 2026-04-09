#!/usr/bin/env python3
"""
Diagnose why goal_kicks are being missed by the dual-pass detector.

For each goal_kick GT event that the matcher marks as FN, look at:
  - what the 8B triage labeled the window(s) containing that timestamp
  - whether any merged candidate window covered it
  - whether that candidate survived the max_candidates cap
  - what (if anything) the 32B classified for that window

Usage:
    python scripts/diagnose_missed_goal_kicks.py \\
        --job-id 0eb76694-edfa-46d3-9ba0-d2102d781cac \\
        --event-type goal_kick \\
        --tolerance 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_detection import (
    load_ground_truth,
    load_detected_events,
    match_events,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--event-type", default="goal_kick")
    p.add_argument("--tolerance", type=float, default=30.0)
    p.add_argument("--flag-window", type=float, default=15.0,
                   help="± seconds around GT timestamp to search triage flags")
    p.add_argument("--job-root", default="/tmp/soccer-pipeline")
    args = p.parse_args()

    job_dir = Path(args.job_root) / args.job_id / "diagnostics"
    flags_path = job_dir / "triage_flags.jsonl"
    cands_path = job_dir / "triage_candidates.jsonl"
    events_path = job_dir / "dual_pass_events.jsonl"

    for path in (flags_path, cands_path, events_path):
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1

    flags = [json.loads(line) for line in open(flags_path) if line.strip()]
    cands = [json.loads(line) for line in open(cands_path) if line.strip()]

    print(f"Job: {args.job_id}")
    print(f"Event type: {args.event_type}")
    print(f"Match tolerance: {args.tolerance}s")
    print(f"Flag search window: ±{args.flag_window}s")
    print()

    print(f"Total triage flags: {len(flags)}")
    label_counts = Counter(f["label"] for f in flags)
    print(f"Flag label distribution: {dict(label_counts)}")
    print(f"Candidate windows (post-cap): {len(cands)}")
    print()

    gt = load_ground_truth(half2_video_start=3916.0, half2_game_offset=2700.0)
    det = load_detected_events(str(events_path))

    results = match_events(gt, det, tolerance_sec=args.tolerance)
    r = results.get(args.event_type)
    if r is None:
        print(f"no GT entries for event_type={args.event_type}")
        return 0

    print(f"GT {args.event_type}: {r['gt']}  detected: {r['detected']}  "
          f"TP: {r['tp']}  FN: {r['fn']}  FP: {r['fp']}")
    print()

    missed = sorted(r["fn_events"], key=lambda e: e.video_time_sec)
    print(f"Missed {args.event_type}s ({len(missed)}):")
    print(f"{'vid_sec':>9}  {'half':>4}  {'game':>6}  "
          f"{'flags_near':<40}  {'cand_labels':<20}  {'in_cap':>6}  "
          f"{'nearest_det':<25}")
    print("-" * 120)

    flagged_sp = 0
    flagged_other = 0
    not_flagged = 0
    survived_cap = 0

    for e in missed:
        t = e.video_time_sec
        nearby = [f for f in flags if abs(f["center_sec"] - t) <= args.flag_window]
        nearby_c = Counter(f["label"] for f in nearby)
        nl_str = " ".join(f"{k}:{v}" for k, v in nearby_c.most_common())

        covering = [c for c in cands if c["start_sec"] - 2 <= t <= c["end_sec"] + 2]
        cand_labels = set()
        for c in covering:
            cand_labels.update(c.get("labels", []))
        cand_str = ",".join(sorted(cand_labels)) if cand_labels else "-"
        in_cap = "Y" if covering else "N"
        if covering:
            survived_cap += 1

        # Find nearest 32B detection (any type) within the flag window
        nearest = min(
            det,
            key=lambda d: abs(d.center_sec - t),
            default=None,
        )
        nd = "-"
        if nearest and abs(nearest.center_sec - t) <= 30:
            nd = f"{nearest.event_type}@{nearest.center_sec:.0f}"

        if any(f["label"] == "SET_PIECE" for f in nearby):
            flagged_sp += 1
        elif nearby:
            flagged_other += 1
        else:
            not_flagged += 1

        half = "1H" if e.half == 0 else "2H"
        gm = int(e.game_time_sec // 60)
        gs = e.game_time_sec % 60
        print(f"{t:>9.1f}  {half:>4}  {gm:>2}:{gs:>04.1f}  "
              f"{nl_str[:40]:<40}  {cand_str[:20]:<20}  {in_cap:>6}  {nd[:25]:<25}")

    total = len(missed)
    print()
    print("=" * 60)
    print(f"SUMMARY for missed {args.event_type}:")
    print(f"  SET_PIECE flagged near miss: {flagged_sp}/{total} "
          f"({100*flagged_sp/total:.0f}%)")
    print(f"  Other label near miss:       {flagged_other}/{total} "
          f"({100*flagged_other/total:.0f}%)")
    print(f"  No 8B flag within ±{args.flag_window}s: {not_flagged}/{total} "
          f"({100*not_flagged/total:.0f}%)")
    print(f"  Survived 150-candidate cap:  {survived_cap}/{total} "
          f"({100*survived_cap/total:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
