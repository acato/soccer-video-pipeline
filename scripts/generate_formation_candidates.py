"""Generate goal-candidate windows from formation signature in cached YOLO data.

A "formation frame" has:
- wide_shot (≥18 field players)
- balanced L/R distribution (|p_left - p_right| ≤ 3)
- 1-3 players in center circle
- ball detection NOT required

These conditions are necessary (but not sufficient) for a post-goal kickoff.
We cluster consecutive/nearby formation frames into ~30s windows and emit
each cluster as a candidate. The VLM (v3) is the discriminator.

Usage:
    python scripts/generate_formation_candidates.py \\
        --per-frame /tmp/kickoff_rush_frames.jsonl \\
        --out /tmp/kickoff_rush_formation_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Formation thresholds (same as the FORMATION_* constants in detect_kickoffs.py)
WIDE_MIN = 18
TOTAL_MAX = 30
LR_BALANCE_MAX = 5
CIRCLE_MIN = 0      # don't require center-circle player (VLM will discriminate)
CIRCLE_MAX = 5

# Clustering — group formation frames within this many seconds into one window
CLUSTER_GAP_SECONDS = 30
MIN_CLUSTER_FRAMES = 1  # even a single formation frame triggers a candidate


def is_formation(r: dict) -> bool:
    total = r.get("total_field", 0)
    if not (WIDE_MIN <= total <= TOTAL_MAX):
        return False
    if abs(r.get("p_left", 0) - r.get("p_right", 0)) > LR_BALANCE_MAX:
        return False
    in_c = r.get("in_circle", 0)
    if not (CIRCLE_MIN <= in_c <= CIRCLE_MAX):
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-frame", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gap", type=float, default=CLUSTER_GAP_SECONDS)
    p.add_argument("--min-frames", type=int, default=MIN_CLUSTER_FRAMES)
    args = p.parse_args()

    rows = [json.loads(l) for l in Path(args.per_frame).read_text().splitlines() if l.strip()]
    formation = [r for r in rows if is_formation(r)]
    print(f"frames: {len(rows)}, formation frames: {len(formation)} "
          f"({100*len(formation)/max(1,len(rows)):.1f}%)", file=sys.stderr)

    # Cluster by time gap
    clusters: list[list[dict]] = []
    current: list[dict] = []
    for f in formation:
        if current and (f["t"] - current[-1]["t"]) > args.gap:
            clusters.append(current)
            current = []
        current.append(f)
    if current:
        clusters.append(current)
    clusters = [c for c in clusters if len(c) >= args.min_frames]
    print(f"clusters: {len(clusters)}", file=sys.stderr)

    out_rows = []
    for c in clusters:
        # Anchor at the MIDDLE of the cluster — gives VLM the best chance
        # to see celebration before AND kickoff resumption after
        t_mid = c[len(c) // 2]["t"]
        out_rows.append({
            "event_type": "goal",
            "start_sec": t_mid,
            "end_sec": t_mid + 2.0,
            "confidence": 0.5,
            "reasoning": (
                f"formation cluster: {len(c)} formation frames spanning "
                f"{c[0]['t']:.1f}-{c[-1]['t']:.1f}s"
            ),
            "triage_labels": ["kickoff_formation"],
            "_method": "formation_cluster",
            "_kickoff_start": t_mid,
            "_kickoff_end": t_mid + 5.0,
            "_cluster_start": c[0]["t"],
            "_cluster_end": c[-1]["t"],
            "_cluster_size": len(c),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} candidates → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
