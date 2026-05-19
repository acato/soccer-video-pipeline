"""Merge kickoff-ensemble GOAL detections into a dual_pass_events.jsonl.

The ensemble produces formation_base (and pattern_v11 once vLLM 0.19.1 is
re-pinned) outputs in the same general format as dual_pass events:

    {"event_type": "goal", "start_sec": <float>, "end_sec": <float>,
     "confidence": <float>, "reasoning": "...", "triage_labels": [...]}

We keep only entries with _vlm_verdict == "GOAL", strip the ensemble-only
metadata fields, and dedup against existing goals in the dual_pass file
within a configurable tolerance.

Usage:
    python scripts/merge_ensemble_into_events.py \\
        --dual-pass /tmp/soccer-pipeline/<job_id>/diagnostics/dual_pass_events.jsonl \\
        --ensemble /tmp/kickoff_game_22_formation_v2_base.jsonl \\
        --ensemble /tmp/kickoff_game_22_pattern_v11_verified.jsonl \\
        --out      /tmp/soccer-pipeline/<job_id>/diagnostics/dual_pass_events_augmented.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEDUP_WINDOW_SEC = 30.0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def strip_ensemble_meta(event: dict) -> dict:
    """Drop _vlm_*, _kickoff_*, _cluster_* fields the eval doesn't need."""
    return {k: v for k, v in event.items() if not k.startswith("_")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dual-pass", required=True, type=Path)
    p.add_argument("--ensemble", action="append", required=True, type=Path,
                   help="ensemble output jsonl (one or more)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dedup-window", type=float, default=DEDUP_WINDOW_SEC)
    args = p.parse_args()

    base = load_jsonl(args.dual_pass)
    base_goal_times = sorted(
        e["start_sec"] for e in base if e.get("event_type") == "goal"
    )

    added = []
    seen_added_times: list[float] = []
    for ens_path in args.ensemble:
        if not ens_path.exists():
            print(f"  WARN: {ens_path} missing, skipping")
            continue
        for e in load_jsonl(ens_path):
            if e.get("_vlm_verdict") != "GOAL":
                continue
            if e.get("event_type") != "goal":
                continue
            t = e["start_sec"]
            # Skip if a base or earlier-added goal is within the dedup window
            if any(abs(t - bt) <= args.dedup_window for bt in base_goal_times):
                continue
            if any(abs(t - at) <= args.dedup_window for at in seen_added_times):
                continue
            added.append(strip_ensemble_meta(e))
            seen_added_times.append(t)

    merged = base + added
    merged.sort(key=lambda x: x["start_sec"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")

    print(f"dual_pass goals: {len(base_goal_times)}")
    print(f"ensemble GOAL events read: "
          f"{sum(len(load_jsonl(p)) for p in args.ensemble if p.exists())}")
    print(f"ensemble goals ADDED (dedup applied): {len(added)}")
    for e in added:
        print(f"  +{e['start_sec']:.0f}s  conf={e.get('confidence', '?')}  "
              f"reason={e.get('reasoning','')[:60]}")
    print(f"wrote {len(merged)} events to {args.out}")


if __name__ == "__main__":
    main()
