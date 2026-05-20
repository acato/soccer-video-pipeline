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

DEDUP_WINDOW_SEC = 60.0  # game_20 has 65s-spaced GT goals — keep this <65


def aggregate_relaxed(labels):
    """Relaxed rule from verify_kickoffs_vlm_v3.py — celebration, goal→reset,
    OR kickoff_restart→active fires GOAL."""
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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def strip_ensemble_meta(event: dict) -> dict:
    """Drop _vlm_*, _kickoff_*, _cluster_* fields the eval doesn't need."""
    return {k: v for k, v in event.items() if not k.startswith("_")}


def tag_tier(event: dict, tier: str) -> dict:
    """Add a goal_tier field to an event's metadata (or top-level if absent)."""
    e = dict(event)
    e.setdefault("metadata", {})
    e["metadata"]["goal_tier"] = tier
    return e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dual-pass", required=True, type=Path)
    p.add_argument("--ensemble", action="append", required=True, type=Path,
                   help="ensemble output jsonl (one or more)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dedup-window", type=float, default=DEDUP_WINDOW_SEC)
    p.add_argument("--relaxed-aggregation", action="store_true",
                   help="re-aggregate ensemble events with the relaxed kickoff_restart rule "
                        "(catches more TPs at the cost of more FPs)")
    args = p.parse_args()

    base = load_jsonl(args.dual_pass)
    # Tag dual_pass goals as "confirmed" tier. Use the timestamp field actually
    # present in dual_pass events (timestamp_start) — fall back to start_sec.
    def _t(e):
        return e.get("start_sec", e.get("timestamp_start"))
    base_goal_times = sorted(_t(e) for e in base if e.get("event_type") == "goal")
    base_tagged = []
    for e in base:
        if e.get("event_type") == "goal":
            base_tagged.append(tag_tier(e, "confirmed"))
        else:
            base_tagged.append(e)

    added = []
    seen_added_times: list[float] = []
    for ens_path in args.ensemble:
        if not ens_path.exists():
            print(f"  WARN: {ens_path} missing, skipping")
            continue
        for e in load_jsonl(ens_path):
            if e.get("event_type") != "goal":
                continue
            # Apply the requested aggregation rule
            if args.relaxed_aggregation and "_vlm_labels" in e:
                if aggregate_relaxed(e["_vlm_labels"]) != "GOAL":
                    continue
            else:
                verdict = e.get("_vlm_verdict")
                if verdict and verdict != "GOAL":
                    continue
            t = e["start_sec"]
            # Skip if a base or earlier-added goal is within the dedup window
            if any(abs(t - bt) <= args.dedup_window for bt in base_goal_times):
                continue
            if any(abs(t - at) <= args.dedup_window for at in seen_added_times):
                continue
            tagged = tag_tier(strip_ensemble_meta(e), "candidate")
            added.append(tagged)
            seen_added_times.append(t)

    merged = base_tagged + added
    merged.sort(key=lambda x: _t(x))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")

    print(f"  confirmed tier (dual_pass goals): {len(base_goal_times)}")
    print(f"  ensemble GOAL events read: "
          f"{sum(len(load_jsonl(p)) for p in args.ensemble if p.exists())}")
    print(f"  candidate tier (ensemble goals after dedup): {len(added)}")
    for e in added:
        print(f"    +{e['start_sec']:.0f}s  conf={e.get('confidence', '?')}  "
              f"tier={e.get('metadata',{}).get('goal_tier','?')}")
    print(f"  wrote {len(merged)} events to {args.out}")


if __name__ == "__main__":
    main()
