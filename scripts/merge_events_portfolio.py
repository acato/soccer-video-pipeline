"""Portfolio merge: combine two detector event lists by event-type routing.

Use case (v10 disambiguation, May 2026): the v10 c150 LoRA is the only
checkpoint that produces precision-1.00 new-venue goal detections, while
v8 has higher recall on majority classes (corner_kick, throw_in,
goal_kick). The hybrid takes goal/save events from the "precision model"
(e.g. v10 c150) and everything else from the "recall model" (e.g. v8).

Inputs: two dual_pass_events.jsonl files (or events.jsonl — auto-detects
schema). Output: a merged dual_pass_events.jsonl that
scripts/evaluate_detection.py can score directly.

Example:
    python scripts/merge_events_portfolio.py \\
      --precision /tmp/soccer-pipeline/<c150_job>/diagnostics/dual_pass_events.jsonl \\
      --recall    /tmp/soccer-pipeline/<v8_job>/diagnostics/dual_pass_events.jsonl \\
      --out       /tmp/hybrid_<game_id>_events.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Event types the precision model owns. Everything else falls to the recall
# model. Default tuned for v10 c150 (clean goal/save detection) + v8.
DEFAULT_PRECISION_TYPES = {
    "goal",
    "shot_stop_diving",
    "shot_stop_standing",
    "punch",
    "catch",
    "one_on_one",
}


def _to_diag_event(d: dict) -> dict:
    """Normalize an event line to the diagnostics schema.

    Handles both:
      events.jsonl       (full Event with timestamp_start/timestamp_end + metadata)
      dual_pass_events.jsonl (start_sec/end_sec + top-level reasoning)
    """
    if "timestamp_start" in d:
        return {
            "event_type": d["event_type"],
            "start_sec": d["timestamp_start"],
            "end_sec": d["timestamp_end"],
            "confidence": d.get("confidence", 0.0),
            "reasoning": d.get("metadata", {}).get("vlm_reasoning", ""),
            "triage_labels": d.get("metadata", {}).get("triage_labels", []),
        }
    return {
        "event_type": d.get("event_type", ""),
        "start_sec": d.get("start_sec", 0.0),
        "end_sec": d.get("end_sec", 0.0),
        "confidence": d.get("confidence", 0.0),
        "reasoning": d.get("reasoning", ""),
        "triage_labels": d.get("triage_labels", []),
    }


def load_events(path: Path) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(_to_diag_event(json.loads(line)))
    return events


def merge(precision_events: list[dict], recall_events: list[dict],
          precision_types: set[str]) -> tuple[list[dict], dict]:
    """Route events: precision_types come from precision_events, rest from
    recall_events. Returns (merged, stats)."""
    from_precision = [e for e in precision_events if e["event_type"] in precision_types]
    from_recall = [e for e in recall_events if e["event_type"] not in precision_types]

    merged = from_precision + from_recall
    merged.sort(key=lambda e: e["start_sec"])

    stats = {
        "from_precision": len(from_precision),
        "from_recall": len(from_recall),
        "total": len(merged),
        "precision_types": sorted(precision_types),
        "precision_input_total": len(precision_events),
        "recall_input_total": len(recall_events),
        "precision_other_dropped": sum(
            1 for e in precision_events if e["event_type"] not in precision_types
        ),
        "recall_priority_dropped": sum(
            1 for e in recall_events if e["event_type"] in precision_types
        ),
    }
    return merged, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--precision", required=True, type=Path,
                    help="events.jsonl from the precision model (goal/save events kept)")
    ap.add_argument("--recall", required=True, type=Path,
                    help="events.jsonl from the recall model (all other events kept)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output merged events.jsonl path")
    ap.add_argument("--precision-types", default=",".join(sorted(DEFAULT_PRECISION_TYPES)),
                    help=("comma-separated event types owned by the precision model. "
                          f"Default: {','.join(sorted(DEFAULT_PRECISION_TYPES))}"))
    args = ap.parse_args()

    if not args.precision.exists():
        print(f"ERROR: precision events file not found: {args.precision}", file=sys.stderr)
        return 1
    if not args.recall.exists():
        print(f"ERROR: recall events file not found: {args.recall}", file=sys.stderr)
        return 1

    precision_types = {t.strip() for t in args.precision_types.split(",") if t.strip()}

    p_events = load_events(args.precision)
    r_events = load_events(args.recall)
    print(f"loaded precision={len(p_events)} from {args.precision}", file=sys.stderr)
    print(f"loaded recall   ={len(r_events)} from {args.recall}", file=sys.stderr)
    print(f"precision types: {sorted(precision_types)}", file=sys.stderr)

    merged, stats = merge(p_events, r_events, precision_types)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")

    print(f"\nMerge stats:", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"\nWrote {len(merged)} merged events to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
