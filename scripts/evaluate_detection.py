#!/usr/bin/env python3
"""
Evaluate dual-pass detection results against ground truth.

Usage:
    python evaluate_detection.py <events_jsonl> [--video-offset 418.0] [--tolerance 10]

The ground truth is loaded from the Rush game analytics JSON files on the NAS.
Matches detected events to GT within a configurable time tolerance.
Reports per-type precision, recall, and F1.
"""
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
# Video offset: game starts at 6:58 in the video file
VIDEO_OFFSET = 418.0

# Second half starts at approximately 58:20 in game time (from analytics)
# The 2nd half JSON resets event_time to 0, so we need the offset
# from video timestamp to align. We'll compute from the data.

# NAS paths for GT files (macOS)
GT_FILES = [
    "/Volumes/transit/08 GA (U19) vs Washington Rush U19 (W)_1st Half.json",
    "/Volumes/transit/08 GA (U19) vs Washington Rush U19 (W)_2nd Half.json",
]

# Map GT event names/types to our pipeline event types
GT_TO_PIPELINE = {
    "Saves/Catches": "catch",
    "Saves/Parries": "shot_stop_diving",  # parries are diving saves
    "Set Pieces/Corners": "corner_kick",
    "Set Pieces/Goal Kicks": "goal_kick",
    "Set Pieces/Throw-Ins": "throw_in",
    "Set Pieces/Freekicks": "free_kick_shot",
    "Set Pieces/Penalty Kicks": "penalty",
    "Goals Conceded": "goal",
    "Shots & Goals": "shot_on_target",  # approximate
}


@dataclass
class GTEvent:
    """Ground truth event."""
    half: int  # 0 = 1st, 1 = 2nd
    game_time_sec: float  # seconds from start of half
    video_time_sec: float  # seconds in video file
    event_type: str  # pipeline event type
    gt_name: str  # original GT event name
    team: str
    player: str


@dataclass
class DetectedEvent:
    """Detected event from pipeline."""
    event_type: str
    start_sec: float
    end_sec: float
    center_sec: float
    confidence: float
    reasoning: str


def load_ground_truth(
    half2_video_start: float = 3916.0,
    half2_game_offset: float = 2700.0,
) -> list[GTEvent]:
    """Load ground truth events from NAS JSON files.

    The analytics JSON uses cumulative game clock (event_time in ms from
    kickoff).  The 2nd half JSON does NOT reset to zero — times start around
    2700000ms (45:00 game time).

    Args:
        half2_video_start: Video timestamp where 2nd half kickoff occurs.
        half2_game_offset: Game-clock time (seconds) at 2nd half start.
                          Used to convert 2H game_sec to elapsed-in-half.
    """
    gt_events = []

    for half_idx, fn in enumerate(GT_FILES):
        p = Path(fn)
        if not p.exists():
            print(f"WARNING: GT file not found: {fn}", file=sys.stderr)
            continue

        with open(fn) as f:
            data = json.load(f)

        for entry in data["data"]:
            evtime_ms = entry["event_time"]
            team = entry.get("team_name", "")
            player = entry.get("player_name", "")

            for ev in entry.get("events", []):
                name = ev["event_name"]
                prop = ev.get("property", {})
                typ = prop.get("Type", "")

                key = name + ("/" + typ if typ else "")
                if key not in GT_TO_PIPELINE:
                    continue

                pipeline_type = GT_TO_PIPELINE[key]
                game_sec = evtime_ms / 1000.0

                # Convert to video timestamp
                if half_idx == 0:
                    video_sec = game_sec + VIDEO_OFFSET
                else:
                    # game_sec is cumulative clock (e.g. 2723s = 45:23);
                    # subtract the 2H game-clock offset to get elapsed
                    # time within the 2nd half, then add video offset.
                    elapsed_in_half = game_sec - half2_game_offset
                    video_sec = elapsed_in_half + half2_video_start

                gt_events.append(GTEvent(
                    half=half_idx,
                    game_time_sec=game_sec,
                    video_time_sec=video_sec,
                    event_type=pipeline_type,
                    gt_name=key,
                    team=team,
                    player=player,
                ))

    return gt_events


def load_detected_events(events_path: str) -> list[DetectedEvent]:
    """Load detected events from events.jsonl or dual_pass_events.jsonl."""
    events = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Handle both pipeline events.jsonl and diagnostics format
            if "timestamp_start" in d:
                start = d["timestamp_start"]
                end = d["timestamp_end"]
                etype = d["event_type"]
                conf = d.get("confidence", 0)
                reasoning = d.get("metadata", {}).get("vlm_reasoning", "")
            else:
                start = d.get("start_sec", 0)
                end = d.get("end_sec", 0)
                etype = d.get("event_type", "")
                conf = d.get("confidence", 0)
                reasoning = d.get("reasoning", "")

            events.append(DetectedEvent(
                event_type=etype,
                start_sec=start,
                end_sec=end,
                center_sec=(start + end) / 2,
                confidence=conf,
                reasoning=reasoning,
            ))

    return events


def match_events(
    gt_events: list[GTEvent],
    detected: list[DetectedEvent],
    tolerance_sec: float = 10.0,
) -> dict:
    """Match detected events to GT within tolerance.

    Returns dict with TP, FN, FP per event type and overall.
    """
    # Group GT and detected by type
    gt_by_type = defaultdict(list)
    for e in gt_events:
        gt_by_type[e.event_type].append(e)

    det_by_type = defaultdict(list)
    for e in detected:
        det_by_type[e.event_type].append(e)

    all_types = sorted(set(list(gt_by_type.keys()) + list(det_by_type.keys())))

    results = {}
    total_tp = total_fn = total_fp = 0

    for etype in all_types:
        gt_list = sorted(gt_by_type.get(etype, []), key=lambda e: e.video_time_sec)
        det_list = sorted(det_by_type.get(etype, []), key=lambda e: e.center_sec)

        # Greedy matching: for each GT event, find closest unmatched detection
        matched_det = set()
        tp = 0
        fn_events = []

        for gt in gt_list:
            best_idx = None
            best_dist = float("inf")
            for i, det in enumerate(det_list):
                if i in matched_det:
                    continue
                dist = abs(det.center_sec - gt.video_time_sec)
                if dist < best_dist and dist <= tolerance_sec:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                matched_det.add(best_idx)
                tp += 1
            else:
                fn_events.append(gt)

        fn = len(gt_list) - tp
        fp = len(det_list) - len(matched_det)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[etype] = {
            "gt": len(gt_list),
            "detected": len(det_list),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fn_events": fn_events,
        }

        total_tp += tp
        total_fn += fn
        total_fp += fp

    # Overall
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0

    results["__overall__"] = {
        "tp": total_tp,
        "fn": total_fn,
        "fp": total_fp,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1": overall_f1,
    }

    return results


def print_report(results: dict):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 80)
    print("DETECTION EVALUATION REPORT")
    print("=" * 80)

    # Per-type table
    print(f"\n{'Event Type':<20} {'GT':>4} {'Det':>4} {'TP':>4} {'FN':>4} {'FP':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 74)

    for etype in sorted(results.keys()):
        if etype == "__overall__":
            continue
        r = results[etype]
        print(f"{etype:<20} {r['gt']:4d} {r['detected']:4d} {r['tp']:4d} {r['fn']:4d} {r['fp']:4d} "
              f"{r['precision']:6.2f} {r['recall']:6.2f} {r['f1']:6.2f}")

    # Overall
    print("-" * 74)
    o = results["__overall__"]
    print(f"{'OVERALL':<20} {'':>4} {'':>4} {o['tp']:4d} {o['fn']:4d} {o['fp']:4d} "
          f"{o['precision']:6.2f} {o['recall']:6.2f} {o['f1']:6.2f}")

    # Missed events detail
    print("\n" + "=" * 80)
    print("MISSED EVENTS (False Negatives)")
    print("=" * 80)
    for etype in sorted(results.keys()):
        if etype == "__overall__":
            continue
        fn_events = results[etype].get("fn_events", [])
        if not fn_events:
            continue
        print(f"\n  {etype} ({len(fn_events)} missed):")
        for e in fn_events:
            game_min = int(e.game_time_sec // 60)
            game_sec = e.game_time_sec % 60
            vid_min = int(e.video_time_sec // 60)
            vid_sec = e.video_time_sec % 60
            half = "1H" if e.half == 0 else "2H"
            print(f"    {half} game={game_min:2d}:{game_sec:04.1f}  video={vid_min:2d}:{vid_sec:04.1f}  "
                  f"{e.player:<20s} {e.team[:20]}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate detection results")
    parser.add_argument("events_file", help="Path to events.jsonl or dual_pass_events.jsonl")
    parser.add_argument("--video-offset", type=float, default=418.0,
                        help="Video offset in seconds (default: 418.0)")
    parser.add_argument("--tolerance", type=float, default=15.0,
                        help="Matching tolerance in seconds (default: 15.0)")
    parser.add_argument("--half2-start", type=float, default=3916.0,
                        help="Video timestamp where 2nd half starts (default: 3916.0)")
    parser.add_argument("--half2-game-offset", type=float, default=2700.0,
                        help="Game-clock seconds at 2nd half start (default: 2700.0)")
    args = parser.parse_args()

    global VIDEO_OFFSET
    VIDEO_OFFSET = args.video_offset

    print(f"Loading GT (video_offset={VIDEO_OFFSET}s, half2_start={args.half2_start}s, "
          f"half2_game_offset={args.half2_game_offset}s)...")
    gt_events = load_ground_truth(
        half2_video_start=args.half2_start,
        half2_game_offset=args.half2_game_offset,
    )
    print(f"  GT events: {len(gt_events)}")

    gt_counts = Counter(e.event_type for e in gt_events)
    for t, c in gt_counts.most_common():
        print(f"    {t}: {c}")

    print(f"\nLoading detected events from {args.events_file}...")
    detected = load_detected_events(args.events_file)
    print(f"  Detected events: {len(detected)}")

    det_counts = Counter(e.event_type for e in detected)
    for t, c in det_counts.most_common():
        print(f"    {t}: {c}")

    print(f"\nMatching with tolerance={args.tolerance}s...")
    results = match_events(gt_events, detected, tolerance_sec=args.tolerance)
    print_report(results)


if __name__ == "__main__":
    main()
