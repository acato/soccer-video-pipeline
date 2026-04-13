"""Evaluate a run's detections against ground truth with proper TP/FP/FN matching.

Usage: python eval_run.py <events_jsonl> [--tolerance 30]

Converts GT match-clock timestamps to video-clock, then does greedy
nearest-neighbor matching within a tolerance window.
"""
import json
import sys
from collections import Counter

# ── Video timing ──────────────────────────────────────────────────────
# The video has a pre-game segment before the match starts.
# We need to find the right offset.  GT note says "~7 min pre-game offset".
# Half 1: match clock 0:00 → video ~420s
# Halftime gap: between end of H1 video and start of H2 match clock.
# We'll calibrate these from known anchor points if available.

PRE_GAME_SEC = 800.0       # Calibrated: GT goal 1 (382s match) = det 1182s => offset 800

# GT timestamps are CONTINUOUS across halves (the match clock includes halftime).
# No separate halftime gap needed: video_sec = PRE_GAME_SEC + match_sec.
# Verified: GT goal 3 (3037s match) => video 3837s, detected at 3831s (6s error).


def gt_to_video_sec(time_ms: int, half: int) -> float:
    """Convert GT match-clock milliseconds to video seconds.

    The GT note says "continuous across halves" — H2 times continue from
    H1 start without reset.  The video offset (pre-game) is ~800s.
    """
    match_sec = time_ms / 1000.0
    return PRE_GAME_SEC + match_sec


def load_gt(gt_path="tests/fixtures/ground_truth_rush_vs_reign.json"):
    with open(gt_path) as f:
        gt = json.load(f)

    gt_events = {}  # type -> list of video_sec

    # Goals
    gt_events["goal"] = [
        gt_to_video_sec(e["time_ms"], e["half"])
        for e in gt["goals"]["events"]
    ]

    # Shots on target (GT "shots_on_target" = shots saved by GK, not goals)
    gt_events["shot_on_target"] = [
        gt_to_video_sec(e["time_ms"], e["half"])
        for e in gt["shots_on_target"]["events"]
    ]

    # Corners
    gt_events["corner_kick"] = [
        gt_to_video_sec(e["time_ms"], e["half"])
        for e in gt["corners"]["events"]
    ]

    # Goal kicks (combine both GKs)
    gk_events = []
    for e in gt["goal_kicks"]["elena_goal_kicks"]["events"]:
        gk_events.append(gt_to_video_sec(e["time_ms"], e["half"]))
    for e in gt["goal_kicks"]["opponent_goal_kicks"]["events"]:
        gk_events.append(gt_to_video_sec(e["time_ms"], e["half"]))
    gt_events["goal_kick"] = sorted(gk_events)

    # Throw-ins
    gt_events["throw_in"] = [
        gt_to_video_sec(e["time_ms"], e["half"])
        for e in gt["throw_ins"]["events"]
    ]

    # Free kicks
    gt_events["free_kick_shot"] = [
        gt_to_video_sec(e["time_ms"], e["half"])
        for e in gt["free_kicks"]["events"]
    ]

    # Saves (combine both GKs, map catch→catch, parry→shot_stop_diving)
    save_events = []
    for e in gt["saves"]["elena_saves"]["events"]:
        save_events.append(gt_to_video_sec(e["time_ms"], e["half"]))
    for e in gt["saves"]["opponent_gk_saves"]["events"]:
        save_events.append(gt_to_video_sec(e["time_ms"], e["half"]))
    gt_events["saves"] = sorted(save_events)

    return gt_events


def match_events(det_times, gt_times, tolerance=30.0):
    """Greedy nearest-neighbor matching. Returns (TP, FP, FN, matches)."""
    gt_remaining = list(gt_times)
    tp = 0
    matches = []

    for dt in sorted(det_times):
        best_idx = None
        best_dist = tolerance + 1
        for i, gt in enumerate(gt_remaining):
            dist = abs(dt - gt)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None and best_dist <= tolerance:
            matches.append((dt, gt_remaining[best_idx], best_dist))
            gt_remaining.pop(best_idx)
            tp += 1

    fp = len(det_times) - tp
    fn = len(gt_remaining)
    return tp, fp, fn, matches


def main():
    events_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/soccer-pipeline/e4728f36-394b-45ce-9852-d7887c5838ae/diagnostics/dual_pass_events.jsonl"
    tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    with open(events_path) as f:
        det_events = [json.loads(l) for l in f]

    gt_events = load_gt()

    # Group detections by type
    det_by_type = {}
    for e in det_events:
        det_by_type.setdefault(e["event_type"], []).append(e["start_sec"])

    print(f"Tolerance: {tolerance:.0f}s")
    print(f"{'Type':<22} {'GT':>4} {'Det':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 70)

    total_tp = total_fp = total_fn = 0

    eval_types = [
        ("goal", "goal"),
        ("shot_on_target", "shot_on_target"),
        ("set_piece", "set_piece"),  # merged: GT corner+free_kick vs det set_piece+corner+free_kick
        ("goal_kick", "goal_kick"),
        ("throw_in", "throw_in"),
        ("saves", "saves"),  # special: match against shot_stop_diving + catch
    ]

    for gt_key, label in eval_types:
        if label == "set_piece":
            # Merge GT corner_kick + free_kick_shot
            gt_times = sorted(
                gt_events.get("corner_kick", []) +
                gt_events.get("free_kick_shot", [])
            )
        else:
            gt_times = gt_events.get(gt_key, [])

        if label == "saves":
            det_times = sorted(
                det_by_type.get("shot_stop_diving", []) +
                det_by_type.get("catch", [])
            )
        elif label == "set_piece":
            # Merge det set_piece + corner_kick + free_kick_shot
            det_times = sorted(
                det_by_type.get("set_piece", []) +
                det_by_type.get("corner_kick", []) +
                det_by_type.get("free_kick_shot", [])
            )
        else:
            det_times = sorted(det_by_type.get(gt_key, []))

        tp, fp, fn, matches = match_events(det_times, gt_times, tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(f"{label:<22} {len(gt_times):>4} {len(det_times):>4} {tp:>4} {fp:>4} {fn:>4} {prec:>5.0%} {rec:>5.0%} {f1:>5.3f}")

        # Show unmatched GT (FN)
        if fn > 0:
            matched_gt = set(m[1] for m in matches)
            missed = [t for t in gt_times if t not in matched_gt]
            for t in missed[:5]:
                print(f"  MISS GT@{t:.0f}s")

    print("-" * 70)
    total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec) if (total_prec + total_rec) > 0 else 0
    print(f"{'TOTAL':<22} {'':<4} {'':<4} {total_tp:>4} {total_fp:>4} {total_fn:>4} {total_prec:>5.0%} {total_rec:>5.0%} {total_f1:>5.3f}")


if __name__ == "__main__":
    main()
