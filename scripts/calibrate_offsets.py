"""Calibrate VIDEO_OFFSET + half2_video_start for a new game.

Given a run's detected events (dual_pass_events.jsonl) and its GT
JSON files, grid-search the (VIDEO_OFFSET, half2_video_start) pair
that maximises alignment between detected `goal` events and GT
`Goals Conceded` entries — falling back to shot / shot-on-target
alignment when goals are too sparse.

Usage (on Mac, where GT lives on /Volumes/transit):

    python scripts/calibrate_offsets.py \\
        /tmp/soccer-pipeline/<job_id>/diagnostics/dual_pass_events.jsonl \\
        "/Volumes/transit/Games/17/<1st_half>.json" \\
        "/Volumes/transit/Games/17/<2nd_half>.json" \\
        --half2-game-offset 2700 \\
        --tolerance 45

Prints the best (VIDEO_OFFSET, half2_video_start) and the CLI
arguments to feed evaluate_detection.py.

The calibrator does NOT require working offsets to start — that's
the whole point. It assumes only:
  - GT uses cumulative game-clock in ms (event_time field)
  - Half 2 starts at game_clock ≈ half2_game_offset sec (default 2700)
  - Detected events have start_sec in video time
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_gt_events(
    gt_paths: list[str],
    match_event_name: str,
    match_type: str | None = None,
) -> list[tuple[int, float]]:
    """Return list of (half_idx, game_sec) for GT events matching the filter.

    half_idx: 0 for 1st half, 1 for 2nd half.
    game_sec: entry["event_time"] / 1000.
    """
    out: list[tuple[int, float]] = []
    for half_idx, p in enumerate(gt_paths):
        data = json.loads(Path(p).read_text())["data"]
        for entry in data:
            evtime_sec = entry["event_time"] / 1000.0
            for ev in entry.get("events", []):
                if ev.get("event_name") != match_event_name:
                    continue
                if match_type is not None:
                    if ev.get("property", {}).get("Type") != match_type:
                        continue
                out.append((half_idx, evtime_sec))
    return out


def _load_detected(
    events_path: str, event_type: str,
) -> list[float]:
    """Return sorted list of detected event video-time starts."""
    rows = [json.loads(l) for l in Path(events_path).read_text().splitlines() if l.strip()]
    return sorted(r["start_sec"] for r in rows if r.get("event_type") == event_type)


_MIN_HALFTIME_SEC: float = 300.0  # 5-minute minimum break — videos with
                                  # < 5 min halftime are vanishingly rare.


def _best_offsets_for_signal(
    gt_events: list[tuple[int, float]],
    detected: list[float],
    half2_game_offset: float,
    tolerance: float,
    video_offset_range: tuple[float, float, float] = (0.0, 600.0, 5.0),
    half2_start_range: tuple[float, float, float] = (300.0, 2000.0, 10.0),
    min_halftime_sec: float = _MIN_HALFTIME_SEC,
) -> tuple[float, float, int]:
    """Grid-search the (video_offset, half2_video_start) pair that maximises
    match count, breaking ties by minimising sum-of-squared residuals
    between each matched GT event and its nearest detection.

    half2_start_range is a RELATIVE range: actual half2_start is
    video_offset + half2_game_offset + halftime_guess where halftime_guess
    sweeps the range. This prevents half2 from falling before half1 end.

    Returns (video_offset, half2_start, match_count).
    """
    import bisect
    vo_start, vo_end, vo_step = video_offset_range
    h2_start, h2_end, h2_step = half2_start_range

    best: tuple[float, float, int, float] = (0.0, 0.0, -1, float("inf"))

    # Enforce a minimum halftime break — h2_start sweep floor is the
    # max of the caller's h2_start and the min halftime.
    h2_start = max(h2_start, min_halftime_sec)

    vo = vo_start
    while vo <= vo_end:
        # Half 2 video start = vo + half2_game_offset + halftime
        # halftime sweeps from min_halftime_sec (physically floor) to h2_end
        h2 = vo + half2_game_offset + h2_start
        h2_ceil = vo + half2_game_offset + h2_end
        while h2 <= h2_ceil:
            matches = 0
            ssr = 0.0  # sum of squared residuals for matched events
            for half_idx, game_sec in gt_events:
                if half_idx == 0:
                    v_sec = game_sec + vo
                else:
                    elapsed_in_half = game_sec - half2_game_offset
                    v_sec = elapsed_in_half + h2
                # Find nearest detection (may or may not be within tolerance)
                i = bisect.bisect_left(detected, v_sec)
                nearest = None
                if i < len(detected):
                    nearest = detected[i]
                if i > 0:
                    prev = detected[i - 1]
                    if nearest is None or abs(prev - v_sec) < abs(nearest - v_sec):
                        nearest = prev
                if nearest is not None and abs(nearest - v_sec) <= tolerance:
                    matches += 1
                    ssr += (nearest - v_sec) ** 2
            # Prefer more matches; tie-break by smaller SSR
            if matches > best[2] or (matches == best[2] and ssr < best[3]):
                best = (vo, h2, matches, ssr)
            h2 += h2_step
        vo += vo_step

    return (best[0], best[1], best[2])


def _score_offsets(
    signals: list[tuple[list[tuple[int, float]], list[float], float]],
    vo: float,
    h2: float,
    half2_game_offset: float,
    tolerance: float,
) -> tuple[float, float, int]:
    """Score a (vo, h2) candidate across multiple (gt, detected, weight) signals.

    Returns (weighted_match_score, sum_squared_residuals, raw_match_count).
    """
    import bisect
    total_score = 0.0
    total_ssr = 0.0
    total_matches = 0
    for gt_events, detected, weight in signals:
        for half_idx, game_sec in gt_events:
            if half_idx == 0:
                v_sec = game_sec + vo
            else:
                v_sec = (game_sec - half2_game_offset) + h2
            i = bisect.bisect_left(detected, v_sec)
            nearest = None
            if i < len(detected):
                nearest = detected[i]
            if i > 0:
                prev = detected[i - 1]
                if nearest is None or abs(prev - v_sec) < abs(nearest - v_sec):
                    nearest = prev
            if nearest is not None and abs(nearest - v_sec) <= tolerance:
                total_score += weight
                total_ssr += (nearest - v_sec) ** 2
                total_matches += 1
    return total_score, total_ssr, total_matches


def _best_offsets_combined(
    signals: list[tuple[list[tuple[int, float]], list[float], float]],
    half2_game_offset: float,
    tolerance: float,
    video_offset_range: tuple[float, float, float],
    half2_start_range: tuple[float, float, float],
    min_halftime_sec: float = _MIN_HALFTIME_SEC,
) -> tuple[float, float, float, int]:
    """Grid-search using a weighted-combined score across signals.

    Returns (video_offset, half2_video_start, best_score, best_match_count).
    """
    vo_start, vo_end, vo_step = video_offset_range
    h2_start, h2_end, h2_step = half2_start_range
    h2_start = max(h2_start, min_halftime_sec)

    best: tuple[float, float, float, float, int] = (
        0.0, 0.0, -1.0, float("inf"), 0,
    )

    vo = vo_start
    while vo <= vo_end:
        h2 = vo + half2_game_offset + h2_start
        h2_ceil = vo + half2_game_offset + h2_end
        while h2 <= h2_ceil:
            score, ssr, matches = _score_offsets(
                signals, vo, h2, half2_game_offset, tolerance,
            )
            if score > best[2] or (score == best[2] and ssr < best[3]):
                best = (vo, h2, score, ssr, matches)
            h2 += h2_step
        vo += vo_step

    return (best[0], best[1], best[2], best[4])


def calibrate(
    events_path: str,
    gt_paths: list[str],
    half2_game_offset: float = 2700.0,
    tolerance: float = 45.0,
    goal_weight: float = 10.0,
    shot_weight: float = 1.0,
) -> dict:
    """Find the best VIDEO_OFFSET + half2_video_start for a run.

    Combines two signals with different weights:
      - `goal` ↔ GT `Goals Conceded`: high weight (rare, precise events)
      - `shot_on_target` ↔ GT `Shots & Goals`: low weight (numerous,
        noisier, but essential when goals don't cover both halves)

    A weighted score per (vo, h2) candidate: goal_matches * goal_weight +
    shot_matches * shot_weight. Ties broken by minimising residual SSR.

    Falls back to shots-only if no goals exist on either side. Returns
    an error if neither signal has enough data.
    """
    gt_goals = _load_gt_events(gt_paths, "Goals Conceded")
    det_goals = _load_detected(events_path, "goal")
    gt_shots = _load_gt_events(gt_paths, "Shots & Goals")
    det_shots = _load_detected(events_path, "shot_on_target")

    signals: list[tuple[list[tuple[int, float]], list[float], float]] = []
    signal_names: list[str] = []
    if len(gt_goals) >= 1 and len(det_goals) >= 1:
        signals.append((gt_goals, det_goals, goal_weight))
        signal_names.append("goal")
    if len(gt_shots) >= 3 and len(det_shots) >= 3:
        signals.append((gt_shots, det_shots, shot_weight))
        signal_names.append("shot_on_target")

    if not signals:
        return {
            "error": "insufficient signal for calibration",
            "gt_goals": len(gt_goals),
            "det_goals": len(det_goals),
            "gt_shots": len(gt_shots),
            "det_shots": len(det_shots),
        }

    # Coarse grid search
    vo_c, h2_c, score_c, matches_c = _best_offsets_combined(
        signals, half2_game_offset, tolerance,
        video_offset_range=(0.0, 600.0, 10.0),
        half2_start_range=(300.0, 1800.0, 30.0),
    )

    # Fine search around the coarse winner
    halftime_center = h2_c - vo_c - half2_game_offset
    vo_f, h2_f, score_f, matches_f = _best_offsets_combined(
        signals, half2_game_offset, tolerance,
        video_offset_range=(max(0.0, vo_c - 20.0), vo_c + 20.0, 2.0),
        half2_start_range=(max(_MIN_HALFTIME_SEC, halftime_center - 40.0),
                           halftime_center + 40.0, 5.0),
    )

    video_offset, half2_start = vo_f, h2_f

    # Per-signal match counts at the winning offsets
    per_signal_matches: dict[str, int] = {}
    for (gt, det, _w), name in zip(signals, signal_names):
        _, _, m = _score_offsets(
            [(gt, det, 1.0)], video_offset, half2_start,
            half2_game_offset, tolerance,
        )
        per_signal_matches[name] = m

    return {
        "video_offset": round(video_offset, 1),
        "half2_video_start": round(half2_start, 1),
        "half2_game_offset": half2_game_offset,
        "tolerance": tolerance,
        "signals_used": signal_names,
        "weighted_score": round(score_f, 2),
        "matches": per_signal_matches,
        "gt_counts": {"goal": len(gt_goals), "shot_on_target": len(gt_shots)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate video offsets from detected events vs GT."
    )
    parser.add_argument("events_file",
                        help="Path to dual_pass_events.jsonl")
    parser.add_argument("gt_files", nargs="+",
                        help="Path(s) to GT JSON files (1st then 2nd half)")
    parser.add_argument("--half2-game-offset", type=float, default=2700.0,
                        help="Game-clock seconds at 2nd half kickoff "
                             "(default: 2700 = 45 min for U19+; use 2400 "
                             "for U15/youth with 40-min halves)")
    parser.add_argument("--tolerance", type=float, default=45.0,
                        help="Match tolerance in seconds (default: 45)")
    args = parser.parse_args()

    result = calibrate(
        events_path=args.events_file,
        gt_paths=args.gt_files,
        half2_game_offset=args.half2_game_offset,
        tolerance=args.tolerance,
    )

    print(json.dumps(result, indent=2))

    if "error" in result:
        return 1

    # Print a ready-to-paste evaluate_detection.py argument string.
    print(
        "\nSuggested evaluate_detection.py args:\n"
        f"  --tolerance {int(args.tolerance)} "
        f"--video-offset {result['video_offset']} "
        f"--half2-start {result['half2_video_start']} "
        f"--half2-game-offset {int(args.half2_game_offset)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
