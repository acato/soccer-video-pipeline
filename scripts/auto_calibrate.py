#!/usr/bin/env python3
"""
Auto-calibrate halftime offsets for training data extraction.

Uses visual motion analysis (frame differencing) to detect:
1. Pre-game → match transition (H1 kickoff)
2. Halftime gap (sustained low motion)
3. H2 kickoff (motion resumes)

Then computes offsets: video_time = match_clock_ms/1000 + offset

Usage:
    python scripts/auto_calibrate.py \
        --games-dir /Volumes/transit/Games \
        --output calibration.json

    # Single game with verbose:
    python scripts/auto_calibrate.py \
        --games-dir /Volumes/transit/Games \
        --game 11 --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_INTERVAL = 30    # seconds between sampled frames
FRAME_W = 160
FRAME_H = 90
SMOOTH_WINDOW = 5       # smooth over 5 samples = 2.5 min


# ---------------------------------------------------------------------------
# Motion extraction
# ---------------------------------------------------------------------------

def extract_motion_profile(mp4_path: Path, interval: int = SAMPLE_INTERVAL) -> list[float]:
    """Extract motion profile by computing frame differences at regular intervals.

    Returns list of motion values, one per interval. Index i corresponds to
    video time (i+1)*interval seconds.
    """
    frames_dir = Path(f"/tmp/soccer-pipeline/cal_{os.getpid()}")
    frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract tiny frames
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(mp4_path),
            "-vf", f"fps=1/{interval},scale={FRAME_W}:{FRAME_H}",
            "-pix_fmt", "rgb24",
            "-f", "image2", str(frames_dir / "f_%05d.raw"),
        ]
        # Using rawvideo pipe per-frame is simpler
        cmd2 = [
            "ffmpeg", "-v", "error",
            "-i", str(mp4_path),
            "-vf", f"fps=1/{interval},scale={FRAME_W}:{FRAME_H}",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo", "-",
        ]
        result = subprocess.run(cmd2, capture_output=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr.decode()[:200]}")

        raw = result.stdout
        frame_size = FRAME_W * FRAME_H * 3
        n_frames = len(raw) // frame_size

        motions = []
        for i in range(1, n_frames):
            prev = raw[(i - 1) * frame_size : i * frame_size]
            curr = raw[i * frame_size : (i + 1) * frame_size]
            diff = sum(abs(a - b) for a, b in zip(prev, curr))
            motion = diff / frame_size
            motions.append(motion)

        return motions
    finally:
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)


def smooth(values: list[float], window: int = SMOOTH_WINDOW) -> list[float]:
    """Moving average smoothing."""
    result = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


# ---------------------------------------------------------------------------
# Transition detection
# ---------------------------------------------------------------------------

def find_transitions(
    motions: list[float],
    interval: int = SAMPLE_INTERVAL,
    video_duration: float = 0,
) -> tuple[float, float, float, float]:
    """Find H1 kickoff, halftime start, halftime end, H2 kickoff.

    Strategy:
    1. Compute smoothed motion profile
    2. Find the longest sustained low-motion period in the middle 60% of the video
       (this is halftime)
    3. H1 kickoff = first sustained high-motion period before halftime
    4. H2 kickoff = first sustained high-motion period after halftime

    Returns (h1_kickoff_sec, ht_start_sec, ht_end_sec, h2_kickoff_sec).
    """
    sm = smooth(motions)
    n = len(sm)

    # Compute threshold: median of all motion values
    sorted_m = sorted(sm)
    median = sorted_m[n // 2]
    low_threshold = median * 0.65  # Below this = "low motion"

    # --- Find halftime gap ---
    # Search in the middle 60% of the video
    search_start = int(n * 0.20)
    search_end = int(n * 0.80)

    # Find longest run of below-threshold motion
    best_run_start = search_start
    best_run_len = 0
    run_start: Optional[int] = None

    for i in range(search_start, search_end):
        if sm[i] < low_threshold:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len > best_run_len:
                    best_run_len = run_len
                    best_run_start = run_start
                run_start = None

    # Check final run
    if run_start is not None:
        run_len = search_end - run_start
        if run_len > best_run_len:
            best_run_len = run_len
            best_run_start = run_start

    ht_start_idx = best_run_start
    ht_end_idx = best_run_start + best_run_len

    # Convert to seconds (motion[i] corresponds to time (i+1)*interval)
    ht_start_sec = (ht_start_idx + 1) * interval
    ht_end_sec = (ht_end_idx + 1) * interval

    # --- Find H1 kickoff ---
    # Look for first sustained high-motion period in first 20% of video
    high_threshold = median * 0.85
    h1_kickoff_idx = 0
    consecutive = 0
    required = 4  # 4 consecutive high-motion windows = 2 min of play

    for i in range(min(ht_start_idx, int(n * 0.20))):
        if sm[i] >= high_threshold:
            consecutive += 1
            if consecutive >= required:
                h1_kickoff_idx = i - required + 1
                break
        else:
            consecutive = 0

    h1_kickoff_sec = (h1_kickoff_idx + 1) * interval

    # --- Find H2 kickoff ---
    # Look for first sustained high-motion period after halftime
    h2_kickoff_idx = ht_end_idx
    consecutive = 0

    for i in range(ht_end_idx, n):
        if sm[i] >= high_threshold:
            consecutive += 1
            if consecutive >= required:
                h2_kickoff_idx = i - required + 1
                break
        else:
            consecutive = 0

    h2_kickoff_sec = (h2_kickoff_idx + 1) * interval

    return h1_kickoff_sec, ht_start_sec, ht_end_sec, h2_kickoff_sec


# ---------------------------------------------------------------------------
# GT event parsing
# ---------------------------------------------------------------------------

EVENT_MAP = {
    ("Shots & Goals", "Goals"): "goal",
    ("Shots & Goals", "Shots On Target"): "shot_on_target",
    ("Shots & Goals", "Shots Off Target"): "shot_off_target",
    ("Shots & Goals", "Blocked Shots"): "shot_blocked",
    ("Saves", "Catches"): "save_catch",
    ("Saves", "Parries"): "save_parry",
    ("Set Pieces", "Corners"): "corner_kick",
    ("Set Pieces", "Goal Kicks"): "goal_kick",
    ("Set Pieces", "Throw-Ins"): "throw_in",
    ("Set Pieces", "Freekicks"): "free_kick",
    ("Set Pieces", "Penalty Kicks"): "penalty",
}


@dataclass
class GTEvent:
    event_time_ms: int
    half: int
    event_label: str


def parse_json_events(json_path: Path) -> list[GTEvent]:
    with open(json_path) as f:
        data = json.load(f)
    events = []
    for entry in data.get("data", []):
        half = 1 if entry.get("period_name", "").startswith("1st") else 2
        for ev in entry.get("events", []):
            outcome = ev.get("property", {}).get(
                "Outcome", ev.get("property", {}).get("Type", "")
            )
            key = (ev.get("event_name", ""), outcome)
            if key in EVENT_MAP:
                events.append(GTEvent(
                    event_time_ms=entry.get("event_time", 0),
                    half=half,
                    event_label=EVENT_MAP[key],
                ))
    return events


def get_video_duration(mp4_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------

def calibrate_game(game_dir: Path, game_num: str, verbose: bool = False) -> dict:
    mp4_files = list(game_dir.glob("*.mp4"))
    if not mp4_files:
        return {"error": "No MP4 found"}
    mp4_path = mp4_files[0]

    json_files = sorted(game_dir.glob("*.json"))
    if not json_files:
        return {"error": "No JSON found"}

    all_events = []
    for jp in json_files:
        all_events.extend(parse_json_events(jp))

    h1_events = sorted([e for e in all_events if e.half == 1], key=lambda e: e.event_time_ms)
    h2_events = sorted([e for e in all_events if e.half == 2], key=lambda e: e.event_time_ms)

    if not h1_events or not h2_events:
        return {"error": f"Missing events: H1={len(h1_events)}, H2={len(h2_events)}"}

    video_duration = get_video_duration(mp4_path)

    # First and last events in match clock
    h1_first_sec = h1_events[0].event_time_ms / 1000.0
    h1_last_sec = h1_events[-1].event_time_ms / 1000.0
    h2_first_sec = h2_events[0].event_time_ms / 1000.0
    h2_last_sec = h2_events[-1].event_time_ms / 1000.0

    if verbose:
        print(f"  Video: {mp4_path.name} ({video_duration / 60:.1f} min)")
        print(f"  Events: H1={len(h1_events)}, H2={len(h2_events)}")
        print(f"  H1 match clock: {h1_first_sec / 60:.1f}-{h1_last_sec / 60:.1f} min")
        print(f"  H2 match clock: {h2_first_sec / 60:.1f}-{h2_last_sec / 60:.1f} min")
        print("  Extracting motion profile...")

    motions = extract_motion_profile(mp4_path)

    if verbose:
        print(f"  Got {len(motions)} motion samples ({len(motions) * SAMPLE_INTERVAL / 60:.0f} min)")

    h1_kickoff, ht_start, ht_end, h2_kickoff = find_transitions(
        motions, SAMPLE_INTERVAL, video_duration
    )

    if verbose:
        print(f"  H1 kickoff: {h1_kickoff / 60:.1f} min")
        print(f"  Halftime: {ht_start / 60:.1f} - {ht_end / 60:.1f} min "
              f"({(ht_end - ht_start) / 60:.1f} min gap)")
        print(f"  H2 kickoff: {h2_kickoff / 60:.1f} min")

    # Compute offsets
    # video_time = match_clock_sec + offset
    # At H1 kickoff (match clock ~0): h1_offset = h1_kickoff_video_sec
    # At H2 kickoff (match clock = h2_first_sec): h2_offset = h2_kickoff_video_sec - h2_first_sec
    # But match clock for H2 continues from H1 (e.g., starts at ~2700s = 45:00)

    h1_offset = h1_kickoff
    h2_offset = h2_kickoff - h2_first_sec

    if verbose:
        print(f"  Computed H1 offset: {h1_offset:.0f}s ({h1_offset / 60:.1f} min)")
        print(f"  Computed H2 offset: {h2_offset:.0f}s ({h2_offset / 60:.1f} min)")

    # Sanity checks
    # H2 offset should be larger than H1 offset (halftime gap adds time)
    # But both offsets represent: video_time = match_clock + offset
    # For H1, match_clock starts at ~0, so offset ≈ pre-game duration
    # For H2, match_clock starts at ~2700, so offset ≈ pre-game + halftime_gap - match_gap
    # The halftime video gap = ht_end - ht_start
    # The match clock gap = h2_first_sec - h1_last_sec (typically small, <120s)
    # So h2_offset ≈ h1_offset + (ht_video_gap - match_clock_gap)

    halftime_video_gap = ht_end - ht_start
    match_clock_gap = h2_first_sec - h1_last_sec
    expected_h2_offset = h1_offset + halftime_video_gap - match_clock_gap

    if verbose:
        print(f"  Halftime video gap: {halftime_video_gap / 60:.1f} min")
        print(f"  Match clock gap H1→H2: {match_clock_gap / 60:.1f} min")
        print(f"  Expected H2 offset (cross-check): {expected_h2_offset:.0f}s "
              f"(actual: {h2_offset:.0f}s, delta: {abs(h2_offset - expected_h2_offset):.0f}s)")

    # Anchor events for manual verification
    h1_anchor = next(
        (e for e in h1_events if e.event_label in ("throw_in", "corner_kick", "goal_kick")),
        h1_events[0],
    )
    h2_anchor = next(
        (e for e in h2_events if e.event_label in ("throw_in", "corner_kick", "goal_kick")),
        h2_events[0],
    )

    return {
        "mp4": mp4_path.name,
        "video_duration_sec": round(video_duration, 1),
        "h1_offset_sec": round(h1_offset, 1),
        "h2_offset_sec": round(h2_offset, 1),
        "h1_kickoff_video_sec": round(h1_kickoff, 1),
        "h2_kickoff_video_sec": round(h2_kickoff, 1),
        "halftime_start_sec": round(ht_start, 1),
        "halftime_end_sec": round(ht_end, 1),
        "halftime_gap_sec": round(ht_end - ht_start, 1),
        "h1_anchor_event_ms": h1_anchor.event_time_ms,
        "h1_anchor_event_label": h1_anchor.event_label,
        "h1_anchor_predicted_video_sec": round(
            h1_anchor.event_time_ms / 1000 + h1_offset, 1
        ),
        "h2_anchor_event_ms": h2_anchor.event_time_ms,
        "h2_anchor_event_label": h2_anchor.event_label,
        "h2_anchor_predicted_video_sec": round(
            h2_anchor.event_time_ms / 1000 + h2_offset, 1
        ),
        "notes": f"H1={len(h1_events)}, H2={len(h2_events)} events",
        "h1_anchor_video_sec": None,
        "h2_anchor_video_sec": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-calibrate game video offsets")
    parser.add_argument("--games-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("calibration.json"))
    parser.add_argument("--game", type=str, help="Single game number")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.games_dir.is_dir():
        print(f"Error: {args.games_dir} is not a directory")
        sys.exit(1)

    existing = {}
    if args.output.exists():
        with open(args.output) as f:
            existing = json.load(f)

    game_dirs = sorted(
        [d for d in args.games_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    if args.game:
        game_dirs = [d for d in game_dirs if d.name == args.game]
        if not game_dirs:
            print(f"Error: Game {args.game} not found")
            sys.exit(1)

    calibration = {}
    for game_dir in game_dirs:
        game_num = game_dir.name
        print(f"\n{'=' * 60}")
        print(f"Game {game_num}")
        print(f"{'=' * 60}")

        try:
            entry = calibrate_game(game_dir, game_num, verbose=True)
            calibration[game_num] = entry

            # Preserve manual overrides
            if game_num in existing:
                ex = existing[game_num]
                if ex.get("h1_anchor_video_sec") is not None:
                    entry["h1_anchor_video_sec"] = ex["h1_anchor_video_sec"]
                    entry["h1_offset_sec"] = round(
                        ex["h1_anchor_video_sec"] - entry["h1_anchor_event_ms"] / 1000, 1
                    )
                    print(f"  -> Manual H1 override: {ex['h1_anchor_video_sec']}s")
                if ex.get("h2_anchor_video_sec") is not None:
                    entry["h2_anchor_video_sec"] = ex["h2_anchor_video_sec"]
                    entry["h2_offset_sec"] = round(
                        ex["h2_anchor_video_sec"] - entry["h2_anchor_event_ms"] / 1000, 1
                    )
                    print(f"  -> Manual H2 override: {ex['h2_anchor_video_sec']}s")

            if entry.get("error"):
                print(f"  ERROR: {entry['error']}")
            else:
                print(f"\n  Result: H1={entry['h1_offset_sec']:.0f}s, "
                      f"H2={entry['h2_offset_sec']:.0f}s, "
                      f"HT={entry['halftime_gap_sec'] / 60:.1f}min")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            calibration[game_num] = {"error": str(e)}

    merged = {**existing, **calibration}
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Written to {args.output}")
    print(f"\n{'Game':>4}  {'H1 Off':>8}  {'H2 Off':>8}  {'HT Gap':>7}")
    print("-" * 40)
    for gn in sorted(calibration.keys(), key=lambda x: int(x)):
        e = calibration[gn]
        if e.get("error"):
            print(f"{gn:>4}  ERROR: {e['error']}")
        else:
            print(f"{gn:>4}  {e['h1_offset_sec']:>6.0f}s  {e['h2_offset_sec']:>6.0f}s  "
                  f"{e['halftime_gap_sec'] / 60:>5.1f}m")


if __name__ == "__main__":
    main()
