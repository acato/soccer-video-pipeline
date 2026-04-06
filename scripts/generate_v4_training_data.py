#!/usr/bin/env python3
"""
Generate v4 training data for fine-tuned soccer event classifier.

Augmentation strategy (addresses pipeline timestamp noise + free_kick fallback):
1. Shifted timestamps: ±5s, ±10s, ±15s offsets from GT events
   - If GT event is still within window → keep original label
   - If GT event falls outside window → relabel as "none" (hard negative)
2. Variable window widths: 4s, 6s, 8s, 10s, 12s (all subsampled to 8 frames)
3. Extra "none" samples: random moments + hard negatives 2-5s from events
4. Target: ~5000-7000 total samples, 40-45% "none" class

Runs on LLM server where videos and existing training data are located.

Usage:
    python3 generate_v4_training_data.py \
        --input /mnt/transit/soccer-finetune/stage2_train_augmented.jsonl \
        --output /mnt/transit/soccer-finetune/stage2_v4_train.jsonl \
        --frames-dir /mnt/transit/soccer-finetune/frames_v4 \
        --games-dir /mnt/transit/Games
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional


# --- Frame extraction ---

def extract_frames(
    video_path: str,
    centre_sec: float,
    window_sec: float,
    num_frames: int,
    out_dir: Path,
    prefix: str,
    width: int = 768,
) -> list[str]:
    """Extract evenly-spaced frames from a window centred at centre_sec."""
    start = max(0, centre_sec - window_sec / 2)
    duration = window_sec

    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{prefix}_%03d.jpg")

    # Extract at 2 FPS then subsample
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-vf", f"fps=2,scale={width}:-2",
        "-q:v", "5",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        return []

    frame_files = sorted(glob.glob(str(out_dir / f"{prefix}_*.jpg")))
    if not frame_files:
        return []

    # Subsample to num_frames
    if len(frame_files) > num_frames:
        step = len(frame_files) / num_frames
        indices = [int(i * step) for i in range(num_frames)]
        selected = [frame_files[i] for i in indices]
        # Remove unselected frames
        for f in frame_files:
            if f not in selected:
                os.unlink(f)
        frame_files = selected
    elif len(frame_files) < num_frames:
        # Not enough frames — pad by duplicating last
        while len(frame_files) < num_frames:
            frame_files.append(frame_files[-1])

    return frame_files


def build_prompt(num_frames: int, window_sec: float) -> str:
    """Build the classification prompt matching training format."""
    return f"""You are analyzing {num_frames} frames sampled from a {window_sec:.1f}-second clip of a soccer match recorded by a SIDELINE CAMERA at ~50 metres from the field.

CAMERA LIMITATIONS -- at this distance:
- You CANNOT see whether the goalkeeper's hands touched the ball
- You CANNOT see the ball cross the goal line with certainty
- You CAN see player positions, formations, celebrations, and restart patterns

CLASSIFICATION STRATEGY -- what happens AFTER the action is the best signal:
- Kickoff restart (players at center circle) = goal was scored
- Corner kick restart = goalkeeper parried the shot
- Goal kick restart with no GK save action = shot went wide/over
- Goalkeeper holding ball and distributing = goalkeeper caught it
- Play continues normally = nothing significant

Classify the MAIN event visible in these frames. Choose ONE:
- "goal": A goal was scored -- you see celebration OR kickoff restart at center circle
- "save": Goalkeeper stopped a shot -- GK touched/blocked/caught the ball, preventing a goal
- "shot_on_target": Shot toward goal that was saved (but not clearly a catch or parry)
- "shot_off_target": Shot that missed -- followed by goal kick (ball went wide/over)
- "shot_blocked": Shot blocked by a defender (not the GK)
- "corner_kick": Ball placed at corner flag arc, kicked into the box
- "goal_kick": Ball kicked from six-yard box, no preceding shot visible
- "throw_in": Player holds ball overhead at sideline, throws it in
- "free_kick": Ball placed on ground, kicked from a stoppage after a foul
- "penalty": One shooter vs goalkeeper from penalty spot, others outside box
- "kickoff": Kick-off from center circle (start of half or after goal)
- "none": Normal play, nothing significant, or cannot determine

Respond with EXACTLY this JSON (no other text):
{{"event": "<type>", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


def build_response(label: str) -> str:
    """Build a training response matching the expected format."""
    reasonings = {
        "goal": "After the shot, players celebrate with arms raised and run toward each other. Teams walk back to the center circle for a kickoff restart.",
        "save": "The goalkeeper catches and holds the ball after a shot, preventing a goal.",
        "shot_on_target": "A shot is taken toward goal. The goalkeeper makes a save.",
        "shot_off_target": "A shot is taken but misses the goal. A goal kick is awarded.",
        "shot_blocked": "A shot is taken but blocked by a defender before reaching the goal.",
        "corner_kick": "Ball placed at corner flag arc. Players gathered in the penalty area for the delivery.",
        "goal_kick": "The ball is kicked from the six-yard box with no preceding shot visible.",
        "throw_in": "A player holds the ball overhead at the sideline and throws it back into play.",
        "free_kick": "The ball is placed on the ground and kicked from a stoppage after a foul.",
        "penalty": "One shooter faces the goalkeeper from the penalty spot. Other players wait outside the box.",
        "kickoff": "Players lined up at the center circle for a kick-off restart.",
        "none": "Normal play continues with no significant event visible.",
    }
    conf = 0.95 if label != "none" else 0.85
    reason = reasonings.get(label, "Normal play, nothing significant.")
    return json.dumps({"event": label, "confidence": conf, "reasoning": reason})


def build_record(
    frames: list[str],
    label: str,
    game: str,
    video_ts: float,
    window_sec: float,
    augmentation: str,
) -> dict:
    """Build a training JSONL record."""
    num_frames = len(frames)
    prompt = build_prompt(num_frames, window_sec)
    query = "<image>" * num_frames + "\n" + prompt

    return {
        "query": query,
        "response": build_response(label),
        "images": frames,
        "_meta": {
            "game": game,
            "label": label,
            "video_ts": video_ts,
            "augmentation": augmentation,
            "window_sec": window_sec,
        },
    }


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return float(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="Generate v4 training data")
    parser.add_argument("--input", required=True, help="Existing training JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--frames-dir", required=True, help="Directory for extracted frames")
    parser.add_argument("--games-dir", required=True, help="Directory containing game folders")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--none-target-pct", type=float, default=0.42,
                        help="Target percentage for 'none' class (default 42%%)")
    args = parser.parse_args()

    random.seed(args.seed)
    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # --- Load existing training data ---
    print("Loading existing training data...")
    existing = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                existing.append(json.loads(line))

    print(f"  Loaded {len(existing)} existing samples")

    # --- Collect event metadata per game ---
    from collections import defaultdict
    game_events = defaultdict(list)  # game -> [(video_ts, label)]
    for rec in existing:
        meta = rec.get("_meta", {})
        game = meta.get("game", "?")
        vts = meta.get("video_ts", 0)
        label = meta.get("label", "none")
        aug = meta.get("augmentation", "")
        if aug == "original":  # Only use original events, not existing augmentations
            game_events[game].append((vts, label))

    # --- Find video files per game ---
    game_videos = {}
    for game in game_events:
        vids = glob.glob(os.path.join(args.games_dir, game, "*.mp4"))
        if vids:
            game_videos[game] = vids[0]

    print(f"  Found videos for {len(game_videos)}/{len(game_events)} games")

    # --- Generate augmented samples ---
    all_records = []
    sample_id = 0

    # Keep all original training records
    for rec in existing:
        all_records.append(rec)
    print(f"  Kept {len(existing)} original samples")

    # --- Augmentation 1: Shifted timestamps ---
    # For each original event, generate shifted variants
    shifts = [-15, -10, -5, 5, 10, 15]
    window_widths = [4.0, 6.0, 8.0, 10.0, 12.0]

    for game, events in game_events.items():
        if game not in game_videos:
            continue
        video = game_videos[game]

        try:
            duration = get_video_duration(video)
        except Exception:
            print(f"  Skipping game {game}: can't get duration")
            continue

        for vts, label in events:
            # Pick 3 random shifts + 2 random window widths per event
            selected_shifts = random.sample(shifts, min(3, len(shifts)))
            selected_widths = random.sample(window_widths, 2)

            for shift in selected_shifts:
                new_centre = vts + shift
                if new_centre < 3 or new_centre > duration - 3:
                    continue

                # Determine label: is the GT event within the new window?
                window = 6.0  # default window
                half = window / 2
                event_in_window = abs(vts - new_centre) <= half

                new_label = label if event_in_window else "none"

                prefix = f"g{game}_shift{shift}_{sample_id}"
                frame_files = extract_frames(
                    video, new_centre, window, args.num_frames,
                    frames_dir / f"game_{game}", prefix,
                )
                if not frame_files:
                    continue

                rec = build_record(
                    frame_files, new_label, game, new_centre, window,
                    f"shift_{shift}s",
                )
                all_records.append(rec)
                sample_id += 1

            # Variable window widths at original timestamp
            for width in selected_widths:
                if width == 6.0:
                    continue  # Already have original at 6s

                prefix = f"g{game}_w{width}_{sample_id}"
                frame_files = extract_frames(
                    video, vts, width, args.num_frames,
                    frames_dir / f"game_{game}", prefix,
                )
                if not frame_files:
                    continue

                rec = build_record(
                    frame_files, label, game, vts, width,
                    f"window_{width}s",
                )
                all_records.append(rec)
                sample_id += 1

        print(f"  Game {game}: generated shifted/width variants")

    # --- Augmentation 2: Hard negatives (near events) ---
    for game, events in game_events.items():
        if game not in game_videos:
            continue
        video = game_videos[game]

        try:
            duration = get_video_duration(video)
        except Exception:
            continue

        for vts, label in events:
            # Extract frames 2-5s before and after each event
            for offset in [-4, -3, 3, 4]:
                neg_centre = vts + offset
                if neg_centre < 3 or neg_centre > duration - 3:
                    continue

                # Only label as "none" if the window doesn't contain the event
                window = 4.0  # narrow window for hard negatives
                half = window / 2
                event_in_window = abs(vts - neg_centre) <= half
                if event_in_window:
                    continue  # Skip if event is still visible

                prefix = f"g{game}_hardneg_{offset}_{sample_id}"
                frame_files = extract_frames(
                    video, neg_centre, window, args.num_frames,
                    frames_dir / f"game_{game}", prefix,
                )
                if not frame_files:
                    continue

                rec = build_record(
                    frame_files, "none", game, neg_centre, window,
                    f"hard_negative_{offset}s",
                )
                all_records.append(rec)
                sample_id += 1

        print(f"  Game {game}: generated hard negatives")

    # --- Augmentation 3: Random "none" samples ---
    # Calculate how many more "none" samples we need
    from collections import Counter
    label_counts = Counter(
        r.get("_meta", {}).get("label", "none") for r in all_records
    )
    total = len(all_records)
    current_none = label_counts.get("none", 0)
    current_none_pct = current_none / total if total > 0 else 0

    target_total = max(total, 5000)
    target_none = int(target_total * args.none_target_pct)
    needed_none = max(0, target_none - current_none)

    print(f"\n  Current: {total} samples, {current_none} none ({current_none_pct:.1%})")
    print(f"  Target: {target_total} samples, {target_none} none ({args.none_target_pct:.0%})")
    print(f"  Need {needed_none} more 'none' samples")

    if needed_none > 0:
        games_list = list(game_videos.keys())
        per_game = needed_none // len(games_list) + 1

        for game in games_list:
            video = game_videos[game]
            try:
                duration = get_video_duration(video)
            except Exception:
                continue

            event_times = [vts for vts, _ in game_events[game]]
            samples_this_game = 0

            for _ in range(per_game * 3):  # Try extra in case some fail
                if samples_this_game >= per_game:
                    break

                # Random timestamp, avoiding ±10s of any event
                t = random.uniform(30, duration - 30)
                too_close = any(abs(t - et) < 10 for et in event_times)
                if too_close:
                    continue

                window = random.choice([4.0, 6.0, 8.0, 10.0])
                prefix = f"g{game}_none_{sample_id}"
                frame_files = extract_frames(
                    video, t, window, args.num_frames,
                    frames_dir / f"game_{game}", prefix,
                )
                if not frame_files:
                    continue

                rec = build_record(
                    frame_files, "none", game, t, window,
                    "random_negative",
                )
                all_records.append(rec)
                sample_id += 1
                samples_this_game += 1

            print(f"  Game {game}: added {samples_this_game} random 'none' samples")

    # --- Final stats ---
    random.shuffle(all_records)

    final_counts = Counter(
        r.get("_meta", {}).get("label", "none") for r in all_records
    )
    print(f"\n=== Final dataset: {len(all_records)} samples ===")
    for label, count in sorted(final_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_records)
        print(f"  {label}: {count} ({pct:.1f}%)")

    # --- Write output ---
    with open(args.output, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
