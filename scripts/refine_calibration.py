#!/usr/bin/env python3
"""
Refine calibration offsets using VLM to identify anchor events.

Takes rough calibration estimates (from auto_calibrate.py) and refines them
by extracting frames near predicted anchor timestamps and asking the VLM
to confirm whether the expected event (throw-in, corner, etc.) is visible.

Searches in a ±3 minute window around the rough estimate at 10-second intervals.
The VLM confirms the first match, giving us a precise anchor timestamp.

Usage:
    python scripts/refine_calibration.py \
        --calibration calibration.json \
        --games-dir /Volumes/transit/Games \
        --vllm-url http://10.10.2.222:8000 \
        --output calibration_refined.json

    # Single game:
    python scripts/refine_calibration.py \
        --calibration calibration.json \
        --games-dir /Volumes/transit/Games \
        --game 11
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import requests


SEARCH_RADIUS = 180  # ±3 minutes around rough estimate
STEP_SEC = 10        # 10-second search intervals
FRAME_W = 768        # Match production resolution


def extract_frame_b64(mp4_path: Path, timestamp_sec: float, width: int = FRAME_W) -> str:
    """Extract a single frame at the given timestamp, return as base64 JPEG."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", str(timestamp_sec),
        "-i", str(mp4_path),
        "-frames:v", "1",
        "-vf", f"scale={width}:-1",
        "-f", "image2", "-c:v", "mjpeg",
        "-q:v", "5", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=15)
    if result.returncode != 0 or not result.stdout:
        return ""
    return base64.b64encode(result.stdout).decode()


def ask_vlm(
    vllm_url: str,
    frame_b64: str,
    event_label: str,
    model: str = "Qwen/Qwen3-VL-32B-Instruct-FP8",
) -> tuple[bool, str]:
    """Ask VLM if the expected event is visible in the frame.

    Returns (is_match, reasoning).
    """
    event_descriptions = {
        "throw_in": "a throw-in (player holding ball overhead at the sideline, about to throw it in)",
        "corner_kick": "a corner kick (ball placed at the corner flag arc, player about to kick it)",
        "goal_kick": "a goal kick (ball placed in the six-yard box, goalkeeper about to kick it)",
        "free_kick": "a free kick (ball placed on the ground, players lined up in a wall)",
        "goal": "a goal being scored or celebrated (players celebrating, ball in net, kickoff restart)",
    }

    description = event_descriptions.get(event_label, f"a {event_label}")

    prompt = (
        f"Look at this frame from a soccer match (sideline camera, ~50m away). "
        f"Is this frame showing {description}? "
        f"Answer with EXACTLY this JSON: "
        f'{{\"match\": true/false, \"reasoning\": \"brief explanation\"}}'
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 150,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(
            f"{vllm_url}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]

        # Parse JSON from response
        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)
        return data.get("match", False), data.get("reasoning", "")
    except Exception as e:
        return False, f"VLM error: {e}"


def refine_half(
    mp4_path: Path,
    rough_video_sec: float,
    event_label: str,
    vllm_url: str,
    verbose: bool = False,
) -> tuple[float | None, str]:
    """Search for the anchor event near the rough estimate.

    Returns (precise_video_sec, reasoning) or (None, error_msg).
    """
    search_start = max(0, rough_video_sec - SEARCH_RADIUS)
    search_end = rough_video_sec + SEARCH_RADIUS

    if verbose:
        print(f"    Searching for {event_label} in "
              f"{search_start / 60:.1f}-{search_end / 60:.1f} min "
              f"(center: {rough_video_sec / 60:.1f} min)")

    timestamps = []
    t = search_start
    while t <= search_end:
        timestamps.append(t)
        t += STEP_SEC

    for t in timestamps:
        frame_b64 = extract_frame_b64(mp4_path, t)
        if not frame_b64:
            continue

        is_match, reasoning = ask_vlm(vllm_url, frame_b64, event_label)

        if verbose and is_match:
            mm = int(t) // 60
            ss = int(t) % 60
            print(f"    MATCH at {mm}:{ss:02d} ({t:.0f}s): {reasoning[:80]}")

        if is_match:
            return t, reasoning

    return None, "No match found in search window"


def main():
    parser = argparse.ArgumentParser(description="Refine calibration with VLM")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--games-dir", type=Path, required=True)
    parser.add_argument("--vllm-url", type=str, default="http://10.10.2.222:8000")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--game", type=str, help="Single game number")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.calibration  # Overwrite in place

    with open(args.calibration) as f:
        calibration = json.load(f)

    for game_num, entry in sorted(calibration.items(), key=lambda x: int(x[0])):
        if args.game and game_num != args.game:
            continue

        if entry.get("error"):
            print(f"Game {game_num}: SKIPPED (error)")
            continue

        # Skip if already manually calibrated
        if (entry.get("h1_anchor_video_sec") is not None
                and entry.get("h2_anchor_video_sec") is not None):
            print(f"Game {game_num}: already calibrated (manual anchors)")
            continue

        mp4_path = args.games_dir / game_num / entry["mp4"]
        if not mp4_path.exists():
            print(f"Game {game_num}: MP4 not found at {mp4_path}")
            continue

        print(f"\n{'=' * 50}")
        print(f"Game {game_num}: {entry['mp4']}")
        print(f"{'=' * 50}")

        # Refine H1 anchor
        if entry.get("h1_anchor_video_sec") is None:
            h1_rough = entry["h1_anchor_predicted_video_sec"]
            h1_label = entry["h1_anchor_event_label"]
            print(f"  H1: searching for {h1_label} near {h1_rough / 60:.1f} min...")

            h1_precise, h1_reason = refine_half(
                mp4_path, h1_rough, h1_label, args.vllm_url,
                verbose=args.verbose,
            )

            if h1_precise is not None:
                entry["h1_anchor_video_sec"] = round(h1_precise, 1)
                entry["h1_offset_sec"] = round(
                    h1_precise - entry["h1_anchor_event_ms"] / 1000, 1
                )
                print(f"  H1 anchor: {h1_precise / 60:.1f} min → offset={entry['h1_offset_sec']:.0f}s")
            else:
                print(f"  H1: {h1_reason}")

        # Refine H2 anchor
        if entry.get("h2_anchor_video_sec") is None:
            h2_rough = entry["h2_anchor_predicted_video_sec"]
            h2_label = entry["h2_anchor_event_label"]
            print(f"  H2: searching for {h2_label} near {h2_rough / 60:.1f} min...")

            h2_precise, h2_reason = refine_half(
                mp4_path, h2_rough, h2_label, args.vllm_url,
                verbose=args.verbose,
            )

            if h2_precise is not None:
                entry["h2_anchor_video_sec"] = round(h2_precise, 1)
                entry["h2_offset_sec"] = round(
                    h2_precise - entry["h2_anchor_event_ms"] / 1000, 1
                )
                print(f"  H2 anchor: {h2_precise / 60:.1f} min → offset={entry['h2_offset_sec']:.0f}s")
            else:
                print(f"  H2: {h2_reason}")

    with open(args.output, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nRefined calibration written to {args.output}")

    # Summary
    print(f"\n{'Game':>4}  {'H1 Off':>8}  {'H2 Off':>8}  {'Status'}")
    print("-" * 45)
    for gn in sorted(calibration.keys(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        e = calibration[gn]
        if e.get("error"):
            print(f"{gn:>4}  ERROR")
        else:
            status = "refined" if e.get("h1_anchor_video_sec") else "rough"
            print(f"{gn:>4}  {e['h1_offset_sec']:>6.0f}s  {e['h2_offset_sec']:>6.0f}s  {status}")


if __name__ == "__main__":
    main()
