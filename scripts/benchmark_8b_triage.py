#!/usr/bin/env python3
"""Benchmark the 8B base model as a fast triage scanner on Rush game frames.

Usage:
    python scripts/benchmark_8b_triage.py [--interval 3] [--start 0] [--duration 600] [--width 640]

Extracts frames from the Rush game at regular intervals, sends them to the 8B model
for binary triage ("is something happening?"), and measures speed + quality.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

VLLM_URL = os.environ.get("VLLM_URL", "http://10.10.2.222:8000")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3-vl-8b")
VIDEO_PATH = os.environ.get(
    "VIDEO_PATH",
    "/Volumes/SoccerGames/2026-02-07 - Rush - GA2008.mp4",
)
VIDEO_OFFSET = 418.0  # game starts at 6:58

# Simple triage prompt — cast a wide net, favour recall
TRIAGE_PROMPT = """\
You are scanning a soccer match video frame by frame. Your job is to flag frames where \
a notable event is happening or about to happen.

Flag as TRUE if you see ANY of these:
- A goal being scored or celebration
- A save, catch, or parry by a goalkeeper
- A shot on goal or shot attempt
- A corner kick being taken
- A free kick being taken
- A throw-in
- A goal kick
- A penalty kick
- A kickoff or restart
- Players clustered near a goal (potential scoring chance)
- A goalkeeper making a diving/jumping action

Flag as FALSE if you see:
- Normal midfield play with no immediate action
- Players just passing in their own half
- Wide camera angle with no clear event
- Pre-game, half-time, or post-game scenes
- Replay graphics or scoreboard overlays

Respond with ONLY this JSON, no other text:
{"flag": true, "reason": "brief description"} or {"flag": false}"""

# Batched version for multiple frames
TRIAGE_BATCH_PROMPT = """\
You are scanning a soccer match video frame by frame. For each frame, decide if \
a notable soccer event is happening (goal, save, shot, corner, free kick, throw-in, \
goal kick, penalty, kickoff, goalkeeper action) or if it's just normal play.

Respond with ONLY a JSON array, one entry per frame:
[{{"frame": 0, "flag": true/false, "reason": "brief if true"}}, ...]"""


def extract_frames(video_path, start_sec, duration_sec, interval_sec, width):
    """Extract frames using ffmpeg, return list of (timestamp, jpeg_bytes)."""
    tmpdir = tempfile.mkdtemp(prefix="triage_bench_")
    end_sec = start_sec + duration_sec

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", video_path,
        "-vf", f"fps=1/{interval_sec},scale={width}:-1",
        "-q:v", "3",
        os.path.join(tmpdir, "frame_%05d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    frames = []
    for i, fname in enumerate(sorted(Path(tmpdir).glob("frame_*.jpg"))):
        t = start_sec + i * interval_sec
        frames.append((t, fname.read_bytes()))
    return frames, tmpdir


def call_vllm_single(frame_b64, prompt, timeout=30):
    """Send a single frame to vLLM for triage."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.1,
    }
    t0 = time.monotonic()
    resp = httpx.post(
        f"{VLLM_URL}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, elapsed, usage


def call_vllm_batch(frames_b64, prompt, timeout=60):
    """Send multiple frames in one call for batched triage."""
    content = []
    for i, fb64 in enumerate(frames_b64):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
        })
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    payload = {
        "model": VLLM_MODEL,
        "messages": messages,
        "max_tokens": 50 * len(frames_b64),
        "temperature": 0.1,
    }
    t0 = time.monotonic()
    resp = httpx.post(
        f"{VLLM_URL}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, elapsed, usage


def parse_flag(text):
    """Parse the model's JSON response into a boolean flag."""
    text = text.strip()
    # Try to extract JSON
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj.get("flag", False), obj.get("reason", "")
        elif isinstance(obj, list):
            return [(o.get("flag", False), o.get("reason", "")) for o in obj]
    except json.JSONDecodeError:
        # Try to find flag in text
        if '"flag": true' in text.lower() or '"flag":true' in text.lower():
            return True, text
        return False, f"PARSE_ERROR: {text[:100]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between frames")
    parser.add_argument("--start", type=float, default=418.0, help="Video start time (sec)")
    parser.add_argument("--duration", type=float, default=300.0, help="Duration to scan (sec)")
    parser.add_argument("--width", type=int, default=640, help="Frame width in pixels")
    parser.add_argument("--batch", type=int, default=0, help="Batch size (0=single frame mode)")
    args = parser.parse_args()

    print(f"=== 8B Triage Benchmark ===")
    print(f"Video: {VIDEO_PATH}")
    print(f"Interval: {args.interval}s  Start: {args.start}s  Duration: {args.duration}s  Width: {args.width}px")
    print(f"Mode: {'batch=' + str(args.batch) if args.batch else 'single frame'}")
    print()

    # Check vLLM is up
    try:
        r = httpx.get(f"{VLLM_URL}/v1/models", timeout=5)
        models = [m["id"] for m in r.json()["data"]]
        print(f"vLLM models: {models}")
    except Exception as e:
        print(f"ERROR: vLLM not reachable at {VLLM_URL}: {e}")
        sys.exit(1)

    print(f"\nExtracting frames...")
    frames, tmpdir = extract_frames(VIDEO_PATH, args.start, args.duration, args.interval, args.width)
    print(f"Extracted {len(frames)} frames to {tmpdir}")
    print(f"Frame size: ~{len(frames[0][1])//1024}KB per JPEG")

    results = []
    total_tokens = 0
    total_time = 0.0

    if args.batch > 0:
        # Batched mode
        for batch_start in range(0, len(frames), args.batch):
            batch = frames[batch_start : batch_start + args.batch]
            b64s = [base64.b64encode(f[1]).decode() for _, f in batch]
            try:
                text, elapsed, usage = call_vllm_batch(b64s, TRIAGE_BATCH_PROMPT)
                total_time += elapsed
                total_tokens += usage.get("total_tokens", 0)
                parsed = parse_flag(text)
                if isinstance(parsed, list):
                    for i, (flag, reason) in enumerate(parsed):
                        if batch_start + i < len(batch):
                            t = batch[i][0]
                            game_t = t - VIDEO_OFFSET
                            results.append({"video_sec": t, "game_sec": game_t, "flag": flag, "reason": reason})
                            marker = "***" if flag else "   "
                            print(f"  {marker} {game_t:7.1f}s ({int(game_t//60):02d}:{game_t%60:04.1f})  flag={flag}  {reason[:60]}")
                else:
                    flag, reason = parsed
                    print(f"  Batch {batch_start}: single response, flag={flag}")
                print(f"  [batch {batch_start}-{batch_start+len(batch)-1}: {elapsed:.1f}s, {usage.get('total_tokens',0)} tokens]")
            except Exception as e:
                print(f"  ERROR batch {batch_start}: {e}")
    else:
        # Single frame mode
        for i, (t, jpeg_bytes) in enumerate(frames):
            game_t = t - VIDEO_OFFSET
            b64 = base64.b64encode(jpeg_bytes).decode()
            try:
                text, elapsed, usage = call_vllm_single(b64, TRIAGE_PROMPT)
                total_time += elapsed
                total_tokens += usage.get("total_tokens", 0)
                flag, reason = parse_flag(text)
                results.append({"video_sec": t, "game_sec": game_t, "flag": flag, "reason": reason, "elapsed": elapsed})
                marker = "***" if flag else "   "
                print(f"  {marker} {game_t:7.1f}s ({int(game_t//60):02d}:{game_t%60:04.1f})  flag={flag}  {elapsed:.1f}s  {reason[:60]}")
            except Exception as e:
                print(f"  ERR {game_t:7.1f}s: {e}")
                results.append({"video_sec": t, "game_sec": game_t, "flag": False, "reason": f"ERROR: {e}", "elapsed": 0})

    # Summary
    flagged = [r for r in results if r["flag"]]
    print(f"\n=== Summary ===")
    print(f"Frames scanned: {len(results)}")
    print(f"Flagged: {len(flagged)} ({100*len(flagged)/max(len(results),1):.0f}%)")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg per frame: {total_time/max(len(results),1):.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput: {len(results)/max(total_time,0.01):.1f} frames/sec")
    print(f"\nFlagged timestamps (game time):")
    for r in flagged:
        gt = r["game_sec"]
        print(f"  {int(gt//60):02d}:{gt%60:04.1f}  {r['reason'][:80]}")

    # Save results
    out_path = "/tmp/triage_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
