#!/usr/bin/env python3
"""Extract the 5 frames a single-pass VLM window was shown, as JPEGs on disk.

Reproduces the sampling logic from dual_pass_detector.py:
    center = (start + end) / 2
    half_span = (end - start) / 2
    interval = max(1.0, (end - start) / n_frames)
Then caps to n_frames using the same index striding.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def extract_frames(video: str, win_start: float, win_end: float,
                   n_frames: int, out_dir: Path, tag: str):
    center = (win_start + win_end) / 2
    half_span = (win_end - win_start) / 2
    interval = max(1.0, (win_end - win_start) / n_frames)
    start = max(0.0, center - half_span)
    end = center + half_span

    # Build candidate timestamps like FrameSampler.sample_range for small ranges.
    # sample_range uses _extract_batch only if expected_count > 10, otherwise
    # single-shot calls. For n_frames=5, interval=3, window=15 → expected=6 → single-shot.
    # Either way, let's compute 6 candidate timestamps then cap to 5 via index striding
    # (same as dual_pass_detector.py:639-642).
    expected = int((end - start) / interval) + 1
    candidates = [start + i * interval for i in range(expected)]
    if len(candidates) > n_frames:
        s = len(candidates) / n_frames
        candidates = [candidates[int(i * s)] for i in range(n_frames)]

    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    saved = []
    for i, t in enumerate(candidates):
        out = out_dir / f"{tag}_f{i}_t{t:.1f}s.jpg"
        cmd = [
            ffmpeg_bin, "-ss", f"{t}", "-i", video,
            "-vframes", "1", "-q:v", "3",
            "-vf", "scale=640:-1",  # downscale to 640w so the bundle is small
            "-y", str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {tag} t={t}: {r.stderr[-300:]}")
        else:
            saved.append((t, out))
            print(f"  {tag} f{i}  t={t:.1f}s -> {out.name}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--win-start", type=float, required=True)
    ap.add_argument("--win-end", type=float, required=True)
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True, help="Filename prefix")
    args = ap.parse_args()
    extract_frames(args.video, args.win_start, args.win_end,
                   args.n_frames, Path(args.out_dir), args.tag)


if __name__ == "__main__":
    main()
