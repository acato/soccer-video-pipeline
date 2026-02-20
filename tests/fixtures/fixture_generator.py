#!/usr/bin/env python3
"""
Generate synthetic test fixture videos using FFmpeg.
No real match footage required — all fixtures use color/noise sources.

Usage:
    python tests/fixtures/fixture_generator.py [--output-dir tests/fixtures/videos]
"""
import argparse
import subprocess
import sys
from pathlib import Path


FIXTURES = [
    {
        "name": "sample_60s_4k.mp4",
        "description": "60-second synthetic 4K video",
        "width": 3840, "height": 2160, "fps": 30, "duration": 60,
        "color": "0x1a7a1a",  # soccer green
    },
    {
        "name": "sample_30s_1080p.mp4",
        "description": "30-second 1080p video for fast integration tests",
        "width": 1920, "height": 1080, "fps": 30, "duration": 30,
        "color": "0x1a7a1a",
    },
    {
        "name": "sample_10s_720p.mp4",
        "description": "10-second 720p clip for unit tests",
        "width": 1280, "height": 720, "fps": 30, "duration": 10,
        "color": "0x2d8a2d",
    },
    {
        "name": "sample_gk_sequence_10s.mp4",
        "description": "Simulated GK save sequence (red-tinted to simulate different jersey)",
        "width": 1280, "height": 720, "fps": 30, "duration": 10,
        "color": "0x8a1a1a",  # red-ish to simulate GK jersey
    },
    {
        "name": "sample_h265_30s.mp4",
        "description": "H.265 encoded 30s clip for codec compatibility tests",
        "width": 1920, "height": 1080, "fps": 30, "duration": 30,
        "color": "0x1a7a1a",
        "codec": "libx265",
    },
]


def generate_fixture(fixture: dict, output_dir: Path, force: bool = False) -> Path:
    output_path = output_dir / fixture["name"]
    if output_path.exists() and not force:
        print(f"  SKIP {fixture['name']} (already exists)")
        return output_path

    codec = fixture.get("codec", "libx264")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={fixture['color']}:size={fixture['width']}x{fixture['height']}:rate={fixture['fps']}",
        "-f", "lavfi",
        "-i", "sine=frequency=440:sample_rate=48000",
        "-t", str(fixture["duration"]),
        "-c:v", codec, "-crf", "28", "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "64k",
        str(output_path),
    ]

    print(f"  GEN  {fixture['name']} ({fixture['description']})")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"       → {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test fixtures")
    parser.add_argument("--output-dir", default="tests/fixtures/videos", type=Path)
    parser.add_argument("--force", action="store_true", help="Regenerate even if exists")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check ffmpeg
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: ffmpeg not found in PATH", file=sys.stderr)
        sys.exit(1)

    print(f"Generating fixtures in {args.output_dir}/")
    for fixture in FIXTURES:
        generate_fixture(fixture, args.output_dir, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
