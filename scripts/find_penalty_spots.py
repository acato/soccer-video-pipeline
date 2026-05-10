"""Find static penalty-spot locations per video by clustering YOLO ball detections.

The penalty spot is a small white circular pitch mark that the existing soccer-tuned
YOLO systematically misclassifies as "ball" from a high+wide camera. There are exactly
2 per video (one per goal), each at a fixed pixel coordinate across the entire match.

This script samples random play-time frames, runs YOLO at probe-ceiling settings,
aggregates all ball-class detection coordinates into a coarse grid, and picks the
two highest-count cells as the penalty-spot locations.

Output: penalty_spots.json keyed by game_id, value = [[x_norm, y_norm], ...]
        (typically 2 entries per game).

Usage:
    ~/quant-env/bin/python ~/find_penalty_spots.py \
        --output /mnt/transit/soccer-finetune/yolo_ball_v9_raw/penalty_spots.json \
        --frames-per-game 200
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/usr/bin/ffprobe"

YOLO_MODEL_DEFAULT = "/home/aless/yolov8_soccer_uisikdag.pt"
BALL_CLASS_ID = 0

# Mirror of GAMES_DEFAULT from extract_and_label_balls.py
GAMES = {
    "game_19": "/mnt/transit/Games/19/1773631328983_seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-1aa48cf7-b31f-41c9-aa17-cef4b4c22267-1773632512.672041-encoded.mp4",
    "game_20": "/mnt/transit/Games/20/2026-04-18 Celtic - Reign GA 11.mp4",
    "game_21": "/mnt/transit/Games/21/2026-04-25 Eastern WA Surf - Reign GA11.mp4",
    "game_22": "/mnt/transit/Games/22/2026-04-26 Spokane Shadow - Reign GA11.mp4",
}


def video_duration(video: Path) -> float:
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
         str(video)],
        text=True,
    )
    return float(out.strip())


def extract_frame(video: Path, ts: float, out_path: Path) -> bool:
    try:
        subprocess.run([
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}", "-i", str(video),
            "-frames:v", "1", "-q:v", "2",
            "-y", str(out_path),
        ], check=True, timeout=30)
        return out_path.exists() and out_path.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def detect_ball_coords(model, jpeg_path: Path, imgsz: int, conf: float):
    """Return list of (x_norm, y_norm) ball detections (any confidence)."""
    import cv2
    img = cv2.imread(str(jpeg_path))
    if img is None:
        return []
    results = model([img], imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return []
    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None:
        return []
    classes = r.boxes.cls.cpu().numpy().astype(int)
    xywhn = r.boxes.xywhn.cpu().numpy()
    return [(float(xywh[0]), float(xywh[1]))
            for cls, xywh in zip(classes, xywhn) if int(cls) == BALL_CLASS_ID]


def find_top_clusters(coords: list, grid_n: int = 50, top_k: int = 2,
                      min_count: int = 5):
    """Bin coords into grid_n × grid_n grid; return top_k cells with most
    detections, as their CENTER coordinates."""
    if not coords:
        return []
    bins = Counter()
    for x, y in coords:
        gx = min(grid_n - 1, max(0, int(x * grid_n)))
        gy = min(grid_n - 1, max(0, int(y * grid_n)))
        bins[(gx, gy)] += 1
    out = []
    for (gx, gy), cnt in bins.most_common(top_k * 4):
        if cnt < min_count:
            break
        # Skip if too close to an already-selected cluster (within 5 grid cells)
        cx = (gx + 0.5) / grid_n
        cy = (gy + 0.5) / grid_n
        too_close = any(abs(cx - sx) < 5 / grid_n and abs(cy - sy) < 5 / grid_n
                        for sx, sy, _ in out)
        if too_close:
            continue
        out.append((cx, cy, cnt))
        if len(out) >= top_k:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True,
                    help="Output JSON path")
    ap.add_argument("--yolo-model", default=YOLO_MODEL_DEFAULT)
    ap.add_argument("--imgsz", type=int, default=2560)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--frames-per-game", type=int, default=200)
    ap.add_argument("--grid-n", type=int, default=50,
                    help="Grid resolution (cells per dim)")
    args = ap.parse_args()

    print(f"loading YOLO ({args.yolo_model})")
    from ultralytics import YOLO
    model = YOLO(args.yolo_model)

    spots = {}
    for gid, video_str in GAMES.items():
        video = Path(video_str)
        if not video.exists():
            print(f"  [skip] {gid}: video not found")
            continue
        try:
            dur = video_duration(video)
        except Exception as e:
            print(f"  [skip] {gid}: ffprobe failed: {e}")
            continue
        # Sample uniformly, skip first/last 60s buffer
        sample_lo = 60.0
        sample_hi = max(dur - 60.0, sample_lo + 1)
        ts_list = [sample_lo + i * (sample_hi - sample_lo) / args.frames_per_game
                   for i in range(args.frames_per_game)]
        print(f"\n=== {gid} (duration={dur:.0f}s, sampling {len(ts_list)} frames) ===")
        all_coords = []
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for i, ts in enumerate(ts_list):
                jpg = td / f"f{i:04d}.jpg"
                if not extract_frame(video, ts, jpg):
                    continue
                all_coords.extend(detect_ball_coords(model, jpg, args.imgsz, args.conf))
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(ts_list)}] coords so far: {len(all_coords)}")
        clusters = find_top_clusters(all_coords, grid_n=args.grid_n, top_k=2)
        print(f"  total ball detections: {len(all_coords)}")
        print(f"  top clusters: {clusters}")
        spots[gid] = [[c[0], c[1]] for c in clusters]

    Path(args.output).write_text(json.dumps(spots, indent=2))
    print(f"\nWrote {args.output}")
    print(json.dumps(spots, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
