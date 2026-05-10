"""Dump ball-detection coordinate histogram per game for inspection."""
from __future__ import annotations

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

GAMES = {
    "game_19": "/mnt/transit/Games/19/1773631328983_seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-1aa48cf7-b31f-41c9-aa17-cef4b4c22267-1773632512.672041-encoded.mp4",
    "game_20": "/mnt/transit/Games/20/2026-04-18 Celtic - Reign GA 11.mp4",
    "game_21": "/mnt/transit/Games/21/2026-04-25 Eastern WA Surf - Reign GA11.mp4",
    "game_22": "/mnt/transit/Games/22/2026-04-26 Spokane Shadow - Reign GA11.mp4",
}


def video_duration(video):
    return float(subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        text=True,
    ).strip())


def detect_ball_coords(model, jpeg_path, imgsz=2560, conf=0.02):
    import cv2
    img = cv2.imread(str(jpeg_path))
    if img is None:
        return []
    results = model([img], imgsz=imgsz, conf=conf, verbose=False)
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return []
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()
    xywhn = results[0].boxes.xywhn.cpu().numpy()
    return [(float(xywh[0]), float(xywh[1]), float(c))
            for cls, c, xywh in zip(classes, confs, xywhn) if int(cls) == 0]


def main():
    from ultralytics import YOLO
    model = YOLO("/home/aless/yolov8_soccer_uisikdag.pt")
    for gid, video_str in GAMES.items():
        video = Path(video_str)
        if not video.exists():
            continue
        dur = video_duration(video)
        ts_list = [60 + i * (dur - 120) / 200 for i in range(200)]
        coords = []
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for i, ts in enumerate(ts_list):
                jpg = td / f"f{i:04d}.jpg"
                try:
                    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error",
                                   "-ss", f"{ts:.3f}", "-i", str(video),
                                   "-frames:v", "1", "-q:v", "2", "-y", str(jpg)],
                                   check=True, timeout=30)
                except Exception:
                    continue
                coords.extend(detect_ball_coords(model, jpg))
        print(f"\n=== {gid} ({len(coords)} ball detections) ===")
        # Coarse 20x20 grid (5%-bins)
        bins = Counter()
        for x, y, c in coords:
            bins[(int(x * 20), int(y * 20))] += 1
        print(f"Top 10 grid cells (20x20 bins, ~5% each):")
        for (gx, gy), n in bins.most_common(10):
            print(f"  cell ({gx*5:3d}%-{(gx+1)*5:3d}%, {gy*5:3d}%-{(gy+1)*5:3d}%): {n} detections")
        # Also: average coordinate (would be biased toward static FPs if dominant)
        if coords:
            mean_x = sum(c[0] for c in coords) / len(coords)
            mean_y = sum(c[1] for c in coords) / len(coords)
            print(f"Mean coord: ({mean_x:.3f}, {mean_y:.3f})")


if __name__ == "__main__":
    main()
