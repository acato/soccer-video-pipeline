"""Probe ball detection across (model, imgsz, conf) settings on new-venue frames.

Sample frames at GT event timestamps from Game 20 — at these moments the ball
is visible by definition (shot, throw_in, corner_kick, catch, goal_kick, etc.).
Run YOLO at multiple settings; tally:
  - n_balls_total: any ball detection (recall proxy)
  - n_balls_central: ball in central 30% (frame center is rough action zone)
  - mean confs

Compare configs to find the cheapest setting that gives >70% ball recall.
That's the threshold for ball_crop to actually work.

Usage on llm:
  python probe_ball_detection.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

VIDEO = Path("/mnt/transit/Games/20/2026-04-18 Celtic - Reign GA 11.mp4")
GT_H1 = Path("/mnt/transit/Games/20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_1st Half.json")
GT_H2 = Path("/mnt/transit/Games/20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_2nd Half.json")
VIDEO_OFFSET = 124.0
HALF2_VIDEO = 3554.0
HALF2_GAME_OFFSET = 2400.0

YOLO_MODEL = "/home/aless/yolov8_soccer_uisikdag.pt"
BALL_CLASS = 0
PERSON_CLASSES = (1, 2, 3)

# Event types where ball is visible during a brief window — sample all
BALL_EVENT_NAMES = {
    "Shots & Goals", "Saves/Catches", "Saves/Parries",
    "Set Pieces/Corners", "Set Pieces/Goal Kicks", "Set Pieces/Throw-Ins",
    "Set Pieces/Freekicks", "Goals Conceded",
}

# Probe matrix
SETTINGS = [
    ("baseline_prod",   640,  0.15),  # what production runs at
    ("medium_low_conf", 1280, 0.05),  # last week's tuning
    ("hi_low_conf",     1920, 0.05),
    ("hi_lower_conf",   1920, 0.02),
    ("xhi_lower_conf",  2560, 0.02),
]


def gt_event_video_times() -> list[tuple[float, str]]:
    out = []
    for half_idx, fp in enumerate((GT_H1, GT_H2)):
        d = json.loads(fp.read_text())
        for entry in d.get("data", []):
            game_sec = entry.get("event_time", 0) / 1000.0
            for ev in entry.get("events", []):
                name = ev.get("event_name", "")
                if name not in BALL_EVENT_NAMES:
                    continue
                if half_idx == 0:
                    video_sec = game_sec + VIDEO_OFFSET
                else:
                    video_sec = (game_sec - HALF2_GAME_OFFSET) + HALF2_VIDEO
                out.append((video_sec, name))
                break  # one event-line per entry is enough
    return out


def extract_frame(video: Path, ts: float) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        out = f.name
    try:
        subprocess.run([
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}", "-i", str(video),
            "-frames:v", "1", "-q:v", "2",  # full-resolution JPEG
            "-y", out,
        ], check=True)
        with open(out, "rb") as fp:
            return fp.read()
    finally:
        try: os.unlink(out)
        except OSError: pass


def yolo_read(model, jpeg: bytes, imgsz: int, conf: float):
    import cv2
    import numpy as np
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    results = model([img], imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return {"n_balls": 0, "n_balls_central": 0, "max_ball_conf": 0.0,
                "n_persons": 0}
    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None:
        return {"n_balls": 0, "n_balls_central": 0, "max_ball_conf": 0.0,
                "n_persons": 0}
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    xywhn = r.boxes.xywhn.cpu().numpy()
    n_balls = 0
    n_balls_central = 0
    max_ball_conf = 0.0
    n_persons = 0
    for cls, c, xywh in zip(classes, confs, xywhn):
        cls_int = int(cls)
        cx, cy = float(xywh[0]), float(xywh[1])
        if cls_int == BALL_CLASS:
            n_balls += 1
            max_ball_conf = max(max_ball_conf, float(c))
            if abs(cx - 0.5) <= 0.30 and abs(cy - 0.5) <= 0.30:
                n_balls_central += 1
        if cls_int in PERSON_CLASSES:
            n_persons += 1
    return {"n_balls": n_balls, "n_balls_central": n_balls_central,
            "max_ball_conf": max_ball_conf, "n_persons": n_persons}


def main() -> int:
    events = gt_event_video_times()
    # Subsample to ~50 if many; keep mix
    if len(events) > 50:
        step = len(events) // 50
        events = events[::step][:50]
    print(f"sampling {len(events)} GT events from Game 20")
    print(f"  YOLO model: {YOLO_MODEL}")
    print(f"  configs: {[(n, i, c) for n, i, c in SETTINGS]}")
    print()

    # Pre-extract frames once (single-pass ffmpeg, then each setting reads them)
    print("[extract] pulling 1 frame per event ...")
    t0 = time.time()
    jpegs = []
    for i, (ts, name) in enumerate(events):
        try:
            jpegs.append((ts, name, extract_frame(VIDEO, ts)))
        except subprocess.CalledProcessError as e:
            print(f"  [skip] t={ts:.1f}: {e}")
    print(f"[extract] {len(jpegs)} frames in {time.time()-t0:.1f}s")
    print()

    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)

    # Probe each setting
    summary: dict[str, dict] = {}
    for label, imgsz, conf in SETTINGS:
        print(f"--- {label} (imgsz={imgsz}, conf={conf}) ---")
        t0 = time.time()
        n_with_ball = 0
        n_with_central = 0
        max_confs = []
        for ts, name, jpg in jpegs:
            r = yolo_read(model, jpg, imgsz=imgsz, conf=conf)
            if r["n_balls"] > 0:
                n_with_ball += 1
                max_confs.append(r["max_ball_conf"])
            if r["n_balls_central"] > 0:
                n_with_central += 1
        elapsed = time.time() - t0
        n = len(jpegs)
        summary[label] = {
            "imgsz": imgsz, "conf": conf,
            "n_frames": n,
            "n_with_ball": n_with_ball,
            "ball_recall": n_with_ball / max(1, n),
            "n_with_central": n_with_central,
            "central_rate": n_with_central / max(1, n),
            "mean_max_conf": sum(max_confs) / max(1, len(max_confs)),
            "elapsed_sec": round(elapsed, 1),
        }
        print(f"  ball_recall={n_with_ball}/{n} = {n_with_ball*100/max(1,n):.0f}%   "
              f"central={n_with_central}/{n} ({n_with_central*100/max(1,n):.0f}%)   "
              f"mean_conf={summary[label]['mean_max_conf']:.3f}   "
              f"{elapsed:.1f}s ({elapsed/n:.2f}s/frame)")

    print()
    print("=== SUMMARY ===")
    print(f"{'config':<22}{'imgsz':>7}{'conf':>7}{'recall':>10}{'central':>10}{'mean_conf':>11}{'sec/frame':>11}")
    for label, s in summary.items():
        print(f"{label:<22}{s['imgsz']:>7}{s['conf']:>7}"
              f"{s['ball_recall']*100:>9.0f}%{s['central_rate']*100:>9.0f}%"
              f"{s['mean_max_conf']:>11.3f}"
              f"{s['elapsed_sec']/s['n_frames']:>11.2f}")
    print()
    print("Decision rule: ball_recall ≥ 70% on this sample → ball_crop will work.")
    print("If best config still < 70%, need (3) manual annotation + retrain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
