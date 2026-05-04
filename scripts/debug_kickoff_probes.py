"""Debug: probe Rush GT-goal kickoff windows with YOLO at multiple confs.

For each GT goal timestamp in the Rush video, sample frames at +20/30/40/50s
and run YOLO with conf={0.15, 0.05, 0.03} × imgsz={640, 1024}. Report
ball_central, n_persons, half_imbalance per probe.

Goal: figure out why kickoff_verifier dropped 7/7 in Run 65 — is it ball
confidence, person count, half imbalance, or something else?
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["PATH"] = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:" + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

VIDEO = Path("/Users/aless/soccer-working/2026-02-07 - Rush - GA2008.mp4")
YOLO_MODEL = "/Users/aless/Downloads/soccer-video-pipeline/infra/models/yolov8_soccer_uisikdag.pt"
GT_GOALS = [802.9, 1064.1, 2330.2, 4398.4]
PROBE_OFFSETS = [20.0, 30.0, 40.0, 50.0]
CENTRAL_BOX = 0.10
BALL_CLASS = 0
PERSON_CLASSES = (1, 2, 3)


def extract_frame(ts: float, width: int = 1280) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        out = f.name
    try:
        subprocess.run([
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}", "-i", str(VIDEO),
            "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3",
            "-y", out,
        ], check=True)
        with open(out, "rb") as fp:
            return fp.read()
    finally:
        try: os.unlink(out)
        except OSError: pass


def yolo_read(model, jpeg: bytes, imgsz: int, conf: float) -> dict:
    import cv2
    import numpy as np
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    results = model([img], imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return {"err": "no_result"}
    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None:
        return {"err": "no_boxes"}
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    xywhn = r.boxes.xywhn.cpu().numpy()
    n_persons = 0
    person_xs = []
    n_balls = 0
    ball_central_any = False
    ball_min_central_dist = None
    for cls, c, xywh in zip(classes, confs, xywhn):
        cx, cy = float(xywh[0]), float(xywh[1])
        if int(cls) == BALL_CLASS:
            n_balls += 1
            d = max(abs(cx - 0.5), abs(cy - 0.5))
            if ball_min_central_dist is None or d < ball_min_central_dist:
                ball_min_central_dist = d
            if d <= CENTRAL_BOX:
                ball_central_any = True
        if int(cls) in PERSON_CLASSES:
            n_persons += 1
            person_xs.append(cx)
    half_imb = None
    if n_persons >= 2:
        l = sum(1 for x in person_xs if x < 0.5)
        half_imb = abs(l - (n_persons - l)) / n_persons
    return {
        "n_balls": n_balls, "ball_central": ball_central_any,
        "ball_min_d_to_center": ball_min_central_dist,
        "n_persons": n_persons, "half_imb": half_imb,
    }


def main() -> int:
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)
    settings = [
        ("conf=0.15 imgsz=640", 640, 0.15),
        ("conf=0.05 imgsz=640", 640, 0.05),
        ("conf=0.05 imgsz=1024", 1024, 0.05),
        ("conf=0.03 imgsz=1280", 1280, 0.03),
    ]
    for goal_t in GT_GOALS:
        print(f"\n========== Rush GT goal at t={goal_t}s ==========")
        for off in PROBE_OFFSETS:
            ts = goal_t + off
            jpg = extract_frame(ts)
            print(f"  -- probe t={ts:.1f}s (goal+{off:.0f}s) --")
            for label, imgsz, conf in settings:
                r = yolo_read(model, jpg, imgsz, conf)
                print(f"    [{label}] {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
