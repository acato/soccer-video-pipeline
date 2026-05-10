"""Hybrid ball detection probe v3 — YOLO + motion + player-density prior.

Algorithm:
  For each test moment, sample 3 frames at (t-0.3, t, t+0.3).
  Run YOLO at 2560/0.02 on the center frame to get ball candidates +
  person bboxes.

  Compute motion mask (camera-pan-compensated AND-of-diffs across the
  3 frames) restricted to green-pitch + brightness gate.

  Build candidate pool:
    A. YOLO ball detections that have motion_energy > 0 (kills static
       phantom-ball points like penalty spots, box-edge corners,
       line intersections).
    B. Pure-motion blobs (small bright moving objects) inside the
       player-cluster region — recovers shadow-casting balls YOLO
       blanks on.

  Player centroid = mean of all person-bbox centers.

  Score each candidate:
    + motion_energy / 100             (motion strength)
    + yolo_conf * 5                   (boost for YOLO-confirmed)
    - dist_to_player_centroid * 3     (closer to action = better)
    - 5  if inside any person bbox    (sock/shoe penalty)

  Pick highest-scoring; that's the ball.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

YOLO_MODEL = "/home/aless/yolov8_soccer_uisikdag.pt"
BALL_CLASS_ID = 0
PERSON_CLASSES = (1, 2, 3)

GAMES = {
    "game_20": dict(
        video="/mnt/transit/Games/20/2026-04-18 Celtic - Reign GA 11.mp4",
        gt_h1="/mnt/transit/Games/20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_1st Half.json",
        gt_h2="/mnt/transit/Games/20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_2nd Half.json",
        video_offset=124.0, half2_video_start=3554.0, half2_game_offset=2400.0,
    ),
    "game_22": dict(
        video="/mnt/transit/Games/22/2026-04-26 Spokane Shadow - Reign GA11.mp4",
        gt_h1="/mnt/transit/Games/22/2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_1st Half.json",
        gt_h2="/mnt/transit/Games/22/2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_2nd Half.json",
        video_offset=90.0, half2_video_start=2900.0, half2_game_offset=2700.0,
    ),
}

MOVING_BALL_EVENTS = {
    "Shots & Goals", "Saves/Catches", "Saves/Parries", "Goals Conceded",
}

OUT_ROOT = Path("/mnt/transit/soccer-finetune/yolo_ball_v9_raw/hybrid_probe_out")


@dataclass
class Probe:
    game_id: str
    video_path: Path
    video_ts: float
    event_name: str


def gt_event_video_times(game_id: str, cfg: dict) -> list[Probe]:
    out: list[Probe] = []
    for half_idx, fp in enumerate((cfg["gt_h1"], cfg["gt_h2"])):
        d = json.loads(Path(fp).read_text())
        for entry in d.get("data", []):
            game_sec = entry.get("event_time", 0) / 1000.0
            for ev in entry.get("events", []):
                name = ev.get("event_name", "")
                if name not in MOVING_BALL_EVENTS:
                    continue
                if half_idx == 0:
                    video_sec = game_sec + cfg["video_offset"]
                else:
                    video_sec = (game_sec - cfg["half2_game_offset"]) + cfg["half2_video_start"]
                out.append(Probe(game_id=game_id, video_path=Path(cfg["video"]),
                                 video_ts=video_sec, event_name=name))
                break
    return out


def extract_frame(video: Path, ts: float, out_path: Path) -> bool:
    try:
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error",
                        "-ss", f"{ts:.3f}", "-i", str(video),
                        "-frames:v", "1", "-q:v", "2",
                        "-y", str(out_path)],
                       check=True, timeout=30)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def estimate_homography(prev: np.ndarray, curr: np.ndarray):
    g_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g_curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    s = 0.5
    g_prev_s = cv2.resize(g_prev, None, fx=s, fy=s)
    g_curr_s = cv2.resize(g_curr, None, fx=s, fy=s)
    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(g_prev_s, None)
    k2, d2 = orb.detectAndCompute(g_curr_s, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda m: m.distance)[:200]
    if len(matches) < 20:
        return None
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches]) / s
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches]) / s
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H


def green_field_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))


def run_yolo(model, img_bgr: np.ndarray, imgsz=2560, conf=0.02):
    results = model([img_bgr], imgsz=imgsz, conf=conf, verbose=False)
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return [], []
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()
    xywhn = results[0].boxes.xywhn.cpu().numpy()
    h, w = img_bgr.shape[:2]
    balls = []
    persons = []
    for cls, c, xywh in zip(classes, confs, xywhn):
        cx_n, cy_n = float(xywh[0]), float(xywh[1])
        bw_n, bh_n = float(xywh[2]), float(xywh[3])
        x1 = int((cx_n - bw_n/2) * w); y1 = int((cy_n - bh_n/2) * h)
        x2 = int((cx_n + bw_n/2) * w); y2 = int((cy_n + bh_n/2) * h)
        rec = (cx_n, cy_n, bw_n, bh_n, float(c), x1, y1, x2, y2)
        if int(cls) == BALL_CLASS_ID:
            balls.append(rec)
        elif int(cls) in PERSON_CLASSES:
            persons.append(rec)
    return balls, persons


def hybrid_detect(prev: np.ndarray, curr: np.ndarray, nxt: np.ndarray, model):
    h, w = curr.shape[:2]
    # ── Camera-motion-compensated frame diffs ──
    H_prev = estimate_homography(prev, curr)
    H_nxt = estimate_homography(nxt, curr)
    prev_warp = cv2.warpPerspective(prev, H_prev, (w, h)) if H_prev is not None else prev
    nxt_warp = cv2.warpPerspective(nxt, H_nxt, (w, h)) if H_nxt is not None else nxt
    diff_back = cv2.cvtColor(cv2.absdiff(curr, prev_warp), cv2.COLOR_BGR2GRAY)
    diff_fwd = cv2.cvtColor(cv2.absdiff(curr, nxt_warp), cv2.COLOR_BGR2GRAY)
    motion = cv2.min(diff_back, diff_fwd)
    motion = cv2.threshold(motion, 12, 255, cv2.THRESH_BINARY)[1]
    motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=1)
    field = green_field_mask(curr)
    motion_field = cv2.bitwise_and(motion, field)

    # ── YOLO ──
    balls_yolo, persons = run_yolo(model, curr)

    # Player centroid + cluster radius
    if persons:
        pcs = np.array([(p[0], p[1]) for p in persons])
        centroid = pcs.mean(axis=0)
        # spread = max pairwise dist; used as cluster radius proxy
    else:
        centroid = np.array([0.5, 0.5])

    def motion_energy_at(x, y, bw, bh):
        x1 = max(0, int((x - bw/2) * w))
        y1 = max(0, int((y - bh/2) * h))
        x2 = min(w, int((x + bw/2) * w))
        y2 = min(h, int((y + bh/2) * h))
        if x2 <= x1 or y2 <= y1:
            return 0
        # Expand by 5 px for partial-overlap blobs
        x1 = max(0, x1 - 5); y1 = max(0, y1 - 5)
        x2 = min(w, x2 + 5); y2 = min(h, y2 + 5)
        return int(motion_field[y1:y2, x1:x2].sum() / 255)  # count of motion px

    def inside_person_bbox(x_n, y_n):
        for p in persons:
            _, _, _, _, _, x1, y1, x2, y2 = p
            px = x_n * w; py = y_n * h
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
        return False

    def dist_to_centroid_norm(x_n, y_n):
        return float(np.hypot(x_n - centroid[0], y_n - centroid[1]))

    candidates = []
    # A. YOLO ball candidates with motion energy > 0
    for b in balls_yolo:
        cx_n, cy_n, bw_n, bh_n, conf, x1, y1, x2, y2 = b
        me = motion_energy_at(cx_n, cy_n, bw_n, bh_n)
        if me == 0:
            continue  # static phantom (penalty spot, line corner) — drop
        candidates.append({
            "source": "yolo+motion", "x_n": cx_n, "y_n": cy_n,
            "bw_n": bw_n, "bh_n": bh_n, "conf": conf, "motion_energy": me,
            "bbox": (x1, y1, x2, y2),
        })

    # B. Pure-motion blobs (small bright on field, moving)
    bright = cv2.threshold(cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY), 140, 255,
                           cv2.THRESH_BINARY)[1]
    motion_bright = cv2.bitwise_and(motion_field, bright)
    n_cc, _, stats, centroids = cv2.connectedComponentsWithStats(motion_bright)
    for i in range(1, n_cc):
        x, y, bw, bh, area = stats[i]
        if area < 4 or area > 200: continue
        if bw < 3 or bw > 25 or bh < 3 or bh > 25: continue
        if bw / max(1, bh) > 2.5 or bh / max(1, bw) > 2.5: continue
        cx, cy = centroids[i]
        cx_n, cy_n = cx / w, cy / h
        # Skip if already covered by a YOLO candidate
        if any(abs(c["x_n"] - cx_n) < 0.02 and abs(c["y_n"] - cy_n) < 0.02
               for c in candidates):
            continue
        candidates.append({
            "source": "motion", "x_n": cx_n, "y_n": cy_n,
            "bw_n": bw / w, "bh_n": bh / h,
            "conf": 0.0,
            "motion_energy": int(motion_field[y:y+bh, x:x+bw].sum() / 255),
            "bbox": (x, y, x+bw, y+bh),
        })

    if not candidates:
        return None, persons, centroid

    # ── Score ──
    for c in candidates:
        d_centroid = dist_to_centroid_norm(c["x_n"], c["y_n"])
        c["score"] = (
            c["motion_energy"] / 100.0
            + c["conf"] * 5.0
            - d_centroid * 3.0
            - (5.0 if inside_person_bbox(c["x_n"], c["y_n"]) else 0.0)
        )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0], persons, centroid


def annotate(curr: np.ndarray, ball, persons, centroid, out_path: Path):
    img = curr.copy()
    h, w = img.shape[:2]
    # Draw player bboxes (faint)
    for p in persons:
        _, _, _, _, _, x1, y1, x2, y2 = p
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
    # Player centroid
    cx_px = int(centroid[0] * w); cy_px = int(centroid[1] * h)
    cv2.circle(img, (cx_px, cy_px), 12, (255, 255, 0), 2)
    cv2.putText(img, "centroid", (cx_px+15, cy_px+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    # Ball bbox
    if ball is not None:
        x1, y1, x2, y2 = ball["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"BALL [{ball['source']}] s={ball['score']:.1f}"
        cv2.putText(img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(img, "no ball candidate", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    probes: list[Probe] = []
    for gid, cfg in GAMES.items():
        ev = gt_event_video_times(gid, cfg)
        ev = ev[:30]
        probes.extend(ev)
        print(f"  {gid}: {len(ev)} probes")
    print(f"total: {len(probes)} probes\n")

    print(f"loading YOLO ({YOLO_MODEL})")
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)

    n_found = 0
    n_none = 0
    t0 = time.time()
    for i, p in enumerate(probes):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ok_p = extract_frame(p.video_path, p.video_ts - 0.3, td / "p.jpg")
            ok_c = extract_frame(p.video_path, p.video_ts,        td / "c.jpg")
            ok_n = extract_frame(p.video_path, p.video_ts + 0.3, td / "n.jpg")
            if not (ok_p and ok_c and ok_n):
                continue
            prev = cv2.imread(str(td / "p.jpg"))
            curr = cv2.imread(str(td / "c.jpg"))
            nxt = cv2.imread(str(td / "n.jpg"))

        ball, persons, centroid = hybrid_detect(prev, curr, nxt, model)
        out = OUT_ROOT / f"{p.game_id}_t{int(p.video_ts):05d}.jpg"
        annotate(curr, ball, persons, centroid, out)
        if ball is not None:
            n_found += 1
        else:
            n_none += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(probes)}] found:{n_found} none:{n_none}")

    print(f"\n=== done in {time.time()-t0:.0f}s ===")
    print(f"  found: {n_found}/{len(probes)} = {n_found*100/max(1,len(probes)):.0f}%")
    print(f"  none:  {n_none}/{len(probes)}")
    print(f"\nVisual review at {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
