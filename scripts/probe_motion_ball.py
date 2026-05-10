"""Probe: motion-based ball detection on new-venue frames.

Approach:
  1. Sample a 3-frame window at (t-1.0s, t, t+1.0s) for each test moment.
  2. Estimate camera pan/zoom between frames via ORB+RANSAC homography.
  3. Warp t-1 and t+1 to t's coordinate frame; compute pixel diff.
  4. AND the diffs (object must move across BOTH directions) → motion mask.
  5. AND with green-field mask (HSV) — ball is on pitch.
  6. Find small bright blobs (size 3-30 px diameter, brightness > 180).
  7. Among candidates, pick the one with highest brightness * motion-energy.
  8. Output: candidate (x_norm, y_norm) or None.

Saves visualized output to /mnt/transit/.../motion_probe_out/ for human review.
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

# Reuse the same GT setup we already used for YOLO probes
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

# Events where the ball is in motion (vs static placement like a goal kick setup)
MOVING_BALL_EVENTS = {
    "Shots & Goals", "Saves/Catches", "Saves/Parries",
    "Goals Conceded",
}

OUT_ROOT = Path("/mnt/transit/soccer-finetune/yolo_ball_v9_raw/motion_probe_out")


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


def estimate_homography(prev: np.ndarray, curr: np.ndarray) -> np.ndarray | None:
    """Return 3x3 homography that warps prev → curr. None on failure."""
    g_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g_curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    # Lower-resolution feature matching for speed
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
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H


def green_field_mask(bgr: np.ndarray) -> np.ndarray:
    """Return uint8 mask (255 = green pitch) using HSV thresholds."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40])
    upper = np.array([85, 255, 255])
    return cv2.inRange(hsv, lower, upper)


def detect_motion_ball(prev: np.ndarray, curr: np.ndarray, nxt: np.ndarray):
    """Find a moving small bright object on the green pitch.
    Returns (cx_norm, cy_norm, score, bbox_px) or None."""
    H_prev = estimate_homography(prev, curr)
    H_nxt = estimate_homography(nxt, curr)
    if H_prev is None or H_nxt is None:
        # Fall back to no warping (still works if camera barely moves)
        prev_warp = prev
        nxt_warp = nxt
    else:
        h, w = curr.shape[:2]
        prev_warp = cv2.warpPerspective(prev, H_prev, (w, h))
        nxt_warp = cv2.warpPerspective(nxt, H_nxt, (w, h))

    diff_back = cv2.absdiff(curr, prev_warp)
    diff_fwd = cv2.absdiff(curr, nxt_warp)
    diff_back_g = cv2.cvtColor(diff_back, cv2.COLOR_BGR2GRAY)
    diff_fwd_g = cv2.cvtColor(diff_fwd, cv2.COLOR_BGR2GRAY)

    # AND of motion: must move across both directions
    motion = cv2.min(diff_back_g, diff_fwd_g)
    _, motion = cv2.threshold(motion, 12, 255, cv2.THRESH_BINARY)
    # Light morphological dilation so a small bright moving spot doesn't
    # split into a ring around the ball
    motion = cv2.dilate(motion, np.ones((3, 3), np.uint8), iterations=1)
    # Field mask: ball must be on green pitch
    field = green_field_mask(curr)
    motion = cv2.bitwise_and(motion, field)
    # Brightness mask: ball is white-ish (tunable)
    gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    bright = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)[1]
    candidates_mask = cv2.bitwise_and(motion, bright)

    # Connected components → blob candidates
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(candidates_mask)
    h, w = curr.shape[:2]
    best = None
    best_score = -1
    for i in range(1, n):  # skip background (0)
        x, y, bw, bh, area = stats[i]
        # Size constraint: ball at distance is small
        if area < 4 or area > 200:
            continue
        if bw < 3 or bw > 25 or bh < 3 or bh > 25:
            continue
        # Aspect roughly square
        if bw / max(1, bh) > 2.5 or bh / max(1, bw) > 2.5:
            continue
        cx, cy = centroids[i]
        # Motion energy at this region
        motion_energy = int(motion[y:y+bh, x:x+bw].sum())
        score = motion_energy + area * 5  # weight area + motion
        if score > best_score:
            best_score = score
            best = (cx / w, cy / h, score, (x, y, x+bw, y+bh))
    return best


def annotate(curr: np.ndarray, ball: tuple | None, out_path: Path):
    img = curr.copy()
    if ball is not None:
        _, _, score, (x1, y1, x2, y2) = ball
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"BALL? score={score}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(img, "no motion-ball found", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    probes: list[Probe] = []
    for gid, cfg in GAMES.items():
        ev = gt_event_video_times(gid, cfg)
        # Cap per-game
        ev = ev[:30]
        probes.extend(ev)
        print(f"  {gid}: {len(ev)} probes")
    print(f"total: {len(probes)} probes\n")

    n_found = 0
    n_none = 0
    t0 = time.time()
    for i, p in enumerate(probes):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ok_prev = extract_frame(p.video_path, p.video_ts - 0.3, td / "p.jpg")
            ok_curr = extract_frame(p.video_path, p.video_ts,        td / "c.jpg")
            ok_nxt  = extract_frame(p.video_path, p.video_ts + 0.3, td / "n.jpg")
            if not (ok_prev and ok_curr and ok_nxt):
                print(f"  [skip] {p.game_id}@{p.video_ts:.0f}: extract failed")
                continue
            prev = cv2.imread(str(td / "p.jpg"))
            curr = cv2.imread(str(td / "c.jpg"))
            nxt = cv2.imread(str(td / "n.jpg"))

        ball = detect_motion_ball(prev, curr, nxt)
        if ball is not None:
            n_found += 1
        else:
            n_none += 1

        out = OUT_ROOT / f"{p.game_id}_t{int(p.video_ts):05d}.jpg"
        annotate(curr, ball, out)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(probes)}] found:{n_found} none:{n_none}")

    print(f"\n=== done in {time.time()-t0:.0f}s ===")
    print(f"  found:  {n_found}/{len(probes)} = {n_found*100/max(1,len(probes)):.0f}%")
    print(f"  none:   {n_none}/{len(probes)}")
    print(f"\nVisual review at {OUT_ROOT}")
    print("Open random samples and confirm the red bbox is on a real ball.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
