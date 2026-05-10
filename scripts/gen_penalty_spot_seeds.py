"""Generate reference frames per game with a percent-grid overlay so the
user can eyeball penalty-spot coordinates.

Output: /mnt/transit/soccer-finetune/yolo_ball_v9_raw/penalty_spot_seeds/
        game_19_t<sec>.jpg
        game_20_t<sec>.jpg
        game_21_t<sec>.jpg
        game_22_t<sec>.jpg
3 candidates per game (early-1H, mid-1H, mid-2H) — pick the best one for
penalty-spot visibility (wide angle, both boxes in frame).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

OUT_DIR = Path("/mnt/transit/soccer-finetune/yolo_ball_v9_raw/penalty_spot_seeds")

# 8 candidate timestamps per game — varied across both halves to maximise the
# chance of catching frames where each penalty spot is in view (the camera
# tracks play, so each box appears at different times).
GAMES = {
    "game_19": dict(
        video="/mnt/transit/Games/19/1773631328983_seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-1aa48cf7-b31f-41c9-aa17-cef4b4c22267-1773632512.672041-encoded.mp4",
        timestamps=[300.0, 900.0, 1800.0, 2400.0, 3500.0, 4500.0, 5400.0, 6300.0],
    ),
    "game_20": dict(
        video="/mnt/transit/Games/20/2026-04-18 Celtic - Reign GA 11.mp4",
        timestamps=[500.0, 1100.0, 2000.0, 2800.0, 3900.0, 4800.0, 5700.0, 6600.0],
    ),
    "game_21": dict(
        video="/mnt/transit/Games/21/2026-04-25 Eastern WA Surf - Reign GA11.mp4",
        timestamps=[600.0, 1300.0, 2200.0, 3000.0, 4400.0, 5300.0, 6100.0, 7000.0],
    ),
    "game_22": dict(
        video="/mnt/transit/Games/22/2026-04-26 Spokane Shadow - Reign GA11.mp4",
        timestamps=[300.0, 900.0, 1700.0, 2400.0, 3500.0, 4400.0, 5400.0, 6500.0],
    ),
}


def extract_frame(video: Path, ts: float, out_path: Path) -> bool:
    try:
        subprocess.run([
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}", "-i", str(video),
            "-frames:v", "1", "-q:v", "2", "-y", str(out_path),
        ], check=True, timeout=30)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def overlay_grid(jpeg_path: Path, out_path: Path) -> None:
    import cv2
    img = cv2.imread(str(jpeg_path))
    if img is None:
        return
    h, w = img.shape[:2]

    # Translucent grid: yellow at 5%, magenta at 10% (with labels)
    overlay = img.copy()
    for pct in range(5, 100, 5):
        p = pct / 100.0
        if pct % 10 == 0:
            color = (255, 0, 255)  # magenta — major
            thick = 2
        else:
            color = (0, 255, 255)  # yellow — minor
            thick = 1
        cv2.line(overlay, (int(p * w), 0), (int(p * w), h), color, thick)
        cv2.line(overlay, (0, int(p * h)), (w, int(p * h)), color, thick)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Major-grid labels (every 10%) — black outline + white text
    for pct in range(0, 101, 10):
        p = pct / 100.0
        # Top edge label
        x_text = max(0, int(p * w) - 12)
        cv2.putText(img, f"{pct}", (x_text, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, f"{pct}", (x_text, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        # Left edge label
        y_text = min(h - 5, max(20, int(p * h) + 8))
        cv2.putText(img, f"{pct}", (4, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, f"{pct}", (4, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for gid, cfg in GAMES.items():
        video = Path(cfg["video"])
        if not video.exists():
            print(f"  [skip] {gid}: video missing")
            continue
        for ts in cfg["timestamps"]:
            raw = OUT_DIR / f"_raw_{gid}_t{int(ts):05d}.jpg"
            if not extract_frame(video, ts, raw):
                print(f"  [skip] {gid}@{ts}s: extract failed")
                continue
            out = OUT_DIR / f"{gid}_t{int(ts):05d}.jpg"
            overlay_grid(raw, out)
            raw.unlink(missing_ok=True)
            written.append(out)
            print(f"  wrote {out}")
    print(f"\nGenerated {len(written)} reference frames at {OUT_DIR}")
    print("Open each, pick the BEST one per game (wide angle, both penalty boxes visible).")
    print("Read penalty-spot coords off the percent grid; format response as:")
    print("  game_19: (0.30, 0.55) and (0.70, 0.50)   # left spot, right spot")
    print("  game_20: (...) and (...)")
    print("  game_21: (...) and (...)")
    print("  game_22: (...) and (...)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
