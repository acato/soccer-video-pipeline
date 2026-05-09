"""Auto-label ball positions across new-venue frames for v9 YOLO training.

Pipeline:
  1. Sample frames at GT event timestamps from each new-venue game (Games
     19, 20, 21, 22). Events with ball-visibility constraints (shots,
     throw-ins, corners, catches, etc.) are sampled — we know the ball
     SHOULD be visible at these moments.
  2. Run existing soccer-tuned YOLO at imgsz=2560 / conf=0.02 (the probe
     ceiling: 59% recall, but high-conf detections are likely correct).
  3. For each frame, save:
       a. Original JPEG (full resolution, used for v9 training)
       b. Annotated JPEG with bbox overlay (for human review)
       c. YOLO-format .txt label file (provisional — to be verified)
  4. Also sample frames where NO ball detected — possible "ball missed"
     cases. Human can promote some to hard-negative or correct-with-bbox.

Verification workflow (in README that this script writes):
  - Open `annotated/` directory (Finder, image viewer, etc.)
  - Delete annotated frames where the bbox is wrong (false positive on a
    helmet, ref's whistle, etc.)
  - Whatever remains in `annotated/` defines the verified set; companion
    .txt files in `labels/` constitute the v9 training labels.
  - Optionally: scan `no_detection/` folder for frames where the ball IS
    visible but YOLO missed it; add a label entry by hand.
  - Run `build_yolo_train_set.py` to produce ultralytics dataset layout.

Run on llm where YOLO + videos live:
    ~/quant-env/bin/python ~/extract_and_label_balls.py \
        --output-dir /mnt/transit/soccer-finetune/yolo_ball_v9_raw \
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
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

YOLO_MODEL_DEFAULT = "/home/aless/yolov8_soccer_uisikdag.pt"
BALL_CLASS_ID = 0
PERSON_CLASSES = (1, 2, 3)

# Events whose moment guarantees the ball is visible
BALL_EVENT_NAMES = {
    "Shots & Goals", "Saves/Catches", "Saves/Parries",
    "Set Pieces/Corners", "Set Pieces/Goal Kicks", "Set Pieces/Throw-Ins",
    "Set Pieces/Freekicks", "Goals Conceded",
}

# Per-game configs (from prepare_lora_dataset.py manifest + scripts/find_video_offsets results)
GAMES_DEFAULT = {
    "game_19": dict(
        video="/mnt/transit/Games/19/1773631328983_seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-seattle-reign-academy-2011-ga-vs-oregon-premier-2011-ga-1aa48cf7-b31f-41c9-aa17-cef4b4c22267-1773632512.672041-encoded.mp4",
        gt_h1="/mnt/transit/Games/19/2026-03-15_Seattle Reign 2011 GA (U15) vs Oregon Premier FC U15 (W)_1st Half.json",
        gt_h2="/mnt/transit/Games/19/2026-03-15_Seattle Reign 2011 GA (U15) vs Oregon Premier FC U15 (W)_2nd Half.json",
        video_offset=0.0, half2_video_start=2863.0, half2_game_offset=2400.0,
    ),
    "game_20": dict(
        video="/mnt/transit/Games/20/2026-04-18 Celtic - Reign GA 11.mp4",
        gt_h1="/mnt/transit/Games/20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_1st Half.json",
        gt_h2="/mnt/transit/Games/20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_2nd Half.json",
        video_offset=124.0, half2_video_start=3554.0, half2_game_offset=2400.0,
    ),
    "game_21": dict(
        video="/mnt/transit/Games/21/2026-04-25 Eastern WA Surf - Reign GA11.mp4",
        gt_h1="/mnt/transit/Games/21/2026-04-25_Seattle Reign 2011 GA (U15) vs Washington East Surf SC U15 (W)_1st Half.json",
        gt_h2="/mnt/transit/Games/21/2026-04-25_Seattle Reign 2011 GA (U15) vs Washington East Surf SC U15 (W)_2nd Half.json",
        video_offset=250.0, half2_video_start=4130.0, half2_game_offset=2400.0,
    ),
    "game_22": dict(
        video="/mnt/transit/Games/22/2026-04-26 Spokane Shadow - Reign GA11.mp4",
        gt_h1="/mnt/transit/Games/22/2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_1st Half.json",
        gt_h2="/mnt/transit/Games/22/2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_2nd Half.json",
        video_offset=90.0, half2_video_start=2900.0, half2_game_offset=2700.0,
    ),
}


@dataclass
class FrameSpec:
    game_id: str
    video_path: Path
    video_ts: float
    gt_event_name: str  # e.g. "Shots & Goals"


def gt_event_video_times(game_id: str, cfg: dict) -> list[FrameSpec]:
    out: list[FrameSpec] = []
    for half_idx, fp in enumerate((cfg["gt_h1"], cfg["gt_h2"])):
        d = json.loads(Path(fp).read_text())
        for entry in d.get("data", []):
            game_sec = entry.get("event_time", 0) / 1000.0
            for ev in entry.get("events", []):
                name = ev.get("event_name", "")
                if name not in BALL_EVENT_NAMES:
                    continue
                if half_idx == 0:
                    video_sec = game_sec + cfg["video_offset"]
                else:
                    video_sec = (game_sec - cfg["half2_game_offset"]) + cfg["half2_video_start"]
                out.append(FrameSpec(
                    game_id=game_id, video_path=Path(cfg["video"]),
                    video_ts=video_sec, gt_event_name=name,
                ))
                break  # one event per data entry
    return out


def extract_frame(video: Path, ts: float, out_path: Path) -> bool:
    """Extract a single full-resolution JPEG. Returns True on success."""
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


def detect_with_yolo(model, jpeg_path: Path, imgsz: int, conf: float):
    """Returns list of (class_id, x_norm, y_norm, w_norm, h_norm, conf) tuples."""
    import cv2
    img = cv2.imread(str(jpeg_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    results = model([img], imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return []
    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None:
        return []
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    xywhn = r.boxes.xywhn.cpu().numpy()
    out = []
    for cls, c, xywh in zip(classes, confs, xywhn):
        out.append((int(cls), float(xywh[0]), float(xywh[1]),
                    float(xywh[2]), float(xywh[3]), float(c), w, h))
    return out


def draw_overlay(jpeg_path: Path, dets: list, ball_class: int, out_path: Path):
    """Render a copy of the frame with bbox overlays on detected balls + persons."""
    import cv2
    img = cv2.imread(str(jpeg_path))
    if img is None:
        return
    h, w = img.shape[:2]
    for cls, cx, cy, bw, bh, conf, _, _ in dets:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        if cls == ball_class:
            color = (0, 0, 255)  # red BGR — BALL
            label = f"BALL {conf:.2f}"
            thickness = 3
        elif cls in PERSON_CLASSES:
            color = (0, 255, 0)  # green BGR — person
            label = ""
            thickness = 1
        else:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(img, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # Re-encode to keep output reasonably sized
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_yolo_label(dets: list, ball_class: int, out_path: Path):
    """Write only ball detections in YOLO format: <class> <cx> <cy> <w> <h>.
    Class 0 = ball (single-class for v9 fine-tune)."""
    lines = []
    for cls, cx, cy, bw, bh, conf, _, _ in dets:
        if cls != ball_class:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="Output root directory")
    ap.add_argument("--yolo-model", default=YOLO_MODEL_DEFAULT)
    ap.add_argument("--imgsz", type=int, default=2560,
                    help="YOLO inference imgsz (probe ceiling: 2560)")
    ap.add_argument("--conf", type=float, default=0.02,
                    help="YOLO confidence threshold")
    ap.add_argument("--frames-per-game", type=int, default=200,
                    help="Cap of frames sampled per game")
    ap.add_argument("--games", nargs="*", default=None,
                    help="Subset of game IDs (default: all 4 new-venue)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    (out / "annotated").mkdir(exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    (out / "no_detection").mkdir(exist_ok=True)

    # Build the candidate frame list
    games = args.games or list(GAMES_DEFAULT.keys())
    all_specs: list[FrameSpec] = []
    for gid in games:
        if gid not in GAMES_DEFAULT:
            print(f"unknown game: {gid}", file=sys.stderr)
            continue
        specs = gt_event_video_times(gid, GAMES_DEFAULT[gid])
        # Subsample to frames-per-game (evenly spaced)
        if len(specs) > args.frames_per_game:
            step = len(specs) / args.frames_per_game
            specs = [specs[int(i * step)] for i in range(args.frames_per_game)]
        all_specs.extend(specs)
        print(f"  {gid}: {len(specs)} frames")
    print(f"total: {len(all_specs)} frames\n")

    # Lazy-load YOLO
    print(f"loading YOLO ({args.yolo_model})")
    from ultralytics import YOLO
    model = YOLO(args.yolo_model)
    print(f"  imgsz={args.imgsz} conf={args.conf}\n")

    n_with_ball = 0
    n_no_ball = 0
    t0 = time.time()
    manifest_lines = []
    for i, spec in enumerate(all_specs):
        # Stable filename: gameNN_tNNNNNs (5-digit video time)
        name = f"{spec.game_id}_t{int(spec.video_ts):05d}"
        frame_path = out / "frames" / f"{name}.jpg"
        if not extract_frame(spec.video_path, spec.video_ts, frame_path):
            print(f"  [skip] frame extract failed: {name}")
            continue
        dets = detect_with_yolo(model, frame_path, args.imgsz, args.conf)
        ball_dets = [d for d in dets if d[0] == BALL_CLASS_ID]

        if ball_dets:
            # Save annotated viz + label
            draw_overlay(frame_path, dets, BALL_CLASS_ID,
                         out / "annotated" / f"{name}.jpg")
            write_yolo_label(dets, BALL_CLASS_ID,
                             out / "labels" / f"{name}.txt")
            n_with_ball += 1
        else:
            # Move frame to no_detection bucket so verifier can scan for misses
            shutil.copy(frame_path, out / "no_detection" / f"{name}.jpg")
            n_no_ball += 1

        manifest_lines.append({
            "name": name, "game_id": spec.game_id, "video_ts": spec.video_ts,
            "gt_event": spec.gt_event_name, "n_balls": len(ball_dets),
            "max_ball_conf": max((d[5] for d in ball_dets), default=0.0),
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(all_specs)}] {elapsed:.0f}s — "
                  f"ball: {n_with_ball}  no-ball: {n_no_ball}")

    elapsed = time.time() - t0
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest_lines)
    )

    print(f"\n=== DONE in {elapsed:.0f}s ===")
    print(f"frames extracted: {n_with_ball + n_no_ball}")
    print(f"  with ball detection: {n_with_ball} ({n_with_ball*100/max(1,n_with_ball+n_no_ball):.0f}%)")
    print(f"  no detection:        {n_no_ball}")
    print(f"\nOutputs:")
    print(f"  {out}/frames/        — full-res JPEGs")
    print(f"  {out}/annotated/     — bbox-overlay JPEGs FOR HUMAN REVIEW")
    print(f"  {out}/labels/        — YOLO-format labels (provisional)")
    print(f"  {out}/no_detection/  — frames where YOLO found no ball")
    print(f"  {out}/manifest.jsonl — per-frame metadata")
    print()
    print("Verification (manual):")
    print(f"  1. Open {out}/annotated/ in Finder/image viewer")
    print(f"  2. DELETE annotated JPEGs where the BALL bbox is wrong (false positive)")
    print(f"  3. Optionally scan no_detection/ for frames where ball IS visible")
    print(f"  4. Run build_yolo_train_set.py to produce ultralytics dataset layout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
