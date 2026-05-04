"""Auto-detect 1H + 2H kickoff video timestamps.

Scans early video for kickoff frame signature (ball at center + ≥8 persons +
balanced halves). Then jumps to the expected 2H window based on GT's earliest
2H event time and scans there.

The output offsets are for evaluate_detection.py:
  --video-offset       <- 1H kickoff video time
  --half2-start        <- 2H kickoff video time
  --half2-game-offset  <- floor(earliest_2H_event_time / 1000) (60s-rounded)

Usage:
  python scripts/find_video_offsets.py \
      /path/to/video.mp4 /path/to/1H_GT.json /path/to/2H_GT.json \
      [--yolo-model /path/to/yolov8m.pt] [--cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ssh sessions on macOS / linux servers may lack the right PATH; ffmpeg lives
# in /opt/homebrew/bin (macOS) or /usr/bin (linux).
os.environ["PATH"] = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:" + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"


@dataclass
class Probe:
    t_sec: float
    has_yolo_read: bool = False
    ball_central: Optional[bool] = None
    n_persons: int = 0
    half_imbalance: Optional[float] = None

    @property
    def is_kickoff(self) -> bool:
        return (
            self.has_yolo_read
            and self.ball_central is True
            and self.n_persons >= 8
            and self.half_imbalance is not None
            and self.half_imbalance <= 0.30
        )


def video_duration(video: Path) -> float:
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
         str(video)],
        text=True,
    )
    return float(out.strip())


def extract_frame(video: Path, ts_sec: float, width: int = 1280) -> Optional[bytes]:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        out_path = f.name
    try:
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
               "-ss", f"{ts_sec:.3f}", "-i", str(video),
               "-frames:v", "1",
               "-vf", f"scale={width}:-2",
               "-q:v", "3",
               "-y", out_path]
        subprocess.run(cmd, check=True)
        with open(out_path, "rb") as fp:
            return fp.read()
    except subprocess.CalledProcessError:
        return None
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def yolo_probe(model, frame_jpeg: bytes, imgsz: int, conf: float,
               use_gpu: bool, central_box: float = 0.10,
               ball_class_id: int = 0,
               person_class_ids: tuple[int, ...] = (1, 2, 3)) -> Probe:
    import cv2
    import numpy as np

    arr = np.frombuffer(frame_jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return Probe(0)

    kwargs = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if not use_gpu:
        kwargs["device"] = "cpu"

    try:
        results = model([img], **kwargs)
    except Exception:
        return Probe(0)
    if not results:
        return Probe(0)
    result = results[0]
    if not hasattr(result, "boxes") or result.boxes is None:
        return Probe(0)
    try:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        xywhn = result.boxes.xywhn.cpu().numpy()
    except AttributeError:
        return Probe(0)

    ball_central: Optional[bool] = None
    person_xs: list[float] = []
    person_set = set(person_class_ids)
    for cls, conf_i, xywh in zip(classes, confs, xywhn):
        cls_int = int(cls)
        cx, cy = float(xywh[0]), float(xywh[1])
        if cls_int == ball_class_id:
            in_central = abs(cx - 0.5) <= central_box and abs(cy - 0.5) <= central_box
            if ball_central is None or in_central:
                ball_central = in_central
        if cls_int in person_set:
            person_xs.append(cx)
    n = len(person_xs)
    half_imbalance = None
    if n >= 2:
        left = sum(1 for x in person_xs if x < 0.5)
        half_imbalance = abs(left - (n - left)) / n
    return Probe(
        t_sec=0.0, has_yolo_read=True, ball_central=ball_central,
        n_persons=n, half_imbalance=half_imbalance,
    )


def scan_for_kickoff(model, video: Path, *, base_sec: float, span_sec: float,
                     step_sec: float, imgsz: int, conf: float, use_gpu: bool,
                     ball_class_id: int, person_class_ids: tuple[int, ...]) -> Optional[float]:
    """Coarse scan returning the earliest video timestamp matching the kickoff
    signature, or None if nothing matched."""
    print(f"  scanning [{base_sec:.0f}, {base_sec + span_sec:.0f}]s step={step_sec:.0f}s",
          file=sys.stderr)
    t = base_sec
    end = base_sec + span_sec
    while t <= end:
        jpg = extract_frame(video, t)
        if jpg is None:
            t += step_sec
            continue
        p = yolo_probe(model, jpg, imgsz=imgsz, conf=conf, use_gpu=use_gpu,
                       ball_class_id=ball_class_id,
                       person_class_ids=person_class_ids)
        p.t_sec = t
        flag = "✓" if p.is_kickoff else " "
        print(f"    [{flag}] t={t:7.1f}s yolo={p.has_yolo_read} "
              f"ball_central={p.ball_central} persons={p.n_persons} "
              f"imbalance={p.half_imbalance}", file=sys.stderr)
        if p.is_kickoff:
            return t
        t += step_sec
    return None


def refine(model, video: Path, anchor_sec: float, *, half_span_sec: float,
           step_sec: float, imgsz: int, conf: float, use_gpu: bool,
           ball_class_id: int, person_class_ids: tuple[int, ...]) -> float:
    """Tighten an anchor by re-scanning ±half_span at finer step."""
    base = max(0.0, anchor_sec - half_span_sec)
    earliest = scan_for_kickoff(
        model, video, base_sec=base, span_sec=2 * half_span_sec,
        step_sec=step_sec, imgsz=imgsz, conf=conf, use_gpu=use_gpu,
        ball_class_id=ball_class_id, person_class_ids=person_class_ids,
    )
    return earliest if earliest is not None else anchor_sec


def gt_earliest_event_sec(gt_json: Path) -> float:
    data = json.loads(gt_json.read_text())
    times_ms: list[int] = []
    for entry in data.get("data", []):
        for ev in entry.get("events", []):
            t = entry.get("event_time")
            if t is None:
                continue
            times_ms.append(int(t))
    if not times_ms:
        return 0.0
    return min(times_ms) / 1000.0


def round_to_minute(sec: float) -> float:
    """Snap to the nearest 60s — youth halves come in 5-min increments."""
    return float(int(round(sec / 60.0)) * 60)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("gt_h1", type=Path)
    p.add_argument("gt_h2", type=Path)
    p.add_argument("--yolo-model", default="/Users/aless/Downloads/soccer-video-pipeline/infra/models/yolov8_soccer_uisikdag.pt")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU YOLO (default: auto-select GPU)")
    p.add_argument("--ball-class-id", type=int, default=0,
                   help="0 for soccer-tuned, 32 for COCO sports_ball")
    p.add_argument("--person-class-ids", default="1,2,3",
                   help="comma-separated. soccer-tuned: 1,2,3 (GK+player+ref). COCO: 0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--coarse-step", type=float, default=15.0)
    p.add_argument("--coarse-1h-span", type=float, default=600.0,
                   help="Search 1H kickoff in first N seconds (default 10 min)")
    p.add_argument("--coarse-2h-span", type=float, default=1500.0,
                   help="2H scan window length (default 25 min — covers any halftime")
    args = p.parse_args()

    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 2
    if not args.gt_h2.exists():
        print(f"gt_h2 not found: {args.gt_h2}", file=sys.stderr)
        return 2

    duration = video_duration(args.video)
    print(f"video={args.video.name} duration={duration:.1f}s", file=sys.stderr)

    earliest_h2 = gt_earliest_event_sec(args.gt_h2)
    half2_game_offset = round_to_minute(earliest_h2)
    print(f"earliest 2H event_time={earliest_h2:.1f}s → "
          f"half2_game_offset={half2_game_offset:.0f}s "
          f"(half length ≈ {half2_game_offset/60:.0f} min)", file=sys.stderr)

    person_class_ids = tuple(int(x) for x in args.person_class_ids.split(",") if x.strip())

    # Lazy YOLO load
    print(f"loading YOLO from {args.yolo_model} (cpu={args.cpu})", file=sys.stderr)
    print(f"  ball_class_id={args.ball_class_id}  person_class_ids={person_class_ids}",
          file=sys.stderr)
    from ultralytics import YOLO
    model = YOLO(args.yolo_model)

    scan_kwargs = dict(
        imgsz=args.imgsz, conf=args.conf, use_gpu=not args.cpu,
        ball_class_id=args.ball_class_id, person_class_ids=person_class_ids,
    )

    # 1H scan
    print("[scan 1H] looking for first kickoff frame...", file=sys.stderr)
    h1_anchor = scan_for_kickoff(
        model, args.video, base_sec=0.0, span_sec=args.coarse_1h_span,
        step_sec=args.coarse_step, **scan_kwargs,
    )
    if h1_anchor is None:
        print("ERROR: 1H kickoff not found in coarse scan", file=sys.stderr)
        return 1
    print(f"[scan 1H] anchor={h1_anchor:.1f}s — refining...", file=sys.stderr)
    h1_kickoff = refine(model, args.video, h1_anchor, half_span_sec=15.0,
                        step_sec=3.0, **scan_kwargs)
    print(f"[scan 1H] kickoff={h1_kickoff:.1f}s", file=sys.stderr)

    # 2H scan
    h2_search_base = h1_kickoff + half2_game_offset
    if h2_search_base + args.coarse_2h_span > duration:
        args.coarse_2h_span = max(0.0, duration - h2_search_base - 30)
    print(f"[scan 2H] expected window starts at "
          f"{h2_search_base:.1f}s (h1 + {half2_game_offset:.0f}s)", file=sys.stderr)
    h2_anchor = scan_for_kickoff(
        model, args.video, base_sec=h2_search_base,
        span_sec=args.coarse_2h_span, step_sec=args.coarse_step, **scan_kwargs,
    )
    if h2_anchor is None:
        print("WARN: 2H kickoff not found; using h1 + half2_game_offset + 720",
              file=sys.stderr)
        h2_kickoff = h1_kickoff + half2_game_offset + 720.0
    else:
        h2_kickoff = refine(model, args.video, h2_anchor, half_span_sec=15.0,
                            step_sec=3.0, **scan_kwargs)
    print(f"[scan 2H] kickoff={h2_kickoff:.1f}s", file=sys.stderr)

    # Final report
    print()
    print("=== OFFSETS ===")
    print(f"video_offset     = {h1_kickoff:.1f}")
    print(f"half2_start      = {h2_kickoff:.1f}")
    print(f"half2_game_offset= {half2_game_offset:.0f}")
    print()
    print(f"# eval invocation:")
    print(f"  --video-offset {h1_kickoff:.1f} \\")
    print(f"  --half2-start {h2_kickoff:.1f} \\")
    print(f"  --half2-game-offset {half2_game_offset:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
