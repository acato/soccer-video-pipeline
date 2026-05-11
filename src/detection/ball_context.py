"""v9b ball-location annotation for 32B classification prompts.

Per-frame, runs v9b at low conf and returns a short text annotation the
32B can reason over alongside the visual frames. Designed for the
single-pass classification loop where each frame already has a timestamp
label.

Output examples:
  "ball@(0.42,0.58):0.34"                  # one detection
  "ball@(0.42,0.58):0.34;ball@(0.21,0.35):0.12"  # up to max_dets
  "no_ball"                                # nothing above conf threshold

Coordinates are normalized to the JPEG image dimensions (which match what
the 32B sees — same crop applied beforehand).
"""
from __future__ import annotations

from typing import Optional

import structlog

log = structlog.get_logger(__name__)


_BALL_CONTEXT_PROMPT_PREFIX = (
    "Each frame timestamp may be followed by a ball-location annotation from "
    "an external ball detector (v9b YOLO). The annotation is formatted as "
    "`ball@(x,y):conf` where x,y are normalized frame coordinates (0,0=top-left, "
    "1,1=bottom-right) and conf is the detector's confidence (0-1). Multiple "
    "candidates per frame are separated by semicolons. `no_ball` means the "
    "detector found nothing. The detector has ~60% per-frame recall on small "
    "balls and may emit false positives, so treat it as a HINT not a fact: "
    "use it to locate the ball faster, but verify against what you actually "
    "see. Trajectory across frames (ball position changing over time) is the "
    "strongest signal for shot/goal events.\n\n"
)


def load_model(model_path: str):
    """Lazy-load the v9b YOLO model. Returns the model or None on failure."""
    try:
        from ultralytics import YOLO  # type: ignore
        return YOLO(model_path)
    except Exception as exc:  # pragma: no cover
        log.warning("ball_context.model_load_failed", path=model_path, error=str(exc))
        return None


def annotate_frame(model, frame_jpeg: bytes, *, conf: float = 0.05,
                   imgsz: int = 1920, max_dets: int = 3,
                   use_gpu: bool = True) -> str:
    """Run v9b on one frame, return the text annotation."""
    if model is None:
        return "no_ball"
    import cv2
    import numpy as np

    arr = np.frombuffer(frame_jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return "no_ball"
    H, W = img.shape[:2]
    kwargs = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if not use_gpu:
        kwargs["device"] = "cpu"
    try:
        results = model([img], **kwargs)
    except Exception as exc:  # pragma: no cover
        log.warning("ball_context.inference_error", error=str(exc))
        return "no_ball"
    if not results:
        return "no_ball"
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return "no_ball"
    try:
        confs = boxes.conf.cpu().numpy()
        xywhn = boxes.xywhn.cpu().numpy()
    except AttributeError:
        return "no_ball"
    # Sort by conf desc, take top-K
    order = confs.argsort()[::-1][:max_dets]
    parts = []
    for i in order:
        cx, cy = float(xywhn[i][0]), float(xywhn[i][1])
        c = float(confs[i])
        parts.append(f"ball@({cx:.2f},{cy:.2f}):{c:.2f}")
    return ";".join(parts) if parts else "no_ball"


def prompt_prefix() -> str:
    """Prefix to prepend to the 32B classification prompt when annotations are active."""
    return _BALL_CONTEXT_PROMPT_PREFIX
