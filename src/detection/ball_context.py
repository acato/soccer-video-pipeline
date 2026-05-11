"""v9b per-frame ball detection for 32B classification prompts.

Provides two output modes:
  - annotate_frame() — returns a terse per-frame text string for inline
    timestamp-label annotation (Phase 1, ~flat result on new venue).
  - detect_balls() — returns structured top-K detections, consumed by
    ball_trajectory to derive temporal features (Phase 2).
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


def detect_balls(model, frame_jpeg: bytes, *, conf: float = 0.05,
                 imgsz: int = 1920, max_dets: int = 3,
                 use_gpu: bool = True) -> list[dict]:
    """Run v9b on one frame. Returns list of top-K detections sorted by conf desc:
        [{'cx': float, 'cy': float, 'conf': float}, ...]
    Empty list if no detections or YOLO failure (silent fail-safe)."""
    if model is None:
        return []
    import cv2
    import numpy as np

    arr = np.frombuffer(frame_jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    kwargs = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if not use_gpu:
        kwargs["device"] = "cpu"
    try:
        results = model([img], **kwargs)
    except Exception as exc:  # pragma: no cover
        log.warning("ball_context.inference_error", error=str(exc))
        return []
    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    try:
        confs = boxes.conf.cpu().numpy()
        xywhn = boxes.xywhn.cpu().numpy()
    except AttributeError:
        return []
    order = confs.argsort()[::-1][:max_dets]
    return [
        {"cx": float(xywhn[i][0]), "cy": float(xywhn[i][1]), "conf": float(confs[i])}
        for i in order
    ]


def annotate_frame(model, frame_jpeg: bytes, *, conf: float = 0.05,
                   imgsz: int = 1920, max_dets: int = 3,
                   use_gpu: bool = True) -> str:
    """Per-frame text string for inline timestamp annotation. Phase 1 format."""
    dets = detect_balls(model, frame_jpeg, conf=conf, imgsz=imgsz,
                        max_dets=max_dets, use_gpu=use_gpu)
    if not dets:
        return "no_ball"
    return ";".join(
        f"ball@({d['cx']:.2f},{d['cy']:.2f}):{d['conf']:.2f}" for d in dets
    )


def prompt_prefix() -> str:
    """Phase 1 prompt prefix — per-frame annotation explanation."""
    return _BALL_CONTEXT_PROMPT_PREFIX
