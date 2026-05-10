"""Ball-presence verifier — v9b YOLO precision gate on shot.outcome=goal events.

Complements the existing kickoff_verifier:
  - kickoff_verifier checks the POST-goal scene (+20-50s after the shot):
    ball central + ≥8 persons + balanced halves. Designed around prod YOLO
    (uisikdag) which detects ball + persons + GK.
  - ball_presence_verifier (this module) checks the AT-goal moment
    (within the goal event's own [t_start, t_end] window) using v9b — a
    new-venue-tuned single-class ball detector that catches balls the
    production model misses entirely (R=0.56 vs R=0.0 on new venue val).

Logic:
  Sample N frames evenly across [t_start, t_end] of each shot-outcome paired
  goal event. Run v9b at low conf. If 0/N frames register a ball detection,
  drop the goal (active negative evidence). Otherwise keep. Fail-open on YOLO
  errors so a missing model doesn't tank recall.

The two verifiers compose: kickoff_verifier may keep a goal that
ball_presence_verifier drops (and vice versa). Both must agree to keep ↔
either's drop wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from src.detection.frame_sampler import FrameSampler
from src.detection.models import Event, EventType

log = structlog.get_logger(__name__)


@dataclass
class _Probe:
    timestamp_sec: float
    has_yolo_read: bool
    ball_detected: bool
    n_dets: int
    max_conf: float


def _yolo_ball_inference(model, frame_jpeg: bytes, *, imgsz: int, conf: float,
                         use_gpu: bool, ball_class_id: int) -> Optional[_Probe]:
    """Run v9b on one frame; return ball-detection summary."""
    import cv2
    import numpy as np

    arr = np.frombuffer(frame_jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    kwargs = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if not use_gpu:
        kwargs["device"] = "cpu"
    try:
        results = model([img], **kwargs)
    except Exception as exc:  # pragma: no cover
        log.warning("ball_presence_verifier.yolo_error", error=str(exc))
        return None
    if not results:
        return None
    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return _Probe(0.0, True, False, 0, 0.0)
    try:
        classes = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
    except AttributeError:
        return _Probe(0.0, True, False, 0, 0.0)
    ball_confs = [float(c) for cls, c in zip(classes, confs) if int(cls) == ball_class_id]
    return _Probe(
        timestamp_sec=0.0,
        has_yolo_read=True,
        ball_detected=len(ball_confs) > 0,
        n_dets=len(ball_confs),
        max_conf=max(ball_confs) if ball_confs else 0.0,
    )


def _probe_frame(model, sampler: FrameSampler, video_duration: float,
                 target_sec: float, *, ball_class_id: int, ball_conf: float,
                 inference_size: int, use_gpu: bool) -> _Probe:
    if target_sec >= video_duration or target_sec < 0:
        return _Probe(target_sec, False, False, 0, 0.0)
    frames = sampler.sample_range(
        center_sec=target_sec,
        window_sec=0.5,
        interval_sec=1.0,
        duration_sec=video_duration,
    )
    if not frames:
        return _Probe(target_sec, False, False, 0, 0.0)
    probe = _yolo_ball_inference(
        model, frames[0].jpeg_bytes,
        imgsz=inference_size, conf=ball_conf,
        use_gpu=use_gpu, ball_class_id=ball_class_id,
    )
    if probe is None:
        return _Probe(target_sec, False, False, 0, 0.0)
    probe.timestamp_sec = target_sec
    return probe


def verify_goal_events(
    events: list[Event],
    *,
    sampler: FrameSampler,
    video_duration: float,
    model_path: Optional[str],
    inference_size: int = 1920,
    ball_conf: float = 0.10,
    use_gpu: bool = True,
    ball_class_id: int = 0,
    n_frames: int = 4,
    fail_open: bool = True,
    job_id: Optional[str] = None,
) -> tuple[list[Event], dict]:
    """Drop shot-outcome paired goals where v9b finds no ball anywhere in
    the goal window. Returns (events, stats).

    Non-goal events and goals from other detection paths pass through unchanged.
    """
    stats = {
        "checked": 0,
        "kept_ball_seen": 0,
        "kept_fail_open": 0,
        "dropped_no_ball": 0,
    }

    targets = [
        e for e in events
        if e.event_type == EventType.GOAL
        and (e.metadata or {}).get("detection_method") == "shot_outcome"
    ]
    if not targets:
        log.info("ball_presence_verifier.no_targets", job_id=job_id)
        return events, stats

    if not model_path:
        log.warning("ball_presence_verifier.no_model_path", job_id=job_id,
                    targets=len(targets))
        return events, stats

    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO(model_path)
    except Exception as exc:  # pragma: no cover
        log.warning("ball_presence_verifier.model_load_failed",
                    error=str(exc), path=model_path, job_id=job_id)
        return events, stats

    drop_ids: set[str] = set()
    for ev in targets:
        stats["checked"] += 1
        t0 = float(ev.timestamp_start)
        t1 = float(ev.timestamp_end) if ev.timestamp_end and ev.timestamp_end > t0 else t0 + 4.0
        # Evenly spaced probes within the goal window
        if n_frames <= 1:
            probe_times = [(t0 + t1) / 2.0]
        else:
            step = (t1 - t0) / max(1, n_frames - 1)
            probe_times = [t0 + i * step for i in range(n_frames)]

        probes: list[_Probe] = []
        any_ball_seen = False
        any_complete_read = False
        for pt in probe_times:
            probe = _probe_frame(
                model, sampler, video_duration, pt,
                ball_class_id=ball_class_id, ball_conf=ball_conf,
                inference_size=inference_size, use_gpu=use_gpu,
            )
            probes.append(probe)
            if probe.has_yolo_read:
                any_complete_read = True
                if probe.ball_detected:
                    any_ball_seen = True

        ev.metadata = ev.metadata or {}
        ev.metadata["ball_presence_verifier_probes"] = [
            {"t": p.timestamp_sec, "yolo_read": p.has_yolo_read,
             "ball_detected": p.ball_detected, "n_dets": p.n_dets,
             "max_conf": round(p.max_conf, 3)}
            for p in probes
        ]

        if any_ball_seen:
            ev.metadata["ball_presence_verifier_outcome"] = "kept_ball_seen"
            stats["kept_ball_seen"] += 1
            continue

        if not any_complete_read:
            if fail_open:
                ev.metadata["ball_presence_verifier_outcome"] = "fail_open_no_read"
                stats["kept_fail_open"] += 1
                continue

        ev.metadata["ball_presence_verifier_outcome"] = "drop_no_ball"
        drop_ids.add(ev.event_id or "")
        stats["dropped_no_ball"] += 1

    if drop_ids:
        events = [e for e in events if (e.event_id or "") not in drop_ids]

    log.info("ball_presence_verifier.summary", job_id=job_id, **stats,
             total_events_in=len(events) + len(drop_ids),
             total_events_out=len(events))
    return events, stats
