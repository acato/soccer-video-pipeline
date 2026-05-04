"""Kickoff-frame verifier — precision gate on shot.outcome=goal events.

Run 65 evolution from Run 64. The Run 64 prompt forces every shot to declare
an outcome ∈ {save, corner_kick, goal_kick, goal}; the parser emits a paired
goal event when outcome=goal. That path is high-recall (3/4 GT goals on Rush)
but produces ~3 false-positive goal calls per game.

This verifier samples a small set of frames at fixed offsets after the shot
and uses YOLO to confirm a kickoff scene:
  (a) ball detected within central 20% of the frame (x_norm, y_norm ∈ 0.5±0.10)
  (b) ≥ MIN_PERSONS persons visible (filters close-ups / replays)
  (c) |left_half_count − right_half_count| / total ≤ MAX_HALF_IMBALANCE
       (after a real goal both teams retreat to their halves; a saved-shot
        scene 30s later still has players bunched in one half)

If ANY of the probe frames satisfies all three conditions, the goal stands.
If NONE match (or YOLO fails), behavior depends on `fail_open`:
  - fail_open=False: drop the goal but keep the parent shot.
  - fail_open=True : keep the goal; only drop on positive negative evidence
                     (≥1 frame had a complete YOLO read AND failed the rule).

The verifier targets ONLY events with metadata.detection_method == "shot_outcome"
because those are the new Run 64 paired goals — they reliably carry the SHOT
timestamp, so +20-50s lands in the celebration/kickoff window. Direct VLM goal
classifications fire on POST-goal frames, where +20-50s is unrelated play.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from src.detection.frame_sampler import FrameSampler
from src.detection.models import Event, EventType

log = structlog.get_logger(__name__)


@dataclass
class _ProbeResult:
    """Per-frame YOLO read used by verify()."""
    timestamp_sec: float
    has_yolo_read: bool          # YOLO returned boxes for this frame
    ball_central: Optional[bool] # None if ball not detected
    n_persons: int
    half_imbalance: Optional[float]  # |L-R|/total; None if persons < 2


def _yolo_inference(model, frame_jpeg: bytes, imgsz: int = 640,
                    conf: float = 0.15, use_gpu: bool = True):
    """Run YOLO on a single JPEG. Returns the ultralytics result or None."""
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
    except Exception as exc:  # pragma: no cover - YOLO runtime
        log.warning("kickoff_verifier.yolo_error", error=str(exc))
        return None
    if not results:
        return None
    return results[0]


def _probe_frame(
    model,
    sampler: FrameSampler,
    video_duration: float,
    target_sec: float,
    *,
    ball_class_id: int,
    person_class_ids: frozenset[int],
    ball_conf: float,
    inference_size: int,
    use_gpu: bool,
    central_box: float,
) -> _ProbeResult:
    """Sample one frame at target_sec and YOLO-read it."""
    if target_sec >= video_duration:
        return _ProbeResult(target_sec, False, None, 0, None)
    frames = sampler.sample_range(
        center_sec=target_sec,
        window_sec=0.5,
        interval_sec=1.0,
        duration_sec=video_duration,
    )
    if not frames:
        return _ProbeResult(target_sec, False, None, 0, None)
    frame = frames[0]
    result = _yolo_inference(model, frame.jpeg_bytes,
                             imgsz=inference_size, conf=ball_conf, use_gpu=use_gpu)
    if result is None or not hasattr(result, "boxes") or result.boxes is None:
        return _ProbeResult(target_sec, False, None, 0, None)
    try:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        xywhn = result.boxes.xywhn.cpu().numpy()
    except AttributeError:
        return _ProbeResult(target_sec, False, None, 0, None)

    ball_central: Optional[bool] = None
    person_xs: list[float] = []
    for cls, conf, xywh in zip(classes, confs, xywhn):
        cls_int = int(cls)
        cx, cy = float(xywh[0]), float(xywh[1])
        if cls_int == ball_class_id:
            in_central = (
                abs(cx - 0.5) <= central_box
                and abs(cy - 0.5) <= central_box
            )
            if ball_central is None or in_central:
                ball_central = in_central
        if cls_int in person_class_ids:
            person_xs.append(cx)
    n_persons = len(person_xs)
    half_imbalance: Optional[float] = None
    if n_persons >= 2:
        left = sum(1 for x in person_xs if x < 0.5)
        right = n_persons - left
        half_imbalance = abs(left - right) / n_persons
    return _ProbeResult(
        timestamp_sec=target_sec,
        has_yolo_read=True,
        ball_central=ball_central,
        n_persons=n_persons,
        half_imbalance=half_imbalance,
    )


def verify_outcome_goals(
    events: list[Event],
    *,
    sampler: FrameSampler,
    video_duration: float,
    model_path: Optional[str],
    inference_size: int = 640,
    ball_conf: float = 0.15,
    use_gpu: bool = True,
    ball_class_id: int = 32,
    person_class_ids: frozenset[int] = frozenset({0}),
    probe_offsets_sec: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0),
    central_box: float = 0.10,             # half-width: 0.10 → ball within 0.4-0.6 in x and y
    min_persons: int = 8,
    max_half_imbalance: float = 0.30,
    fail_open: bool = True,
    job_id: Optional[str] = None,
) -> tuple[list[Event], dict]:
    """Filter events: drop shot-outcome paired goals lacking a kickoff frame.

    Returns (events, stats). Non-goal events pass through unchanged. Direct
    VLM goal classifications (detection_method != "shot_outcome") pass through
    unchanged. Only the Run-64-introduced paired goals are gated.
    """
    stats = {
        "checked": 0,
        "kept_match": 0,
        "kept_fail_open": 0,
        "dropped_no_kickoff": 0,
        "skipped_non_outcome": 0,
    }

    targets = [
        e for e in events
        if e.event_type == EventType.GOAL
        and (e.metadata or {}).get("detection_method") == "shot_outcome"
    ]
    if not targets:
        log.info("kickoff_verifier.no_targets", job_id=job_id)
        return events, stats

    if not model_path:
        log.warning("kickoff_verifier.no_model_path", job_id=job_id,
                    targets=len(targets))
        return events, stats

    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO(model_path)
    except Exception as exc:  # pragma: no cover - YOLO load
        log.warning("kickoff_verifier.model_load_failed",
                    error=str(exc), path=model_path, job_id=job_id)
        return events, stats

    drop_ids: set[str] = set()
    for ev in targets:
        stats["checked"] += 1
        any_match = False
        any_complete_read = False
        probes: list[_ProbeResult] = []
        for offset in probe_offsets_sec:
            probe_t = ev.timestamp_start + offset
            probe = _probe_frame(
                model, sampler, video_duration, probe_t,
                ball_class_id=ball_class_id,
                person_class_ids=person_class_ids,
                ball_conf=ball_conf,
                inference_size=inference_size,
                use_gpu=use_gpu,
                central_box=central_box,
            )
            probes.append(probe)
            if not probe.has_yolo_read:
                continue
            # Need ALL three signals present to call a match
            if (probe.ball_central is True
                and probe.n_persons >= min_persons
                and probe.half_imbalance is not None
                and probe.half_imbalance <= max_half_imbalance):
                any_match = True
                # Stop early — one matching frame is enough
                break
            # A complete read means YOLO worked AND we got at least a person
            # count + half stat. (Ball may or may not be detected; that's a
            # negative signal, not a missing read.)
            if probe.n_persons >= 2 and probe.half_imbalance is not None:
                any_complete_read = True

        ev.metadata = ev.metadata or {}
        ev.metadata["kickoff_verifier_probes"] = [
            {
                "t": p.timestamp_sec,
                "yolo_read": p.has_yolo_read,
                "ball_central": p.ball_central,
                "n_persons": p.n_persons,
                "half_imbalance": p.half_imbalance,
            }
            for p in probes
        ]

        if any_match:
            ev.metadata["kickoff_verifier_outcome"] = "match"
            stats["kept_match"] += 1
            continue

        if not any_complete_read:
            # No frame produced a complete read — fail-open keeps the goal.
            if fail_open:
                ev.metadata["kickoff_verifier_outcome"] = "fail_open_no_read"
                stats["kept_fail_open"] += 1
                continue

        # Active negative evidence: at least one frame read cleanly and the
        # rule didn't hold across any probe.
        ev.metadata["kickoff_verifier_outcome"] = "drop_no_kickoff"
        drop_ids.add(ev.event_id or "")
        stats["dropped_no_kickoff"] += 1

    if drop_ids:
        events = [e for e in events if (e.event_id or "") not in drop_ids]

    log.info("kickoff_verifier.summary", job_id=job_id, **stats,
             total_events_in=len(events) + len(drop_ids),
             total_events_out=len(events))
    return events, stats
