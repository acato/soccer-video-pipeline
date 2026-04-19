"""YOLO spatial grounding gate for VLM-detected events.

Breakthrough layer for Run #33 onward.

The 32B VLM in single-pass mode plateaus around F1 ≈ 0.39 because it
over-emits type-confused events (e.g. calls midfield stoppages "throw_in",
tags any ball near the sideline as "corner_kick", treats GK holding a
goal-kicked ball as "catch"). These event types have hard SPATIAL
prerequisites the VLM ignores:

  - throw_in requires the ball to be near a touchline
  - corner_kick requires the ball near a corner of the field
  - goal_kick requires the ball near a goal line

This module runs a lightweight YOLOv8 pass on a few frames around each
tentative event, extracts ball + person coordinates, and enforces the
spatial rule. When the rule is contradicted, the event is rejected.

Fail-open policy: if YOLO doesn't detect a ball in any sampled frame (or
fails entirely), the event is KEPT. We only reject on active evidence of
a spatial contradiction. This preserves recall while cutting FPs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from src.detection.frame_sampler import FrameSampler, SampledFrame
from src.detection.models import Event, EventType

log = structlog.get_logger(__name__)


# COCO class IDs (YOLOv8 default model). These are the defaults when no
# custom class IDs are configured. Soccer-tuned detectors typically use a
# different schema (e.g. uisikdag/yolo-v8-football-players-detection uses
# 0=ball, 1=goalkeeper, 2=player, 3=referee), so class IDs are passed
# into YoloGrounder at construction time.
_COCO_PERSON = 0
_COCO_SPORTS_BALL = 32


# Event types this module actively gates. All others pass through.
_GATED_TYPES: frozenset[EventType] = frozenset({
    EventType.THROW_IN,
    EventType.CORNER_KICK,
    EventType.GOAL_KICK,
})


# Spatial thresholds in normalized frame coords (0..1). These assume a
# typical sideline-camera POV where the touchlines are the top/bottom edges
# of the frame and the goal lines are the left/right edges.
#
# Band widths were widened after the Run #34 post-mortem: many true-positive
# rejections clustered just outside the Run #33 bands (throw_in ys at 0.25,
# goal_kick xs at 0.16–0.17 — within a few percent of the previous cutoffs).
_TOUCHLINE_BAND = 0.28           # throw_in: within 28% of top or bottom
_GOAL_LINE_BAND = 0.20           # goal_kick: within 20% of left or right
_GOAL_LINE_VERT_MIN = 0.25       # goal_kick: vertical middle 50% only
_GOAL_LINE_VERT_MAX = 0.75       # (excludes the field corners)
_CORNER_X_BAND = 0.25            # corner_kick: within 25% of left or right
_CORNER_Y_BAND = 0.25            # corner_kick: AND 25% of top or bottom


@dataclass
class BallDetection:
    """A single YOLO ball detection."""
    x_norm: float                 # Center x in [0, 1]
    y_norm: float                 # Center y in [0, 1]
    confidence: float
    frame_ts: float


@dataclass
class GoalkeeperDetection:
    """A single YOLO goalkeeper detection (soccer-tuned models only)."""
    x_norm: float
    y_norm: float
    confidence: float
    frame_ts: float


@dataclass
class SpatialFeatures:
    """Aggregated spatial signals across the sampled frames."""
    n_frames: int = 0
    ball_detected: bool = False
    ball_x_norm: Optional[float] = None      # Best-confidence ball x
    ball_y_norm: Optional[float] = None      # Best-confidence ball y
    ball_confidence: float = 0.0
    ball_detections: list[BallDetection] = field(default_factory=list)
    person_count_max: int = 0                # Max persons in any single frame
    # Goalkeeper data (populated only when gk_class_ids is configured;
    # empty for COCO-class setups). Collected for the Run #36 GK-action
    # gate design.
    gk_detections: list[GoalkeeperDetection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "ball_detected": self.ball_detected,
            "ball_x_norm": self.ball_x_norm,
            "ball_y_norm": self.ball_y_norm,
            "ball_confidence": self.ball_confidence,
            "person_count_max": self.person_count_max,
            "n_ball_detections": len(self.ball_detections),
            "n_gk_detections": len(self.gk_detections),
            "gk_positions": [
                {"x": round(g.x_norm, 3), "y": round(g.y_norm, 3),
                 "conf": round(g.confidence, 3), "t": round(g.frame_ts, 1)}
                for g in self.gk_detections
            ],
        }


@dataclass
class GroundingDecision:
    """Outcome of running the grounding gate on an event."""
    keep: bool
    reason: str
    features: SpatialFeatures


class YoloGrounder:
    """Spatial-grounding gate for VLM-detected events.

    Thin wrapper around YOLOv8 that extracts ball/person positions from
    3 frames per candidate event and applies per-event-type rules.
    """

    _GATED_TYPES = _GATED_TYPES

    def __init__(
        self,
        *,
        sampler: FrameSampler,
        video_duration: float,
        model_path: Optional[str] = None,
        inference_size: int = 640,
        use_gpu: bool = True,
        ball_conf_threshold: float = 0.15,
        n_frames: int = 5,
        frame_span_sec: float = 2.0,
        fail_open: bool = True,
        diagnostics_path: Optional[Path] = None,
        model: Optional[object] = None,  # Injected YOLO model (tests)
        ball_class_id: int = _COCO_SPORTS_BALL,
        person_class_ids: tuple[int, ...] = (_COCO_PERSON,),
        gk_class_ids: tuple[int, ...] = (),
    ):
        self._sampler = sampler
        self._video_duration = video_duration
        self._model_path = model_path
        self._inference_size = inference_size
        self._use_gpu = use_gpu
        self._ball_conf = ball_conf_threshold
        self._n_frames = max(1, n_frames)
        self._ball_class_id = int(ball_class_id)
        self._person_class_ids = frozenset(int(c) for c in person_class_ids)
        # Goalkeeper class ids are also typically a subset of person_class_ids;
        # they are tracked separately for diagnostics and the future GK-action
        # gate (Run #36).
        self._gk_class_ids = frozenset(int(c) for c in gk_class_ids)
        # Minimum span fallback when the event's own duration is too short
        # (e.g. point events with timestamp_end == timestamp_start).
        self._min_span_sec = max(0.5, frame_span_sec)
        self._fail_open = fail_open
        self._diag_path = diagnostics_path
        self._model = model            # Lazy-loaded via _load() if None
        self._model_load_failed = False
        self._diag_file = None
        if self._diag_path:
            self._diag_path.parent.mkdir(parents=True, exist_ok=True)
            self._diag_file = open(self._diag_path, "w")

    # ── Public API ──────────────────────────────────────────────────────

    def filter(self, events: list[Event]) -> list[Event]:
        """Apply spatial grounding to a list of events.

        Returns a filtered list. Non-gated event types pass through
        unchanged. Gated types are kept iff their spatial rule holds
        (or fail-open fires).
        """
        kept: list[Event] = []
        dropped = 0
        by_type_dropped: dict[str, int] = {}
        for event in events:
            if event.event_type not in self._GATED_TYPES:
                kept.append(event)
                continue
            decision = self._verify(event)
            self._emit_diag(event, decision)
            if decision.keep:
                kept.append(event)
            else:
                dropped += 1
                by_type_dropped[event.event_type.value] = (
                    by_type_dropped.get(event.event_type.value, 0) + 1
                )
                log.info(
                    "yolo_grounding.reject",
                    event_type=event.event_type.value,
                    start=event.timestamp_start,
                    reason=decision.reason,
                    features=decision.features.to_dict(),
                )
        log.info(
            "yolo_grounding.filter_complete",
            total_in=len(events), total_out=len(kept),
            dropped=dropped, by_type_dropped=by_type_dropped,
        )
        return kept

    def close(self) -> None:
        if self._diag_file:
            self._diag_file.close()
            self._diag_file = None

    # ── Verification ────────────────────────────────────────────────────

    def _verify(self, event: Event) -> GroundingDecision:
        frames = self._sample_event_span(event)
        features = self._extract_features(frames)

        # Fail-open: no frames, no ball, or YOLO unavailable → keep.
        if features.n_frames == 0:
            return GroundingDecision(
                keep=True, reason="no_frames_sampled", features=features,
            )
        if not features.ball_detected:
            if self._fail_open:
                return GroundingDecision(
                    keep=True, reason="no_ball_detected_fail_open",
                    features=features,
                )
            return GroundingDecision(
                keep=False, reason="no_ball_detected_fail_closed",
                features=features,
            )

        # Per-type rules
        if event.event_type == EventType.THROW_IN:
            return self._verify_throw_in(features)
        if event.event_type == EventType.CORNER_KICK:
            return self._verify_corner_kick(features)
        if event.event_type == EventType.GOAL_KICK:
            return self._verify_goal_kick(features)

        # Shouldn't reach (not in _GATED_TYPES) — defensive default.
        return GroundingDecision(
            keep=True, reason="ungated_passthrough", features=features,
        )

    def _verify_throw_in(self, f: SpatialFeatures) -> GroundingDecision:
        """Ball must be near a touchline in ANY sampled frame."""
        for det in f.ball_detections:
            y = det.y_norm
            if y <= _TOUCHLINE_BAND or y >= (1.0 - _TOUCHLINE_BAND):
                return GroundingDecision(
                    keep=True,
                    reason=f"ball_near_touchline_y={y:.2f}_t={det.frame_ts:.1f}",
                    features=f,
                )
        ys = [round(d.y_norm, 2) for d in f.ball_detections]
        return GroundingDecision(
            keep=False,
            reason=f"ball_never_near_touchline_ys={ys}",
            features=f,
        )

    def _verify_corner_kick(self, f: SpatialFeatures) -> GroundingDecision:
        """Ball must be in a corner quadrant in ANY sampled frame."""
        for det in f.ball_detections:
            x, y = det.x_norm, det.y_norm
            in_x_band = x <= _CORNER_X_BAND or x >= (1.0 - _CORNER_X_BAND)
            in_y_band = y <= _CORNER_Y_BAND or y >= (1.0 - _CORNER_Y_BAND)
            if in_x_band and in_y_band:
                return GroundingDecision(
                    keep=True,
                    reason=f"ball_in_corner_x={x:.2f}_y={y:.2f}_t={det.frame_ts:.1f}",
                    features=f,
                )
        positions = [(round(d.x_norm, 2), round(d.y_norm, 2)) for d in f.ball_detections]
        return GroundingDecision(
            keep=False,
            reason=f"ball_never_in_corner_xys={positions}",
            features=f,
        )

    def _verify_goal_kick(self, f: SpatialFeatures) -> GroundingDecision:
        """Ball must be near a goal line (vertical middle) in ANY sampled frame."""
        for det in f.ball_detections:
            x, y = det.x_norm, det.y_norm
            near_left = x <= _GOAL_LINE_BAND
            near_right = x >= (1.0 - _GOAL_LINE_BAND)
            vert_ok = _GOAL_LINE_VERT_MIN <= y <= _GOAL_LINE_VERT_MAX
            if (near_left or near_right) and vert_ok:
                return GroundingDecision(
                    keep=True,
                    reason=f"ball_near_goal_line_x={x:.2f}_y={y:.2f}_t={det.frame_ts:.1f}",
                    features=f,
                )
        positions = [(round(d.x_norm, 2), round(d.y_norm, 2)) for d in f.ball_detections]
        return GroundingDecision(
            keep=False,
            reason=f"ball_never_near_goal_line_xys={positions}",
            features=f,
        )

    # ── Frame sampling + YOLO inference ─────────────────────────────────

    def _sample_event_span(self, event: Event) -> list[SampledFrame]:
        """Pull n_frames spanning the full event window [start, end].

        The landmark for throw_in / corner_kick / goal_kick is momentary —
        the ball is only at the touchline/corner/goal-line for a fraction
        of the full event window. Sampling across the entire span (rather
        than ±1s around a single center) gives the gate a chance to see
        that moment even when timestamp_start doesn't coincide with it.
        """
        start = max(0.0, event.timestamp_start)
        end = event.timestamp_end if event.timestamp_end > start else start
        span = max(self._min_span_sec, end - start)
        center = start + span / 2.0
        half_window = span / 2.0
        if self._n_frames == 1:
            interval = span
        else:
            interval = span / max(1, self._n_frames - 1)
        frames = self._sampler.sample_range(
            center_sec=center,
            window_sec=half_window,
            interval_sec=max(0.2, interval),
            duration_sec=self._video_duration,
        )
        if len(frames) > self._n_frames:
            step = len(frames) / self._n_frames
            indices = [int(i * step) for i in range(self._n_frames)]
            frames = [frames[i] for i in indices]
        return frames

    def _extract_features(self, frames: list[SampledFrame]) -> SpatialFeatures:
        feats = SpatialFeatures(n_frames=len(frames))
        if not frames:
            return feats
        model = self._load()
        if model is None:
            return feats

        # Decode JPEG bytes once per frame, run YOLO in a single batch
        import cv2
        import numpy as np

        images: list[np.ndarray] = []
        for f in frames:
            arr = np.frombuffer(f.jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            images.append(img)

        if not images:
            return feats

        try:
            # Batched inference — YOLOv8 accepts list of numpy arrays
            kwargs = {
                "imgsz": self._inference_size,
                "conf": self._ball_conf,
                "verbose": False,
            }
            if not self._use_gpu:
                kwargs["device"] = "cpu"
            # else: let ultralytics auto-select (cuda on Linux, mps on Mac)
            results = model(images, **kwargs)
        except Exception as exc:      # pragma: no cover
            log.warning("yolo_grounding.inference_error", error=str(exc))
            return feats

        best: Optional[BallDetection] = None
        person_max = 0
        for result, frame in zip(results, frames):
            # result.boxes has xywhn (normalized) or xyxyn tensors
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            boxes = result.boxes
            # Support both real ultralytics Boxes and test dicts
            try:
                classes = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                xywhn = boxes.xywhn.cpu().numpy()
            except AttributeError:
                continue

            n_person = 0
            for cls, conf, xywh in zip(classes, confs, xywhn):
                cls_int = int(cls)
                if cls_int == self._ball_class_id:
                    detection = BallDetection(
                        x_norm=float(xywh[0]),
                        y_norm=float(xywh[1]),
                        confidence=float(conf),
                        frame_ts=frame.timestamp_sec,
                    )
                    feats.ball_detections.append(detection)
                    if best is None or detection.confidence > best.confidence:
                        best = detection
                if cls_int in self._gk_class_ids:
                    feats.gk_detections.append(GoalkeeperDetection(
                        x_norm=float(xywh[0]),
                        y_norm=float(xywh[1]),
                        confidence=float(conf),
                        frame_ts=frame.timestamp_sec,
                    ))
                if cls_int in self._person_class_ids:
                    n_person += 1
            if n_person > person_max:
                person_max = n_person

        feats.person_count_max = person_max
        if best is not None:
            feats.ball_detected = True
            feats.ball_x_norm = best.x_norm
            feats.ball_y_norm = best.y_norm
            feats.ball_confidence = best.confidence

        return feats

    def _load(self):
        if self._model is not None or self._model_load_failed:
            return self._model
        if not self._model_path:
            self._model_load_failed = True
            log.warning("yolo_grounding.no_model_path")
            return None
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            log.info("yolo_grounding.model_loaded", path=self._model_path)
        except Exception as exc:  # pragma: no cover
            log.warning(
                "yolo_grounding.model_load_failed",
                path=self._model_path, error=str(exc),
            )
            self._model_load_failed = True
            self._model = None
        return self._model

    # ── Diagnostics ─────────────────────────────────────────────────────

    def _emit_diag(self, event: Event, decision: GroundingDecision) -> None:
        if self._diag_file is None:
            return
        rec = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp_start": event.timestamp_start,
            "keep": decision.keep,
            "reason": decision.reason,
            "features": decision.features.to_dict(),
        }
        json.dump(rec, self._diag_file)
        self._diag_file.write("\n")
        self._diag_file.flush()
