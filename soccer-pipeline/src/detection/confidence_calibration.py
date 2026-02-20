"""
Post-processing confidence calibration.

Raw model confidences are often poorly calibrated — YOLOv8 tends to overestimate,
heuristic motion detectors tend to underestimate. This module applies isotonic
regression calibration using match-level statistics.

For production: fit calibrators on labeled match data and persist weights.
For initial deployment: use the pre-set linear calibrators defined here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.detection.models import Event, EventType


@dataclass
class CalibrationConfig:
    """
    Per-event-type confidence calibration parameters.
    
    new_conf = clamp(slope * raw_conf + intercept, min_conf, max_conf)
    
    Defaults are tuned conservatively — better to miss clips than include garbage.
    """
    # Detection method → (slope, intercept, min_conf, max_conf)
    yolo_player_detection: tuple = (0.85, 0.02, 0.30, 0.92)
    motion_heuristic:      tuple = (1.15, -0.05, 0.40, 0.88)
    action_recognition:    tuple = (0.95, 0.03, 0.50, 0.95)

    # Per-event type floor (never go below this even if model says lower)
    event_floors: dict[str, float] = field(default_factory=lambda: {
        "goal":               0.55,   # Goals are visually obvious; low floor ok
        "shot_stop_diving":   0.50,
        "shot_stop_standing": 0.45,
        "catch":              0.50,
        "tackle":             0.45,
        "dribble_sequence":   0.45,
    })


_DEFAULT_CONFIG = CalibrationConfig()


def calibrate_event(
    event: Event,
    detection_method: str = "motion_heuristic",
    config: CalibrationConfig = _DEFAULT_CONFIG,
) -> Event:
    """
    Apply confidence calibration to a single event.
    Returns new Event with updated confidence (immutable).
    """
    if detection_method == "yolo_player_detection":
        params = config.yolo_player_detection
    elif detection_method == "action_recognition":
        params = config.action_recognition
    else:
        params = config.motion_heuristic

    slope, intercept, min_conf, max_conf = params
    raw = event.confidence
    calibrated = float(np.clip(slope * raw + intercept, min_conf, max_conf))

    # Apply event-type floor
    floor = config.event_floors.get(event.event_type.value, 0.40)
    calibrated = max(calibrated, floor)

    if abs(calibrated - raw) > 0.001:
        data = event.model_dump()
        data["confidence"] = round(calibrated, 4)
        data["metadata"] = {
            **data.get("metadata", {}),
            "raw_confidence": round(raw, 4),
            "calibration_method": detection_method,
        }
        return Event(**data)

    return event


def calibrate_events(
    events: list[Event],
    detection_method: str = "motion_heuristic",
    config: CalibrationConfig = _DEFAULT_CONFIG,
) -> list[Event]:
    """Calibrate all events in a list."""
    return [calibrate_event(e, detection_method, config) for e in events]


def compute_calibration_metrics(
    events: list[Event],
    ground_truth_ids: set[str],  # event_ids known to be true positives
) -> dict:
    """
    Compute precision/recall at various confidence thresholds.
    Used to tune calibration parameters after collecting labeled data.
    
    Returns dict with threshold curves and optimal operating point.
    """
    if not events or not ground_truth_ids:
        return {"error": "insufficient_data"}

    thresholds = np.arange(0.40, 0.95, 0.05)
    results = []

    for thresh in thresholds:
        passing = [e for e in events if e.confidence >= thresh]
        tp = sum(1 for e in passing if e.event_id in ground_truth_ids)
        fp = len(passing) - tp
        fn = len(ground_truth_ids) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "threshold": round(float(thresh), 2),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "n_clips": len(passing),
        })

    # Optimal: maximize F1
    best = max(results, key=lambda r: r["f1"])
    return {
        "threshold_curve": results,
        "optimal_threshold": best["threshold"],
        "optimal_f1": best["f1"],
        "total_events": len(events),
        "total_ground_truth": len(ground_truth_ids),
    }
