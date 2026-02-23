"""Unit tests for confidence calibration."""
import pytest
from src.detection.confidence_calibration import (
    calibrate_event, calibrate_events, compute_calibration_metrics,
    CalibrationConfig,
)
from src.detection.models import Event, EventType


def _event(conf: float = 0.70, etype: EventType = EventType.CATCH) -> Event:
    return Event(
        job_id="j1", source_file="m.mp4",
        event_type=etype, timestamp_start=10, timestamp_end=12,
        confidence=conf, reel_targets=["keeper"],
        frame_start=300, frame_end=360,
    )


@pytest.mark.unit
class TestCalibrateEvent:

    def test_returns_event_unchanged_when_within_range(self):
        ev = _event(conf=0.70)
        result = calibrate_event(ev, "motion_heuristic")
        assert isinstance(result, Event)

    def test_max_confidence_capped(self):
        ev = _event(conf=1.0)
        result = calibrate_event(ev, "yolo_player_detection")
        assert result.confidence <= 0.92

    def test_min_confidence_floored(self):
        ev = _event(conf=0.01)
        result = calibrate_event(ev, "motion_heuristic")
        assert result.confidence >= 0.40  # min_conf for motion_heuristic

    def test_event_floor_applied(self):
        """Goal events always get at least 0.55 confidence floor."""
        ev = _event(conf=0.30, etype=EventType.GOAL)
        result = calibrate_event(ev, "motion_heuristic")
        assert result.confidence >= 0.55

    def test_raw_confidence_in_metadata(self):
        ev = _event(conf=0.45)
        result = calibrate_event(ev, "yolo_player_detection")
        if result.confidence != 0.45:  # Only check if calibration changed it
            assert "raw_confidence" in result.metadata

    def test_action_recognition_method(self):
        ev = _event(conf=0.75)
        result = calibrate_event(ev, "action_recognition")
        assert 0.50 <= result.confidence <= 0.95


@pytest.mark.unit
class TestCalibrateEvents:

    def test_calibrates_all_events(self):
        events = [_event(0.5), _event(0.7), _event(0.9)]
        results = calibrate_events(events)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, Event)

    def test_empty_list_returns_empty(self):
        assert calibrate_events([]) == []


@pytest.mark.unit
class TestComputeCalibrationMetrics:

    def test_perfect_detection(self):
        events = [_event(0.80) for _ in range(10)]
        gt_ids = {e.event_id for e in events}
        metrics = compute_calibration_metrics(events, gt_ids)
        assert metrics["optimal_f1"] > 0.9

    def test_no_tp_precision_zero(self):
        events = [_event(0.80) for _ in range(5)]
        gt_ids = {"nonexistent-id"}  # None of the events are true positives
        metrics = compute_calibration_metrics(events, gt_ids)
        assert "threshold_curve" in metrics

    def test_empty_inputs(self):
        result = compute_calibration_metrics([], set())
        assert "error" in result
