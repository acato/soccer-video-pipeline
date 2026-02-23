"""Unit tests for src/detection/models.py"""
import pytest
from src.detection.models import (
    BoundingBox, Detection, Event, EventType,
    EVENT_REEL_MAP, EVENT_CONFIDENCE_THRESHOLDS, Track,
    GK_REEL_TYPES, is_gk_event_type,
)


@pytest.mark.unit
class TestBoundingBox:
    def test_center_calculation(self):
        bbox = BoundingBox(x=0.1, y=0.2, width=0.4, height=0.6)
        assert abs(bbox.center_x - 0.3) < 0.001  # 0.1 + 0.4/2
        assert abs(bbox.center_y - 0.5) < 0.001  # 0.2 + 0.6/2

    def test_area_calculation(self):
        bbox = BoundingBox(x=0, y=0, width=0.5, height=0.4)
        assert abs(bbox.area - 0.2) < 0.001


@pytest.mark.unit
class TestEventReelMap:
    def test_all_event_types_in_map(self):
        for event_type in EventType:
            assert event_type in EVENT_REEL_MAP, f"{event_type} missing from EVENT_REEL_MAP"

    def test_goal_is_highlights_only(self):
        targets = EVENT_REEL_MAP[EventType.GOAL]
        assert "highlights" in targets
        assert len(targets) == 1

    def test_gk_events_have_empty_default_reel_targets(self):
        """GK event types have empty reel_targets in the map (assigned dynamically)."""
        gk_only_events = [
            EventType.SHOT_STOP_DIVING,
            EventType.SHOT_STOP_STANDING,
            EventType.PUNCH,
            EventType.CATCH,
            EventType.GOAL_KICK,
            EventType.DISTRIBUTION_SHORT,
            EventType.DISTRIBUTION_LONG,
        ]
        for et in gk_only_events:
            assert EVENT_REEL_MAP[et] == [], f"{et} should have empty default reel_targets"

    def test_is_gk_event_type(self):
        assert is_gk_event_type(EventType.SHOT_STOP_DIVING)
        assert is_gk_event_type(EventType.ONE_ON_ONE)
        assert not is_gk_event_type(EventType.GOAL)
        assert not is_gk_event_type(EventType.TACKLE)

    def test_gk_reel_types(self):
        assert GK_REEL_TYPES == ("keeper",)


@pytest.mark.unit
class TestEventShouldInclude:
    def test_goal_requires_high_confidence(self):
        def make_goal(confidence):
            return Event(
                event_id="e1", job_id="j1", source_file="f.mp4",
                event_type=EventType.GOAL,
                timestamp_start=10.0, timestamp_end=11.0,
                confidence=confidence,
                reel_targets=["highlights"],
                frame_start=300, frame_end=330,
            )
        assert not make_goal(0.84).should_include()  # Below 0.85 threshold
        assert make_goal(0.85).should_include()
        assert make_goal(0.90).should_include()

    def test_review_override_true_forces_include(self):
        event = Event(
            event_id="e1", job_id="j1", source_file="f.mp4",
            event_type=EventType.GOAL,
            timestamp_start=10.0, timestamp_end=11.0,
            confidence=0.10,  # Very low
            reel_targets=["highlights"],
            frame_start=300, frame_end=330,
            review_override=True,
        )
        assert event.should_include()

    def test_review_override_false_forces_exclude(self):
        event = Event(
            event_id="e1", job_id="j1", source_file="f.mp4",
            event_type=EventType.SHOT_STOP_DIVING,
            timestamp_start=10.0, timestamp_end=11.0,
            confidence=0.99,  # Very high
            reel_targets=["keeper"],
            frame_start=300, frame_end=330,
            review_override=False,
        )
        assert not event.should_include()

    def test_duration_property(self):
        event = Event(
            event_id="e1", job_id="j1", source_file="f.mp4",
            event_type=EventType.TACKLE,
            timestamp_start=10.0, timestamp_end=13.5,
            confidence=0.70,
            reel_targets=["highlights"],
            frame_start=300, frame_end=405,
        )
        assert abs(event.duration_sec - 3.5) < 0.001


@pytest.mark.unit
class TestTrack:
    def test_start_end_frame_from_detections(self):
        from src.detection.models import Detection
        track = Track(track_id=5, detections=[
            Detection(frame_number=100, timestamp=3.33, class_name="player",
                     confidence=0.9, bbox=BoundingBox(x=0.1, y=0.1, width=0.1, height=0.2)),
            Detection(frame_number=150, timestamp=5.0, class_name="player",
                     confidence=0.85, bbox=BoundingBox(x=0.15, y=0.1, width=0.1, height=0.2)),
        ])
        assert track.start_frame == 100
        assert track.end_frame == 150
        assert track.duration_frames == 50

    def test_empty_track(self):
        track = Track(track_id=1)
        assert track.start_frame == 0
        assert track.end_frame == 0
        assert track.duration_frames == 0
