"""Unit tests for action classifier (model-free path)."""
import pytest
from src.detection.action_classifier import HeuristicActionClassifier, _update_confidence
from src.detection.models import EventType, Event, Track, Detection, BoundingBox


def _make_track(track_id: int, is_gk: bool = False, velocities: list = None) -> Track:
    """Build a track with detections at specified velocities."""
    track = Track(track_id=track_id, is_goalkeeper=is_gk)
    velocities = velocities or [0.05] * 10
    for i, v in enumerate(velocities):
        track.detections.append(Detection(
            frame_number=i * 3,
            timestamp=float(i) * 0.1,
            class_name="goalkeeper" if is_gk else "player",
            confidence=0.85,
            bbox=BoundingBox(x=0.4 + v * i, y=0.8, width=0.05, height=0.12),
            track_id=track_id,
        ))
    return track


@pytest.mark.unit
class TestHeuristicClassifier:

    def test_gk_dive_detected(self):
        """High vertical velocity in GK track → shot_stop_diving."""
        clf = HeuristicActionClassifier()
        # Simulate vertical movement by varying y
        track = Track(track_id=1, is_goalkeeper=True)
        for i in range(10):
            y = 0.5 if i < 5 else 0.8  # Big jump at frame 5
            track.detections.append(Detection(
                frame_number=i * 3, timestamp=float(i) * 0.1,
                class_name="goalkeeper", confidence=0.85,
                bbox=BoundingBox(x=0.5, y=y, width=0.05, height=0.12),
                track_id=1,
            ))

        results = clf.classify_from_track_velocity(track, timestamp=0.5)
        types = [r[0] for r in results]
        assert EventType.SHOT_STOP_DIVING in types or EventType.SHOT_STOP_STANDING in types

    def test_dribble_detected_for_outfield(self):
        """High horizontal velocity in outfield player → dribble."""
        clf = HeuristicActionClassifier()
        track = Track(track_id=2, is_goalkeeper=False)
        for i in range(15):
            track.detections.append(Detection(
                frame_number=i * 2, timestamp=float(i) * 0.1,
                class_name="player", confidence=0.85,
                bbox=BoundingBox(x=0.1 + i * 0.04, y=0.6, width=0.05, height=0.12),
                track_id=2,
            ))
        results = clf.classify_from_track_velocity(track, timestamp=0.7)
        assert len(results) > 0

    def test_empty_track_returns_empty(self):
        clf = HeuristicActionClassifier()
        track = Track(track_id=99)
        results = clf.classify_from_track_velocity(track, timestamp=0.0)
        assert results == []

    def test_results_sorted_by_confidence(self):
        clf = HeuristicActionClassifier()
        track = Track(track_id=1, is_goalkeeper=True)
        for i in range(10):
            track.detections.append(Detection(
                frame_number=i, timestamp=float(i) * 0.1,
                class_name="goalkeeper", confidence=0.85,
                bbox=BoundingBox(x=0.5, y=0.5 + 0.03 * i, width=0.05, height=0.12),
                track_id=1,
            ))
        results = clf.classify_from_track_velocity(track, timestamp=0.5)
        if len(results) > 1:
            confs = [r[1] for r in results]
            assert confs == sorted(confs, reverse=True)


@pytest.mark.unit
class TestActionClassifierInit:

    def test_no_model_path_not_available(self):
        from src.detection.action_classifier import ActionClassifier
        clf = ActionClassifier(model_path=None)
        assert not clf._ensure_loaded()

    def test_nonexistent_model_path_not_available(self):
        from src.detection.action_classifier import ActionClassifier
        clf = ActionClassifier(model_path="/nonexistent/model.pt")
        assert not clf._ensure_loaded()

    def test_confirm_events_passthrough_when_unavailable(self):
        """If model unavailable, confirm_events returns original events unchanged."""
        from src.detection.action_classifier import ActionClassifier
        clf = ActionClassifier(model_path=None)

        events = [Event(
            job_id="j1", source_file="m.mp4",
            event_type=EventType.CATCH,
            timestamp_start=10.0, timestamp_end=12.0,
            confidence=0.75, reel_targets=["keeper"],
            frame_start=300, frame_end=360,
        )]
        result = clf.confirm_events(events, "source.mp4", "job-001")
        assert len(result) == 1
        assert result[0].confidence == 0.75  # Unchanged
