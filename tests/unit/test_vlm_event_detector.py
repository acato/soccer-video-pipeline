"""
Unit tests for src/detection/vlm_event_detector.py

Tests the two-pass orchestration with mocked FrameSampler and SceneAnalyzer.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.detection.frame_sampler import SampledFrame
from src.detection.models import EventBoundary, EventType, GameState, SceneLabel
from src.detection.vlm_event_detector import VLMEventDetector


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


@pytest.fixture
def detector():
    return VLMEventDetector(
        api_key="test-key-123",
        model="claude-sonnet-4-20250514",
        source_file="/tmp/test-match.mp4",
        video_duration=5400.0,
        job_id="test-job-001",
        event_types=["corner_kick", "goal_kick"],
        frame_interval=3.0,
        frame_width=960,
    )


def _make_frames(timestamps: list[float]) -> list[SampledFrame]:
    return [SampledFrame(timestamp_sec=ts, jpeg_bytes=FAKE_JPEG) for ts in timestamps]


def _make_labels(states: list[tuple[float, str]]) -> list[SceneLabel]:
    return [
        SceneLabel(timestamp_sec=ts, game_state=GameState(state))
        for ts, state in states
    ]


# ---------------------------------------------------------------------------
# Tests: detect()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDetect:

    @patch.object(VLMEventDetector, "_identify_event_regions")
    def test_detect_returns_events_for_confirmed_boundaries(self, mock_regions, detector):
        """Full two-pass flow: sample → scan → identify regions → refine → events."""
        # Mock the sampler
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = _make_frames([0.0, 3.0, 6.0, 9.0])
        mock_sampler.sample_range.return_value = _make_frames([25.0, 26.0, 27.0])
        detector._sampler = mock_sampler

        # Mock the analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.scan.return_value = _make_labels([
            (0.0, "active_play"), (3.0, "corner_kick"),
            (6.0, "active_play"), (9.0, "active_play"),
        ])
        mock_analyzer.refine_event.return_value = EventBoundary(
            event_type="corner_kick",
            clip_start_sec=25.0,
            clip_end_sec=40.0,
            confirmed=True,
            reasoning="Clear corner kick",
        )
        detector._analyzer = mock_analyzer

        mock_regions.return_value = [
            {"event_type": "corner_kick", "center_sec": 3.0, "start_sec": 3.0, "end_sec": 3.0},
        ]

        events = detector.detect()

        assert len(events) == 1
        assert events[0].event_type == EventType.CORNER_KICK
        assert events[0].confidence == 1.0
        assert events[0].timestamp_start == 25.0
        assert events[0].timestamp_end == 40.0
        assert events[0].metadata["vlm_confirmed"] is True
        assert events[0].metadata["detection_method"] == "vlm_two_pass"

    @patch.object(VLMEventDetector, "_identify_event_regions")
    def test_detect_skips_unconfirmed_boundaries(self, mock_regions, detector):
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = _make_frames([0.0, 3.0])
        mock_sampler.sample_range.return_value = _make_frames([3.0, 4.0])
        detector._sampler = mock_sampler

        mock_analyzer = MagicMock()
        mock_analyzer.scan.return_value = _make_labels([(0.0, "active_play"), (3.0, "corner_kick")])
        mock_analyzer.refine_event.return_value = EventBoundary(
            event_type="corner_kick",
            clip_start_sec=25.0, clip_end_sec=40.0,
            confirmed=False,
            reasoning="This is a throw-in, not a corner",
        )
        detector._analyzer = mock_analyzer

        mock_regions.return_value = [
            {"event_type": "corner_kick", "center_sec": 3.0, "start_sec": 3.0, "end_sec": 3.0},
        ]

        events = detector.detect()
        assert len(events) == 0

    @patch.object(VLMEventDetector, "_identify_event_regions")
    def test_detect_no_events_when_no_regions(self, mock_regions, detector):
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = _make_frames([0.0, 3.0, 6.0])
        detector._sampler = mock_sampler

        mock_analyzer = MagicMock()
        mock_analyzer.scan.return_value = _make_labels([
            (0.0, "active_play"), (3.0, "active_play"), (6.0, "active_play"),
        ])
        detector._analyzer = mock_analyzer

        mock_regions.return_value = []

        events = detector.detect()
        assert len(events) == 0

    @patch.object(VLMEventDetector, "_identify_event_regions")
    def test_detect_handles_refine_failure(self, mock_regions, detector):
        """When refine_event returns None, the event is skipped."""
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = _make_frames([0.0, 3.0])
        mock_sampler.sample_range.return_value = _make_frames([3.0])
        detector._sampler = mock_sampler

        mock_analyzer = MagicMock()
        mock_analyzer.scan.return_value = _make_labels([(0.0, "active_play"), (3.0, "corner_kick")])
        mock_analyzer.refine_event.return_value = None  # API failure
        detector._analyzer = mock_analyzer

        mock_regions.return_value = [
            {"event_type": "corner_kick", "center_sec": 3.0, "start_sec": 3.0, "end_sec": 3.0},
        ]

        events = detector.detect()
        assert len(events) == 0

    @patch.object(VLMEventDetector, "_identify_event_regions")
    def test_detect_calls_progress_callback(self, mock_regions, detector):
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = _make_frames([0.0])
        mock_sampler.sample_range.return_value = _make_frames([3.0])
        detector._sampler = mock_sampler

        mock_analyzer = MagicMock()
        mock_analyzer.scan.return_value = _make_labels([(0.0, "active_play")])
        detector._analyzer = mock_analyzer

        mock_regions.return_value = []

        progress_values = []
        events = detector.detect(progress_callback=lambda p: progress_values.append(p))

        assert len(progress_values) >= 2
        assert progress_values[0] == 0.0
        assert progress_values[-1] == 1.0


# ---------------------------------------------------------------------------
# Tests: _identify_event_regions()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIdentifyEventRegions:

    def test_single_corner_kick_frame(self, detector):
        labels = _make_labels([
            (0.0, "active_play"),
            (3.0, "corner_kick"),
            (6.0, "active_play"),
        ])
        regions = detector._identify_event_regions(labels)

        assert len(regions) == 1
        assert regions[0]["event_type"] == "corner_kick"
        assert regions[0]["center_sec"] == 3.0

    def test_consecutive_corner_kick_frames(self, detector):
        labels = _make_labels([
            (0.0, "active_play"),
            (3.0, "corner_kick"),
            (6.0, "corner_kick"),
            (9.0, "corner_kick"),
            (12.0, "active_play"),
        ])
        regions = detector._identify_event_regions(labels)

        assert len(regions) == 1
        assert regions[0]["event_type"] == "corner_kick"
        assert regions[0]["center_sec"] == 6.0  # (3 + 9) / 2

    def test_multiple_different_events(self, detector):
        labels = _make_labels([
            (0.0, "active_play"),
            (3.0, "corner_kick"),
            (6.0, "active_play"),
            (9.0, "goal_kick"),
            (12.0, "active_play"),
        ])
        regions = detector._identify_event_regions(labels)

        assert len(regions) == 2
        assert regions[0]["event_type"] == "corner_kick"
        assert regions[1]["event_type"] == "goal_kick"

    def test_no_events_all_active_play(self, detector):
        labels = _make_labels([
            (0.0, "active_play"),
            (3.0, "active_play"),
            (6.0, "active_play"),
        ])
        regions = detector._identify_event_regions(labels)
        assert len(regions) == 0

    def test_ignores_non_target_states(self, detector):
        labels = _make_labels([
            (0.0, "replay"),
            (3.0, "stoppage"),
            (6.0, "other"),
        ])
        regions = detector._identify_event_regions(labels)
        assert len(regions) == 0

    def test_separate_regions_for_different_event_types(self, detector):
        """corner_kick followed by goal_kick creates two separate regions."""
        labels = _make_labels([
            (0.0, "corner_kick"),
            (3.0, "goal_kick"),
        ])
        regions = detector._identify_event_regions(labels)
        assert len(regions) == 2

    def test_event_at_end_of_labels(self, detector):
        labels = _make_labels([
            (0.0, "active_play"),
            (3.0, "corner_kick"),
        ])
        regions = detector._identify_event_regions(labels)
        assert len(regions) == 1
        assert regions[0]["event_type"] == "corner_kick"


# ---------------------------------------------------------------------------
# Tests: _boundary_to_event()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBoundaryToEvent:

    def test_confirmed_boundary_creates_event(self, detector):
        boundary = EventBoundary(
            event_type="corner_kick",
            clip_start_sec=25.0,
            clip_end_sec=40.0,
            confirmed=True,
            reasoning="Confirmed corner kick",
        )
        event = detector._boundary_to_event(boundary)

        assert event is not None
        assert event.event_type == EventType.CORNER_KICK
        assert event.timestamp_start == 25.0
        assert event.timestamp_end == 40.0
        assert event.confidence == 1.0
        assert event.job_id == "test-job-001"
        assert "corner_kick" in event.reel_targets

    def test_unconfirmed_boundary_returns_none(self, detector):
        boundary = EventBoundary(
            event_type="corner_kick",
            clip_start_sec=25.0, clip_end_sec=40.0,
            confirmed=False,
            reasoning="Not a corner kick",
        )
        event = detector._boundary_to_event(boundary)
        assert event is None

    def test_clamps_to_video_bounds(self, detector):
        boundary = EventBoundary(
            event_type="goal_kick",
            clip_start_sec=-5.0,
            clip_end_sec=6000.0,
            confirmed=True,
        )
        event = detector._boundary_to_event(boundary)

        assert event is not None
        assert event.timestamp_start == 0.0
        assert event.timestamp_end == 5400.0

    def test_invalid_event_type_returns_none(self, detector):
        boundary = EventBoundary(
            event_type="nonexistent_type",
            clip_start_sec=10.0, clip_end_sec=20.0,
            confirmed=True,
        )
        event = detector._boundary_to_event(boundary)
        assert event is None

    def test_zero_duration_clip_returns_none(self, detector):
        boundary = EventBoundary(
            event_type="corner_kick",
            clip_start_sec=10.0, clip_end_sec=10.0,
            confirmed=True,
        )
        event = detector._boundary_to_event(boundary)
        assert event is None
