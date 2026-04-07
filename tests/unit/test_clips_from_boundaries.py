"""
Unit tests for clips_from_boundaries() in src/segmentation/clipper.py
"""
import pytest

from src.detection.models import EventBoundary
from src.segmentation.clipper import ClipBoundary, clips_from_boundaries


def _make_boundary(event_type="corner_kick", start=10.0, end=25.0, confirmed=True):
    return EventBoundary(
        event_type=event_type,
        clip_start_sec=start,
        clip_end_sec=end,
        confirmed=confirmed,
        reasoning="test",
    )


@pytest.mark.unit
class TestClipsFromBoundaries:

    def test_single_boundary_creates_single_clip(self):
        boundaries = [_make_boundary(start=10.0, end=25.0)]
        clips = clips_from_boundaries(boundaries, "/tmp/match.mp4", "corner_kick")

        assert len(clips) == 1
        assert clips[0].start_sec == 10.0
        assert clips[0].end_sec == 25.0
        assert clips[0].reel_type == "corner_kick"
        assert clips[0].source_file == "/tmp/match.mp4"
        assert clips[0].primary_event_type == "corner_kick"

    def test_empty_boundaries_returns_empty(self):
        clips = clips_from_boundaries([], "/tmp/match.mp4", "corner_kick")
        assert clips == []

    def test_merges_overlapping_clips(self):
        boundaries = [
            _make_boundary(start=10.0, end=25.0),
            _make_boundary(start=20.0, end=35.0),
        ]
        clips = clips_from_boundaries(boundaries, "/tmp/match.mp4", "corner_kick")

        assert len(clips) == 1
        assert clips[0].start_sec == 10.0
        assert clips[0].end_sec == 35.0

    def test_merges_clips_within_gap(self):
        """Clips less than merge_gap_sec apart get merged."""
        boundaries = [
            _make_boundary(start=10.0, end=20.0),
            _make_boundary(start=22.0, end=30.0),  # 2s gap, < default 3s
        ]
        clips = clips_from_boundaries(boundaries, "/tmp/match.mp4", "corner_kick")

        assert len(clips) == 1
        assert clips[0].start_sec == 10.0
        assert clips[0].end_sec == 30.0

    def test_does_not_merge_clips_beyond_gap(self):
        """Clips more than merge_gap_sec apart stay separate."""
        boundaries = [
            _make_boundary(start=10.0, end=20.0),
            _make_boundary(start=30.0, end=40.0),  # 10s gap, > default 3s
        ]
        clips = clips_from_boundaries(boundaries, "/tmp/match.mp4", "corner_kick")

        assert len(clips) == 2
        assert clips[0].start_sec == 10.0
        assert clips[1].start_sec == 30.0

    def test_caps_merged_clip_at_max_duration(self):
        """Merged clips are capped at max_clip_sec."""
        boundaries = [
            _make_boundary(start=0.0, end=50.0),
            _make_boundary(start=51.0, end=100.0),  # Would merge to 100s
        ]
        clips = clips_from_boundaries(
            boundaries, "/tmp/match.mp4", "corner_kick", max_clip_sec=60.0,
        )

        assert len(clips) == 2  # Not merged because 100s > 60s cap

    def test_sorts_by_start_time(self):
        """Boundaries given out of order are sorted."""
        boundaries = [
            _make_boundary(start=30.0, end=40.0),
            _make_boundary(start=10.0, end=20.0),
        ]
        clips = clips_from_boundaries(boundaries, "/tmp/match.mp4", "corner_kick")

        assert clips[0].start_sec == 10.0
        assert clips[1].start_sec == 30.0

    def test_multiple_event_types(self):
        """Different event types are handled correctly."""
        boundaries = [
            _make_boundary(event_type="corner_kick", start=10.0, end=25.0),
            _make_boundary(event_type="goal_kick", start=50.0, end=65.0),
        ]
        clips = clips_from_boundaries(boundaries, "/tmp/match.mp4", "mixed")

        assert len(clips) == 2
        assert clips[0].primary_event_type == "corner_kick"
        assert clips[1].primary_event_type == "goal_kick"

    def test_custom_merge_gap(self):
        boundaries = [
            _make_boundary(start=10.0, end=20.0),
            _make_boundary(start=21.0, end=30.0),  # 1s gap
        ]
        # merge_gap_sec=0.5 → should NOT merge
        clips = clips_from_boundaries(
            boundaries, "/tmp/match.mp4", "corner_kick", merge_gap_sec=0.5,
        )
        assert len(clips) == 2
