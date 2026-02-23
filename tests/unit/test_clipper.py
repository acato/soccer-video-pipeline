"""Unit tests for src/segmentation/clipper.py"""
import pytest
from src.detection.models import Event, EventType
from src.segmentation.clipper import ClipBoundary, compute_clips, clips_total_duration, clips_stats


def _make_event(
    event_id: str,
    event_type: EventType,
    start: float,
    end: float,
    reel_targets: list[str],
    confidence: float = 0.80,
) -> Event:
    return Event(
        event_id=event_id,
        job_id="job-001",
        source_file="match.mp4",
        event_type=event_type,
        timestamp_start=start,
        timestamp_end=end,
        confidence=confidence,
        reel_targets=reel_targets,
        frame_start=int(start * 30),
        frame_end=int(end * 30),
    )


@pytest.mark.unit
class TestComputeClips:
    def test_empty_events_returns_empty(self):
        clips = compute_clips([], 5400.0, "keeper")
        assert clips == []

    def test_single_event_with_padding(self):
        events = [_make_event("e1", EventType.SHOT_STOP_DIVING, 60.0, 62.0, ["keeper"])]
        clips = compute_clips(events, 5400.0, "keeper", pre_pad=3.0, post_pad=5.0)
        assert len(clips) == 1
        assert clips[0].start_sec == 57.0
        assert clips[0].end_sec == 67.0

    def test_padding_clamped_at_zero(self):
        events = [_make_event("e1", EventType.SHOT_STOP_DIVING, 1.0, 2.0, ["keeper"])]
        clips = compute_clips(events, 5400.0, "keeper", pre_pad=5.0, post_pad=5.0)
        assert clips[0].start_sec == 0.0

    def test_padding_clamped_at_video_end(self):
        # GOAL requires confidence >= 0.85
        events = [_make_event("e1", EventType.GOAL, 5398.0, 5399.0, ["highlights"], confidence=0.90)]
        clips = compute_clips(events, 5400.0, "highlights", pre_pad=3.0, post_pad=10.0)
        assert len(clips) == 1
        assert clips[0].end_sec == 5400.0

    def test_overlapping_events_merged(self):
        events = [
            _make_event("e1", EventType.SHOT_ON_TARGET, 100.0, 102.0, ["highlights"]),
            _make_event("e2", EventType.GOAL, 104.0, 105.0, ["highlights"], confidence=0.90),
        ]
        clips = compute_clips(events, 5400.0, "highlights", pre_pad=3.0, post_pad=5.0, merge_gap_sec=2.0)
        assert len(clips) == 1
        assert clips[0].start_sec == 97.0
        assert clips[0].end_sec == 110.0

    def test_non_overlapping_events_not_merged(self):
        events = [
            _make_event("e1", EventType.SHOT_ON_TARGET, 100.0, 102.0, ["highlights"]),
            _make_event("e2", EventType.GOAL, 200.0, 201.0, ["highlights"], confidence=0.90),
        ]
        clips = compute_clips(events, 5400.0, "highlights", pre_pad=3.0, post_pad=5.0, merge_gap_sec=2.0)
        assert len(clips) == 2

    def test_reel_type_filter(self):
        """Events not in reel_targets should be excluded."""
        events = [
            _make_event("e1", EventType.SHOT_STOP_DIVING, 60.0, 62.0, ["keeper"]),
            _make_event("e2", EventType.GOAL, 200.0, 201.0, ["highlights"], confidence=0.90),
        ]
        gk_clips = compute_clips(events, 5400.0, "keeper")
        hl_clips = compute_clips(events, 5400.0, "highlights")
        assert len(gk_clips) == 1
        assert len(hl_clips) == 1
        assert gk_clips[0].primary_event_type == "shot_stop_diving"
        assert hl_clips[0].primary_event_type == "goal"

    def test_low_confidence_events_excluded(self):
        """GOAL events below 0.85 threshold are excluded."""
        events = [_make_event("e1", EventType.GOAL, 100.0, 101.0, ["highlights"], confidence=0.50)]
        clips = compute_clips(events, 5400.0, "highlights")
        assert len(clips) == 0

    def test_merged_clip_covers_all_event_ids(self):
        events = [
            _make_event("e1", EventType.SHOT_ON_TARGET, 100.0, 101.0, ["highlights"]),
            _make_event("e2", EventType.SHOT_ON_TARGET, 103.0, 104.0, ["highlights"]),
        ]
        clips = compute_clips(events, 5400.0, "highlights", pre_pad=3.0, post_pad=5.0, merge_gap_sec=5.0)
        assert len(clips) == 1
        assert "e1" in clips[0].events
        assert "e2" in clips[0].events

    def test_primary_event_is_highest_confidence(self):
        events = [
            _make_event("e1", EventType.SHOT_ON_TARGET, 100.0, 101.0, ["highlights"], confidence=0.70),
            _make_event("e2", EventType.GOAL, 103.0, 104.0, ["highlights"], confidence=0.92),
        ]
        clips = compute_clips(events, 5400.0, "highlights", pre_pad=3.0, post_pad=5.0, merge_gap_sec=5.0)
        assert clips[0].primary_event_type == "goal"


@pytest.mark.unit
class TestClipsStats:
    def test_empty_clips(self):
        stats = clips_stats([])
        assert stats["count"] == 0
        assert stats["total_duration_sec"] == 0

    def test_duration_calculation(self):
        clips = [
            ClipBoundary(source_file="f.mp4", start_sec=0, end_sec=10,
                        events=[], reel_type="highlights", primary_event_type="goal"),
            ClipBoundary(source_file="f.mp4", start_sec=20, end_sec=35,
                        events=[], reel_type="highlights", primary_event_type="shot_on_target"),
        ]
        assert clips_total_duration(clips) == 25.0
        stats = clips_stats(clips)
        assert stats["count"] == 2
        assert stats["total_duration_sec"] == 25.0
        assert stats["avg_duration_sec"] == 12.5
