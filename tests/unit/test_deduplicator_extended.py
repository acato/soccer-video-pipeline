"""
Extended unit tests for deduplicator post-processing edge cases.
"""
import pytest
from src.segmentation.clipper import ClipBoundary
from src.segmentation.deduplicator import (
    deduplicate_clips, enforce_min_duration,
    enforce_max_reel_duration, postprocess_clips, _temporal_iou,
)


def _clip(start: float, end: float, events=None, reel="highlights", n_events=1) -> ClipBoundary:
    return ClipBoundary(
        source_file="match.mp4", start_sec=start, end_sec=end,
        events=events or [f"e{i}" for i in range(n_events)],
        reel_type=reel, primary_event_type="goal",
    )


@pytest.mark.unit
class TestPostprocessClips:
    def test_full_pipeline_runs(self):
        # Clips 0-10 and 0-10 are identical (100% IoU) - deduped to 1.
        # Clip at 50-60 is separate.
        clips = [_clip(0, 10), _clip(0, 10), _clip(50, 60)]
        result = postprocess_clips(clips, "highlights", min_duration=2.0)
        assert isinstance(result, list)
        # Identical clips deduped → 2 unique clips remain
        assert len(result) == 2

    def test_empty_input(self):
        assert postprocess_clips([], "goalkeeper") == []

    def test_single_clip_survives(self):
        clips = [_clip(0, 30)]
        result = postprocess_clips(clips, "goalkeeper", min_duration=2.0)
        assert len(result) == 1

    def test_max_duration_cap_applied(self):
        # 3 clips × 100s = 300s total; cap at 150s → should drop 1
        clips = [_clip(0, 100, ["e1"]), _clip(200, 300, ["e2"]), _clip(400, 500, ["e3"])]
        confidence_map = {"e1": 0.9, "e2": 0.6, "e3": 0.7}
        result = postprocess_clips(
            clips, "highlights",
            max_reel_duration_sec=150,
            event_confidence_map=confidence_map,
        )
        total = sum(c.end_sec - c.start_sec for c in result)
        assert total <= 150

    def test_keeps_more_events_clip_on_dedup(self):
        """When deduplicating, clip with more events is kept."""
        clips = [
            _clip(0, 10, events=["e1"]),             # 1 event
            _clip(0, 10, events=["e2", "e3", "e4"]), # 3 events
        ]
        result = deduplicate_clips(clips, overlap_threshold=0.8)
        assert len(result) == 1
        assert len(result[0].events) == 3


@pytest.mark.unit
class TestEnforceMaxReelDurationEdgeCases:
    def test_all_equal_confidence_drops_last(self):
        clips = [_clip(i * 100, i * 100 + 60) for i in range(5)]  # 5 × 60s = 300s
        result = enforce_max_reel_duration(clips, "highlights", max_duration_sec=180)
        total = sum(c.end_sec - c.start_sec for c in result)
        assert total <= 180

    def test_zero_duration_cap_returns_empty(self):
        clips = [_clip(0, 60)]
        result = enforce_max_reel_duration(clips, "highlights", max_duration_sec=0)
        assert result == []

    def test_result_always_sorted_by_time(self):
        clips = [_clip(200, 260), _clip(0, 60), _clip(400, 460)]
        result = enforce_max_reel_duration(clips, "highlights", max_duration_sec=9999)
        starts = [c.start_sec for c in result]
        assert starts == sorted(starts)
