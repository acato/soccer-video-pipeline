"""Unit tests for src/segmentation/deduplicator.py"""
import pytest
from src.segmentation.clipper import ClipBoundary
from src.segmentation.deduplicator import (
    deduplicate_clips,
    enforce_min_duration,
    enforce_max_reel_duration,
    postprocess_clips,
    _temporal_iou,
)


def _clip(start: float, end: float, events: list[str] = None, reel_type="highlights") -> ClipBoundary:
    return ClipBoundary(
        source_file="match.mp4",
        start_sec=start,
        end_sec=end,
        events=events or ["e1"],
        reel_type=reel_type,
        primary_event_type="goal",
    )


@pytest.mark.unit
class TestTemporalIoU:
    def test_no_overlap(self):
        assert _temporal_iou(_clip(0, 10), _clip(20, 30)) == 0.0

    def test_identical_clips(self):
        assert _temporal_iou(_clip(0, 10), _clip(0, 10)) == 1.0

    def test_partial_overlap(self):
        iou = _temporal_iou(_clip(0, 10), _clip(5, 15))
        # intersection=5, union=15
        assert abs(iou - 5/15) < 0.01

    def test_one_contains_other(self):
        iou = _temporal_iou(_clip(0, 20), _clip(5, 10))
        # intersection=5, union=20
        assert abs(iou - 5/20) < 0.01


@pytest.mark.unit
class TestDeduplicateClips:
    def test_no_duplicates_unchanged(self):
        clips = [_clip(0, 10), _clip(20, 30), _clip(40, 50)]
        result = deduplicate_clips(clips)
        assert len(result) == 3

    def test_identical_clips_deduplicated(self):
        clips = [_clip(0, 10, ["e1"]), _clip(0, 10, ["e1", "e2"])]
        result = deduplicate_clips(clips, overlap_threshold=0.8)
        assert len(result) == 1
        # Should keep the one with more events
        assert len(result[0].events) == 2

    def test_high_overlap_deduplicated(self):
        clips = [_clip(0, 10), _clip(1, 11)]  # IoU = 9/11 ≈ 0.82
        result = deduplicate_clips(clips, overlap_threshold=0.8)
        assert len(result) == 1

    def test_low_overlap_not_deduplicated(self):
        clips = [_clip(0, 10), _clip(8, 20)]  # IoU = 2/20 = 0.10
        result = deduplicate_clips(clips, overlap_threshold=0.8)
        assert len(result) == 2


@pytest.mark.unit
class TestEnforceMinDuration:
    def test_short_clips_removed(self):
        clips = [_clip(0, 1), _clip(10, 15), _clip(20, 21)]
        result = enforce_min_duration(clips, min_duration=2.0)
        assert len(result) == 1
        assert result[0].start_sec == 10.0

    def test_all_valid_clips_kept(self):
        clips = [_clip(0, 5), _clip(10, 20)]
        result = enforce_min_duration(clips, min_duration=2.0)
        assert len(result) == 2


@pytest.mark.unit
class TestEnforceMaxReelDuration:
    def test_under_limit_unchanged(self):
        clips = [_clip(0, 60), _clip(100, 160)]  # 2 min total
        result = enforce_max_reel_duration(clips, "highlights", max_duration_sec=3600)
        assert len(result) == 2

    def test_over_limit_pruned(self):
        clips = [
            _clip(0, 300, ["high_conf"]),   # 5 min
            _clip(400, 700, ["low_conf"]),   # 5 min
        ]
        confidence_map = {"high_conf": 0.95, "low_conf": 0.55}
        result = enforce_max_reel_duration(
            clips, "highlights",
            max_duration_sec=360,  # Only 6 min allowed
            event_confidence_map=confidence_map,
        )
        # Should keep the high-confidence clip (5 min) and drop low-confidence
        assert len(result) == 1
        assert "high_conf" in result[0].events

    def test_result_is_time_sorted(self):
        clips = [_clip(100, 200), _clip(0, 50), _clip(300, 400)]
        result = enforce_max_reel_duration(clips, "highlights", max_duration_sec=9999)
        starts = [c.start_sec for c in result]
        assert starts == sorted(starts)
