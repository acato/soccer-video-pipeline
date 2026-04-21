"""Unit tests for src/segmentation/clipper.py"""
import pytest
from src.detection.models import Event, EventType
from src.segmentation.clipper import (
    ClipBoundary,
    clips_stats,
    clips_total_duration,
    compute_clips,
    compute_clips_v2,
)


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

    def test_all_events_included_no_reel_target_filter(self):
        """compute_clips no longer filters by reel_targets — callers pre-filter."""
        events = [
            _make_event("e1", EventType.SHOT_STOP_DIVING, 60.0, 62.0, []),
            _make_event("e2", EventType.GOAL, 200.0, 201.0, [], confidence=0.90),
        ]
        clips = compute_clips(events, 5400.0, "keeper")
        assert len(clips) == 2

    def test_caller_prefilters_events(self):
        """Callers should pre-filter events before passing to compute_clips."""
        all_events = [
            _make_event("e1", EventType.SHOT_STOP_DIVING, 60.0, 62.0, []),
            _make_event("e2", EventType.CATCH, 300.0, 302.0, []),
            _make_event("e3", EventType.GOAL, 500.0, 501.0, [], confidence=0.90),
        ]
        keeper_events = [e for e in all_events if e.event_type != EventType.GOAL]
        gk_clips = compute_clips(keeper_events, 5400.0, "keeper")
        assert len(gk_clips) == 2

    def test_low_confidence_events_excluded(self):
        """GOAL events below 0.50 threshold are excluded."""
        events = [_make_event("e1", EventType.GOAL, 100.0, 101.0, ["highlights"], confidence=0.40)]
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

    def test_max_clip_duration_prevents_mega_merge(self):
        """Many close events should not chain-merge into one mega-clip."""
        events = [
            _make_event(f"e{i}", EventType.SHOT_ON_TARGET, 10.0 * i, 10.0 * i + 2.0, ["highlights"])
            for i in range(20)
        ]
        # With default 90s cap, a 200s chain should be split
        clips = compute_clips(
            events, 5400.0, "highlights",
            pre_pad=3.0, post_pad=5.0, merge_gap_sec=2.0,
            max_clip_duration_sec=90.0,
        )
        assert len(clips) > 1
        for clip in clips:
            assert clip.end_sec - clip.start_sec <= 90.0

    def test_max_clip_duration_no_overlap(self):
        """When max_clip_duration blocks a merge, the new clip must not overlap the previous one."""
        # Three events close together: merge would create a ~24s clip, exceeding 15s cap
        events = [
            _make_event("e1", EventType.SHOT_STOP_DIVING, 100.0, 102.0, ["keeper"]),
            _make_event("e2", EventType.SHOT_STOP_STANDING, 108.0, 110.0, ["keeper"]),
            _make_event("e3", EventType.SHOT_STOP_DIVING, 116.0, 118.0, ["keeper"]),
        ]
        clips = compute_clips(
            events, 5400.0, "keeper",
            pre_pad=1.5, post_pad=1.5, merge_gap_sec=2.0,
            max_clip_duration_sec=15.0,
        )
        assert len(clips) >= 2, "Should split due to duration cap"
        # Verify no overlaps
        for i in range(1, len(clips)):
            assert clips[i].start_sec >= clips[i - 1].end_sec, (
                f"Clip {i} starts at {clips[i].start_sec} before clip {i-1} ends at {clips[i-1].end_sec}"
            )

    def test_primary_event_is_highest_confidence(self):
        events = [
            _make_event("e1", EventType.SHOT_ON_TARGET, 100.0, 101.0, ["highlights"], confidence=0.70),
            _make_event("e2", EventType.GOAL, 103.0, 104.0, ["highlights"], confidence=0.92),
        ]
        clips = compute_clips(events, 5400.0, "highlights", pre_pad=3.0, post_pad=5.0, merge_gap_sec=5.0)
        assert clips[0].primary_event_type == "goal"


@pytest.mark.unit
class TestParryAwareClipping:
    """Parry-tagged events get extra post-padding, and the clip inherits
    a primary_signature that summarizes trajectory tags on its events."""

    def _gk_event(self, event_id, start, end, signature=None, conf=0.85):
        e = _make_event(event_id, EventType.SHOT_STOP_DIVING, start, end,
                        ["keeper"], confidence=conf)
        if signature is not None:
            e.metadata["trajectory_signature"] = signature
        return e

    def test_parry_event_gets_bonus_post_pad(self):
        # Without signature: use EVENT_TYPE_CONFIG post_pad (shot_stop_diving).
        baseline = compute_clips_v2(
            [self._gk_event("e1", 100.0, 102.0, signature=None)],
            video_duration=5400.0, reel_name="keeper",
        )
        with_parry = compute_clips_v2(
            [self._gk_event("e2", 100.0, 102.0, signature="parry")],
            video_duration=5400.0, reel_name="keeper",
        )
        # Parry clip should extend 3s further past event end.
        parry_end = with_parry[0].end_sec
        baseline_end = baseline[0].end_sec
        assert parry_end == pytest.approx(baseline_end + 3.0)

    def test_non_parry_signature_no_bonus_pad(self):
        # "catch" signature — no bonus pad
        baseline = compute_clips_v2(
            [self._gk_event("e1", 100.0, 102.0, signature=None)],
            video_duration=5400.0, reel_name="keeper",
        )
        with_catch = compute_clips_v2(
            [self._gk_event("e2", 100.0, 102.0, signature="catch")],
            video_duration=5400.0, reel_name="keeper",
        )
        assert with_catch[0].end_sec == baseline[0].end_sec

    def test_primary_signature_empty_when_no_tags(self):
        clips = compute_clips_v2(
            [self._gk_event("e1", 100.0, 102.0, signature=None)],
            video_duration=5400.0, reel_name="keeper",
        )
        assert clips[0].primary_signature is None

    def test_primary_signature_single_event(self):
        for sig in ("parry", "catch", "deflection", "missed", "insufficient_data"):
            clips = compute_clips_v2(
                [self._gk_event(f"e1_{sig}", 100.0, 102.0, signature=sig)],
                video_duration=5400.0, reel_name="keeper",
            )
            assert clips[0].primary_signature == sig, (
                f"Expected {sig} on clip, got {clips[0].primary_signature}"
            )

    def test_primary_signature_priority_parry_wins(self):
        # Two events merged into one clip: one parry, one catch → parry.
        events = [
            self._gk_event("e_catch", 100.0, 102.0, signature="catch"),
            self._gk_event("e_parry", 103.0, 105.0, signature="parry"),
        ]
        clips = compute_clips_v2(
            events, video_duration=5400.0, reel_name="keeper",
        )
        # Should merge (within merge_gap_sec=2.0)
        assert len(clips) == 1
        assert clips[0].primary_signature == "parry"

    def test_primary_signature_priority_catch_beats_deflection(self):
        events = [
            self._gk_event("e_def", 100.0, 102.0, signature="deflection"),
            self._gk_event("e_catch", 103.0, 105.0, signature="catch"),
        ]
        clips = compute_clips_v2(
            events, video_duration=5400.0, reel_name="keeper",
        )
        assert len(clips) == 1
        assert clips[0].primary_signature == "catch"


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
