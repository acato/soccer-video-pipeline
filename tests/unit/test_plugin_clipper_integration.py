"""
Integration tests: plugin event selection → compute_clips → clip boundaries.

Verifies the full plugin → clipper path without video, FFmpeg, or ML models.
"""
from __future__ import annotations

import pytest

from src.detection.models import Event, EventType
from src.reel_plugins.base import PipelineContext
from src.reel_plugins.keeper import KeeperSavesPlugin, KeeperGoalKickPlugin, KeeperDistributionPlugin
from src.reel_plugins.highlights import HighlightsShotsPlugin
from src.reel_plugins.registry import PluginRegistry
from src.segmentation.clipper import compute_clips
from src.segmentation.deduplicator import postprocess_clips
from tests.conftest import make_match_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_type: EventType,
    start: float,
    end: float | None = None,
    reel_targets: list[str] | None = None,
    confidence: float = 0.85,
    is_gk: bool = False,
) -> Event:
    end = end or start + 1.0
    return Event(
        event_id=f"evt-{event_type.value}-{start}",
        job_id="test-job",
        source_file="match.mp4",
        event_type=event_type,
        timestamp_start=start,
        timestamp_end=end,
        confidence=confidence,
        reel_targets=reel_targets or [],
        is_goalkeeper_event=is_gk,
        frame_start=int(start * 30),
        frame_end=int(end * 30),
    )


def _make_ctx(duration: float = 5400.0) -> PipelineContext:
    return PipelineContext(
        video_duration_sec=duration,
        match_config=make_match_config(),
        keeper_track_ids={"keeper_a": 1, "keeper_b": None},
        job_id="test-job",
    )


def _run_plugin_to_clips(plugin, events, ctx=None):
    """Run a single plugin's full path: select → compute_clips → return."""
    ctx = ctx or _make_ctx()
    selected = plugin.select_events(events, ctx)
    if not selected:
        return []
    p = plugin.clip_params
    return compute_clips(
        events=selected,
        video_duration=ctx.video_duration_sec,
        reel_type=plugin.reel_name,
        pre_pad=p.pre_pad_sec,
        post_pad=p.post_pad_sec,
        merge_gap_sec=p.merge_gap_sec,
        max_clip_duration_sec=p.max_clip_duration_sec,
    )


# ===========================================================================
# Single plugin → clips
# ===========================================================================

@pytest.mark.unit
class TestKeeperSavesClips:
    def test_single_save_clip_boundaries(self):
        events = [
            _make_event(EventType.SHOT_STOP_DIVING, start=30.0, reel_targets=["keeper"], is_gk=True),
        ]
        clips = _run_plugin_to_clips(KeeperSavesPlugin(), events)
        assert len(clips) == 1
        assert clips[0].start_sec == pytest.approx(25.0)  # 30.0 - 5.0
        assert clips[0].end_sec == pytest.approx(32.5)    # 31.0 + 1.5

    def test_two_distant_saves_produce_two_clips(self):
        events = [
            _make_event(EventType.SHOT_STOP_DIVING, start=30.0, reel_targets=["keeper"], is_gk=True),
            _make_event(EventType.SHOT_STOP_STANDING, start=300.0, reel_targets=["keeper"], is_gk=True),
        ]
        clips = _run_plugin_to_clips(KeeperSavesPlugin(), events)
        assert len(clips) == 2

    def test_two_close_saves_merge(self):
        events = [
            _make_event(EventType.SHOT_STOP_DIVING, start=30.0, reel_targets=["keeper"], is_gk=True),
            _make_event(EventType.CATCH, start=32.0, reel_targets=["keeper"], is_gk=True),
        ]
        clips = _run_plugin_to_clips(KeeperSavesPlugin(), events)
        # gap after padding: (32.0-5.0) - (31.0+1.5) = 27.0 - 32.5 < 0 → overlap → merge
        assert len(clips) == 1

    def test_no_matching_events_returns_empty(self):
        events = [
            _make_event(EventType.GOAL, start=50.0, reel_targets=["highlights"]),
        ]
        clips = _run_plugin_to_clips(KeeperSavesPlugin(), events)
        assert clips == []


@pytest.mark.unit
class TestKeeperGoalKickClips:
    def test_goal_kick_clip_boundaries(self):
        events = [
            _make_event(EventType.GOAL_KICK, start=60.0, reel_targets=["keeper"], is_gk=True),
        ]
        clips = _run_plugin_to_clips(KeeperGoalKickPlugin(), events)
        assert len(clips) == 1
        assert clips[0].start_sec == pytest.approx(59.5)  # 60.0 - 0.5
        assert clips[0].end_sec == pytest.approx(67.0)    # 61.0 + 6.0


@pytest.mark.unit
class TestKeeperDistributionClips:
    def test_distribution_clip_boundaries(self):
        events = [
            _make_event(EventType.DISTRIBUTION_SHORT, start=60.0, reel_targets=["keeper"], is_gk=True),
        ]
        clips = _run_plugin_to_clips(KeeperDistributionPlugin(), events)
        assert len(clips) == 1
        assert clips[0].start_sec == pytest.approx(59.0)  # 60.0 - 1.0
        assert clips[0].end_sec == pytest.approx(66.0)    # 61.0 + 5.0


@pytest.mark.unit
class TestHighlightsClips:
    def test_shot_clip_has_wide_padding(self):
        events = [
            _make_event(EventType.SHOT_ON_TARGET, start=100.0, reel_targets=["highlights"]),
        ]
        clips = _run_plugin_to_clips(HighlightsShotsPlugin(), events)
        assert len(clips) == 1
        assert clips[0].start_sec == pytest.approx(97.0)   # 100.0 - 3.0
        assert clips[0].end_sec == pytest.approx(106.0)     # 101.0 + 5.0


# ===========================================================================
# Multiple plugins → same reel → merged
# ===========================================================================

@pytest.mark.unit
class TestMultiPluginMerge:
    def test_saves_and_distribution_overlap_kept_as_separate_clips(self):
        """Two plugins contribute to 'keeper' reel; close events with partial overlap
        are kept as separate clips (postprocess deduplicates near-identical clips,
        not partially overlapping ones)."""
        events = [
            _make_event(EventType.SHOT_STOP_DIVING, start=30.0, reel_targets=["keeper"], is_gk=True),
            _make_event(EventType.DISTRIBUTION_SHORT, start=33.0, reel_targets=["keeper"], is_gk=True),
        ]
        ctx = _make_ctx()
        saves_clips = _run_plugin_to_clips(KeeperSavesPlugin(), events, ctx)
        dist_clips = _run_plugin_to_clips(KeeperDistributionPlugin(), events, ctx)

        all_clips = saves_clips + dist_clips
        all_clips.sort(key=lambda c: c.start_sec)
        merged = postprocess_clips(all_clips, reel_type="keeper")
        # Both clips survive: they overlap partially but IoU < 0.8
        assert len(merged) == 2
        assert merged[0].primary_event_type == "shot_stop_diving"
        assert merged[1].primary_event_type == "distribution_short"

    def test_saves_and_distribution_separate_when_distant(self):
        events = [
            _make_event(EventType.SHOT_STOP_DIVING, start=30.0, reel_targets=["keeper"], is_gk=True),
            _make_event(EventType.DISTRIBUTION_SHORT, start=300.0, reel_targets=["keeper"], is_gk=True),
        ]
        ctx = _make_ctx()
        saves_clips = _run_plugin_to_clips(KeeperSavesPlugin(), events, ctx)
        dist_clips = _run_plugin_to_clips(KeeperDistributionPlugin(), events, ctx)

        all_clips = saves_clips + dist_clips
        all_clips.sort(key=lambda c: c.start_sec)
        merged = postprocess_clips(all_clips, reel_type="keeper")
        assert len(merged) == 2


# ===========================================================================
# Full registry round-trip
# ===========================================================================

@pytest.mark.unit
class TestRegistryRoundTrip:
    def test_default_registry_produces_clips_for_both_reels(self):
        """Default registry with keeper + highlights events produces both reels."""
        events = [
            _make_event(EventType.SHOT_STOP_DIVING, start=30.0, reel_targets=["keeper"], is_gk=True),
            _make_event(EventType.GOAL, start=200.0, reel_targets=["highlights"], confidence=0.92),
        ]
        registry = PluginRegistry.default()
        ctx = _make_ctx()

        clips_by_reel: dict[str, list] = {}
        for reel_name in registry.get_all_reel_names():
            plugins = registry.get_plugins_for_reel(reel_name)
            reel_clips = []
            for plugin in plugins:
                reel_clips.extend(_run_plugin_to_clips(plugin, events, ctx))
            reel_clips.sort(key=lambda c: c.start_sec)
            clips_by_reel[reel_name] = reel_clips

        assert len(clips_by_reel["keeper"]) == 1
        assert len(clips_by_reel["highlights"]) == 1
        assert clips_by_reel["keeper"][0].reel_type == "keeper"
        assert clips_by_reel["highlights"][0].reel_type == "highlights"
