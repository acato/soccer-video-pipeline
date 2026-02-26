"""
Unit tests for ReelPlugin ABC, ClipParams, and PipelineContext.
"""
from __future__ import annotations

import pytest

from src.reel_plugins.base import ClipParams, PipelineContext, ReelPlugin
from src.detection.models import Event
from src.segmentation.clipper import ClipBoundary
from tests.conftest import make_match_config


# ---------------------------------------------------------------------------
# Concrete test plugin (minimal implementation of the ABC)
# ---------------------------------------------------------------------------

class _DummyPlugin(ReelPlugin):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def reel_name(self) -> str:
        return "test_reel"

    def select_events(self, events, ctx):
        return events  # pass-through


# ===========================================================================
# ClipParams
# ===========================================================================

@pytest.mark.unit
class TestClipParams:
    def test_defaults(self):
        p = ClipParams()
        assert p.pre_pad_sec == 3.0
        assert p.post_pad_sec == 3.0
        assert p.merge_gap_sec == 2.0
        assert p.max_clip_duration_sec == 60.0
        assert p.max_reel_duration_sec == 1200.0
        assert p.min_clip_duration_sec == 2.0

    def test_custom_values(self):
        p = ClipParams(pre_pad_sec=1.0, post_pad_sec=2.0, max_clip_duration_sec=15.0)
        assert p.pre_pad_sec == 1.0
        assert p.post_pad_sec == 2.0
        assert p.max_clip_duration_sec == 15.0

    def test_frozen(self):
        p = ClipParams()
        with pytest.raises(AttributeError):
            p.pre_pad_sec = 5.0  # type: ignore[misc]

    def test_negative_pre_pad_raises(self):
        with pytest.raises(ValueError, match="pre_pad_sec"):
            ClipParams(pre_pad_sec=-1.0)

    def test_negative_post_pad_raises(self):
        with pytest.raises(ValueError, match="post_pad_sec"):
            ClipParams(post_pad_sec=-0.5)

    def test_zero_max_clip_duration_raises(self):
        with pytest.raises(ValueError, match="max_clip_duration_sec"):
            ClipParams(max_clip_duration_sec=0)

    def test_negative_max_reel_duration_raises(self):
        with pytest.raises(ValueError, match="max_reel_duration_sec"):
            ClipParams(max_reel_duration_sec=-100)

    def test_negative_min_clip_duration_raises(self):
        with pytest.raises(ValueError, match="min_clip_duration_sec"):
            ClipParams(min_clip_duration_sec=-1)

    def test_zero_padding_is_valid(self):
        p = ClipParams(pre_pad_sec=0.0, post_pad_sec=0.0, min_clip_duration_sec=0.0)
        assert p.pre_pad_sec == 0.0
        assert p.post_pad_sec == 0.0


# ===========================================================================
# PipelineContext
# ===========================================================================

@pytest.mark.unit
class TestPipelineContext:
    def test_construction(self):
        ctx = PipelineContext(
            video_duration_sec=5400.0,
            match_config=make_match_config(),
            keeper_track_ids={"keeper_a": 42, "keeper_b": None},
            job_id="test-job",
        )
        assert ctx.video_duration_sec == 5400.0
        assert ctx.job_id == "test-job"
        assert ctx.keeper_track_ids["keeper_a"] == 42

    def test_frozen(self):
        ctx = PipelineContext(
            video_duration_sec=5400.0,
            match_config=None,
            keeper_track_ids={},
            job_id="job-1",
        )
        with pytest.raises(AttributeError):
            ctx.video_duration_sec = 0.0  # type: ignore[misc]

    def test_none_match_config(self):
        ctx = PipelineContext(
            video_duration_sec=100.0,
            match_config=None,
            keeper_track_ids={},
            job_id="j",
        )
        assert ctx.match_config is None


# ===========================================================================
# ReelPlugin ABC
# ===========================================================================

@pytest.mark.unit
class TestReelPluginABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ReelPlugin()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        plugin = _DummyPlugin()
        assert plugin.name == "dummy"
        assert plugin.reel_name == "test_reel"

    def test_default_clip_params(self):
        plugin = _DummyPlugin()
        assert plugin.clip_params == ClipParams()

    def test_default_post_filter_clips_returns_input(self):
        plugin = _DummyPlugin()
        clips = [
            ClipBoundary(
                source_file="a.mp4", start_sec=0, end_sec=5,
                events=["e1"], reel_type="test", primary_event_type="goal",
            )
        ]
        assert plugin.post_filter_clips(clips) is clips

    def test_select_events_passthrough(self):
        plugin = _DummyPlugin()
        ctx = PipelineContext(
            video_duration_sec=100.0, match_config=None,
            keeper_track_ids={}, job_id="j",
        )
        events = []  # no real events needed for passthrough
        assert plugin.select_events(events, ctx) == []
