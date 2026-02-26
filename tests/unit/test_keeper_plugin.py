"""
Unit tests for keeper reel plugins (KeeperSavesPlugin, KeeperDistributionPlugin).

Tests are pure event filtering — no video, FFmpeg, or ML models needed.
Each test creates synthetic events and verifies plugin selection behavior.
"""
from __future__ import annotations

import pytest

from src.detection.models import Event, EventType
from src.reel_plugins.base import PipelineContext
from src.reel_plugins.keeper import KeeperSavesPlugin, KeeperDistributionPlugin
from tests.conftest import make_match_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_event(
    event_type: EventType,
    start: float = 10.0,
    end: float = 11.0,
    reel_targets: list[str] | None = None,
    confidence: float = 0.85,
    is_gk: bool = True,
) -> Event:
    return Event(
        event_id=f"evt-{event_type.value}-{start}",
        job_id="test-job",
        source_file="match.mp4",
        event_type=event_type,
        timestamp_start=start,
        timestamp_end=end,
        confidence=confidence,
        reel_targets=reel_targets or ["keeper"],
        is_goalkeeper_event=is_gk,
        frame_start=int(start * 30),
        frame_end=int(end * 30),
    )


@pytest.fixture
def ctx():
    return PipelineContext(
        video_duration_sec=5400.0,
        match_config=make_match_config(),
        keeper_track_ids={"keeper_a": 1, "keeper_b": None},
        job_id="test-job",
    )


@pytest.fixture
def mixed_events():
    """Realistic mix of GK saves, distributions, and non-GK events."""
    return [
        _make_event(EventType.SHOT_STOP_DIVING, start=10.0),
        _make_event(EventType.SHOT_STOP_STANDING, start=30.0),
        _make_event(EventType.PUNCH, start=50.0),
        _make_event(EventType.CATCH, start=70.0),
        _make_event(EventType.ONE_ON_ONE, start=90.0),
        _make_event(EventType.GOAL_KICK, start=120.0),
        _make_event(EventType.DISTRIBUTION_SHORT, start=150.0),
        _make_event(EventType.DISTRIBUTION_LONG, start=180.0),
        _make_event(EventType.SHOT_ON_TARGET, start=200.0, reel_targets=["highlights"], is_gk=False),
        _make_event(EventType.GOAL, start=220.0, reel_targets=["highlights"], is_gk=False),
    ]


# ===========================================================================
# KeeperSavesPlugin
# ===========================================================================

@pytest.mark.unit
class TestKeeperSavesPlugin:
    def test_name_and_reel(self):
        p = KeeperSavesPlugin()
        assert p.name == "keeper_saves"
        assert p.reel_name == "keeper"

    def test_clip_params_tight_padding(self):
        p = KeeperSavesPlugin()
        assert p.clip_params.pre_pad_sec == 2.0
        assert p.clip_params.post_pad_sec == 1.5
        assert p.clip_params.max_clip_duration_sec == 15.0

    def test_selects_save_types(self, mixed_events, ctx):
        p = KeeperSavesPlugin()
        selected = p.select_events(mixed_events, ctx)
        types = {e.event_type for e in selected}
        assert types == {
            EventType.SHOT_STOP_DIVING,
            EventType.SHOT_STOP_STANDING,
            EventType.PUNCH,
            EventType.CATCH,
            EventType.ONE_ON_ONE,
        }

    def test_ignores_distribution_types(self, mixed_events, ctx):
        p = KeeperSavesPlugin()
        selected = p.select_events(mixed_events, ctx)
        dist_types = {EventType.GOAL_KICK, EventType.DISTRIBUTION_SHORT, EventType.DISTRIBUTION_LONG}
        assert not any(e.event_type in dist_types for e in selected)

    def test_ignores_highlights_events(self, mixed_events, ctx):
        p = KeeperSavesPlugin()
        selected = p.select_events(mixed_events, ctx)
        assert not any(e.event_type == EventType.SHOT_ON_TARGET for e in selected)
        assert not any(e.event_type == EventType.GOAL for e in selected)

    def test_requires_is_goalkeeper_event(self, ctx):
        event = _make_event(EventType.CATCH, is_gk=False)
        selected = KeeperSavesPlugin().select_events([event], ctx)
        assert selected == []

    def test_requires_keeper_reel_target(self, ctx):
        event = _make_event(EventType.CATCH, reel_targets=["highlights"])
        selected = KeeperSavesPlugin().select_events([event], ctx)
        assert selected == []

    def test_accepts_keeper_a_target(self, ctx):
        event = _make_event(EventType.CATCH, reel_targets=["keeper_a"])
        selected = KeeperSavesPlugin().select_events([event], ctx)
        assert len(selected) == 1

    def test_filters_low_confidence(self, ctx):
        event = _make_event(EventType.CATCH, confidence=0.10)
        selected = KeeperSavesPlugin().select_events([event], ctx)
        assert selected == []

    def test_empty_events(self, ctx):
        assert KeeperSavesPlugin().select_events([], ctx) == []

    def test_all_non_gk_events_returns_empty(self, ctx):
        events = [
            _make_event(EventType.SHOT_ON_TARGET, reel_targets=["highlights"], is_gk=False),
            _make_event(EventType.GOAL, reel_targets=["highlights"], is_gk=False),
        ]
        assert KeeperSavesPlugin().select_events(events, ctx) == []


# ===========================================================================
# KeeperDistributionPlugin
# ===========================================================================

@pytest.mark.unit
class TestKeeperDistributionPlugin:
    def test_name_and_reel(self):
        p = KeeperDistributionPlugin()
        assert p.name == "keeper_distribution"
        assert p.reel_name == "keeper"

    def test_clip_params_distribution_padding(self):
        p = KeeperDistributionPlugin()
        assert p.clip_params.pre_pad_sec == 1.0
        assert p.clip_params.post_pad_sec == 3.0
        assert p.clip_params.max_clip_duration_sec == 10.0

    def test_selects_distribution_types(self, mixed_events, ctx):
        p = KeeperDistributionPlugin()
        selected = p.select_events(mixed_events, ctx)
        types = {e.event_type for e in selected}
        assert types == {
            EventType.GOAL_KICK,
            EventType.DISTRIBUTION_SHORT,
            EventType.DISTRIBUTION_LONG,
        }

    def test_ignores_save_types(self, mixed_events, ctx):
        p = KeeperDistributionPlugin()
        selected = p.select_events(mixed_events, ctx)
        save_types = {EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING, EventType.CATCH}
        assert not any(e.event_type in save_types for e in selected)

    def test_requires_is_goalkeeper_event(self, ctx):
        event = _make_event(EventType.GOAL_KICK, is_gk=False)
        selected = KeeperDistributionPlugin().select_events([event], ctx)
        assert selected == []

    def test_requires_keeper_reel_target(self, ctx):
        event = _make_event(EventType.GOAL_KICK, reel_targets=["highlights"])
        selected = KeeperDistributionPlugin().select_events([event], ctx)
        assert selected == []

    def test_empty_events(self, ctx):
        assert KeeperDistributionPlugin().select_events([], ctx) == []


# ===========================================================================
# Both plugins combined — no overlap
# ===========================================================================

@pytest.mark.unit
class TestKeeperPluginsSeparation:
    def test_saves_and_distribution_are_disjoint(self, mixed_events, ctx):
        """The two plugins never select the same event."""
        saves = KeeperSavesPlugin().select_events(mixed_events, ctx)
        dists = KeeperDistributionPlugin().select_events(mixed_events, ctx)
        save_ids = {e.event_id for e in saves}
        dist_ids = {e.event_id for e in dists}
        assert save_ids.isdisjoint(dist_ids)

    def test_together_cover_all_keeper_events(self, mixed_events, ctx):
        """Between them, saves + distribution cover all GK events."""
        saves = KeeperSavesPlugin().select_events(mixed_events, ctx)
        dists = KeeperDistributionPlugin().select_events(mixed_events, ctx)
        keeper_events = [e for e in mixed_events if e.is_goalkeeper_event]
        assert len(saves) + len(dists) == len(keeper_events)
