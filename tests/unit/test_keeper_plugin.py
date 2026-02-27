"""
Unit tests for keeper reel plugins.

Covers all four keeper plugins (Saves, GoalKick, Distribution, OneOnOne)
and the majority-vote spatial false-positive filter.
"""
from __future__ import annotations

import pytest

from src.detection.models import BoundingBox, Event, EventType
from src.reel_plugins.base import PipelineContext
from src.reel_plugins.keeper import (
    KeeperDistributionPlugin,
    KeeperGoalKickPlugin,
    KeeperOneOnOnePlugin,
    KeeperSavesPlugin,
    _filter_wrong_side_events,
)
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
    bbox_center_x: float | None = None,
) -> Event:
    bbox = None
    if bbox_center_x is not None:
        bbox = BoundingBox(x=bbox_center_x - 0.025, y=0.4, width=0.05, height=0.1)
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
        bounding_box=bbox,
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
        _make_event(EventType.PENALTY, start=200.0),
        _make_event(EventType.SHOT_ON_TARGET, start=220.0, reel_targets=["highlights"], is_gk=False),
        _make_event(EventType.GOAL, start=240.0, reel_targets=["highlights"], is_gk=False),
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
        assert p.clip_params.pre_pad_sec == 8.0
        assert p.clip_params.post_pad_sec == 4.0
        assert p.clip_params.max_clip_duration_sec == 25.0

    def test_selects_save_types(self, mixed_events, ctx):
        p = KeeperSavesPlugin()
        selected = p.select_events(mixed_events, ctx)
        types = {e.event_type for e in selected}
        assert types == {
            EventType.SHOT_STOP_DIVING,
            EventType.SHOT_STOP_STANDING,
            EventType.PUNCH,
            EventType.CATCH,
            EventType.PENALTY,
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

    def test_does_not_select_one_on_one(self, mixed_events, ctx):
        p = KeeperSavesPlugin()
        selected = p.select_events(mixed_events, ctx)
        assert not any(e.event_type == EventType.ONE_ON_ONE for e in selected)

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
# KeeperGoalKickPlugin
# ===========================================================================

@pytest.mark.unit
class TestKeeperGoalKickPlugin:
    def test_name_and_reel(self):
        p = KeeperGoalKickPlugin()
        assert p.name == "keeper_goal_kick"
        assert p.reel_name == "keeper"

    def test_clip_params(self):
        p = KeeperGoalKickPlugin()
        assert p.clip_params.pre_pad_sec == 1.0
        assert p.clip_params.post_pad_sec == 10.0
        assert p.clip_params.max_clip_duration_sec == 20.0

    def test_selects_only_goal_kick(self, mixed_events, ctx):
        p = KeeperGoalKickPlugin()
        selected = p.select_events(mixed_events, ctx)
        types = {e.event_type for e in selected}
        assert types == {EventType.GOAL_KICK}

    def test_requires_is_goalkeeper_event(self, ctx):
        event = _make_event(EventType.GOAL_KICK, is_gk=False)
        assert KeeperGoalKickPlugin().select_events([event], ctx) == []

    def test_requires_keeper_reel_target(self, ctx):
        event = _make_event(EventType.GOAL_KICK, reel_targets=["highlights"])
        assert KeeperGoalKickPlugin().select_events([event], ctx) == []

    def test_empty_events(self, ctx):
        assert KeeperGoalKickPlugin().select_events([], ctx) == []


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
        assert p.clip_params.post_pad_sec == 8.0
        assert p.clip_params.max_clip_duration_sec == 20.0

    def test_selects_distribution_types(self, mixed_events, ctx):
        p = KeeperDistributionPlugin()
        selected = p.select_events(mixed_events, ctx)
        types = {e.event_type for e in selected}
        assert types == {
            EventType.DISTRIBUTION_SHORT,
            EventType.DISTRIBUTION_LONG,
        }

    def test_does_not_select_goal_kick(self, mixed_events, ctx):
        p = KeeperDistributionPlugin()
        selected = p.select_events(mixed_events, ctx)
        assert not any(e.event_type == EventType.GOAL_KICK for e in selected)

    def test_ignores_save_types(self, mixed_events, ctx):
        p = KeeperDistributionPlugin()
        selected = p.select_events(mixed_events, ctx)
        save_types = {EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING, EventType.CATCH}
        assert not any(e.event_type in save_types for e in selected)

    def test_requires_is_goalkeeper_event(self, ctx):
        event = _make_event(EventType.DISTRIBUTION_SHORT, is_gk=False)
        selected = KeeperDistributionPlugin().select_events([event], ctx)
        assert selected == []

    def test_requires_keeper_reel_target(self, ctx):
        event = _make_event(EventType.DISTRIBUTION_SHORT, reel_targets=["highlights"])
        selected = KeeperDistributionPlugin().select_events([event], ctx)
        assert selected == []

    def test_empty_events(self, ctx):
        assert KeeperDistributionPlugin().select_events([], ctx) == []


# ===========================================================================
# KeeperOneOnOnePlugin
# ===========================================================================

@pytest.mark.unit
class TestKeeperOneOnOnePlugin:
    def test_name_and_reel(self):
        p = KeeperOneOnOnePlugin()
        assert p.name == "keeper_one_on_one"
        assert p.reel_name == "keeper"

    def test_clip_params(self):
        p = KeeperOneOnOnePlugin()
        assert p.clip_params.pre_pad_sec == 3.0
        assert p.clip_params.post_pad_sec == 6.0
        assert p.clip_params.max_clip_duration_sec == 30.0

    def test_selects_only_one_on_one(self, mixed_events, ctx):
        p = KeeperOneOnOnePlugin()
        selected = p.select_events(mixed_events, ctx)
        types = {e.event_type for e in selected}
        assert types == {EventType.ONE_ON_ONE}

    def test_requires_is_goalkeeper_event(self, ctx):
        event = _make_event(EventType.ONE_ON_ONE, is_gk=False)
        assert KeeperOneOnOnePlugin().select_events([event], ctx) == []

    def test_requires_keeper_reel_target(self, ctx):
        event = _make_event(EventType.ONE_ON_ONE, reel_targets=["highlights"])
        assert KeeperOneOnOnePlugin().select_events([event], ctx) == []

    def test_empty_events(self, ctx):
        assert KeeperOneOnOnePlugin().select_events([], ctx) == []


# ===========================================================================
# All plugins combined — no overlap, full coverage
# ===========================================================================

@pytest.mark.unit
class TestKeeperPluginsSeparation:
    def test_all_plugins_are_disjoint(self, mixed_events, ctx):
        """No two keeper plugins select the same event."""
        plugins = [
            KeeperSavesPlugin(),
            KeeperGoalKickPlugin(),
            KeeperDistributionPlugin(),
            KeeperOneOnOnePlugin(),
        ]
        all_ids: list[str] = []
        for p in plugins:
            selected = p.select_events(mixed_events, ctx)
            all_ids.extend(e.event_id for e in selected)
        assert len(all_ids) == len(set(all_ids)), "plugins selected overlapping events"

    def test_together_cover_all_keeper_events(self, mixed_events, ctx):
        """Between all four plugins, every GK event is covered."""
        plugins = [
            KeeperSavesPlugin(),
            KeeperGoalKickPlugin(),
            KeeperDistributionPlugin(),
            KeeperOneOnOnePlugin(),
        ]
        total = sum(len(p.select_events(mixed_events, ctx)) for p in plugins)
        keeper_events = [e for e in mixed_events if e.is_goalkeeper_event]
        assert total == len(keeper_events)


# ===========================================================================
# Spatial false-positive filter
# ===========================================================================

@pytest.mark.unit
class TestSpatialFilter:
    """Tests for _filter_wrong_side_events majority-vote spatial filter."""

    def _make_keeper_events_on_side(
        self,
        side: str,
        count: int,
        start_offset: float = 0.0,
    ) -> list[Event]:
        """Create keeper events clustered on one side of the pitch."""
        cx = 0.2 if side == "left" else 0.8
        return [
            _make_event(
                EventType.CATCH,
                start=start_offset + i * 10.0,
                end=start_offset + i * 10.0 + 1.0,
                bbox_center_x=cx,
            )
            for i in range(count)
        ]

    def test_correct_side_events_kept(self):
        """Events on the majority side pass through."""
        # 5 events on left side (first half of a 1000s video)
        all_events = self._make_keeper_events_on_side("left", 5, start_offset=10.0)
        selected = all_events[:3]
        result = _filter_wrong_side_events(selected, all_events, 1000.0)
        assert len(result) == 3

    def test_wrong_side_events_removed(self):
        """Events on the minority side are filtered out."""
        # 5 events on left side + 1 event on right side (all first half)
        left_events = self._make_keeper_events_on_side("left", 5, start_offset=10.0)
        wrong_event = _make_event(
            EventType.CATCH, start=100.0, end=101.0, bbox_center_x=0.8,
        )
        all_events = left_events + [wrong_event]
        selected = [wrong_event]
        result = _filter_wrong_side_events(selected, all_events, 1000.0)
        assert result == []

    def test_halftime_sides_independent(self):
        """First half left + second half right — both correctly filtered."""
        video_dur = 1000.0
        # First half: keeper on left (0-500s)
        first_half = self._make_keeper_events_on_side("left", 5, start_offset=10.0)
        # Second half: keeper on right (500-1000s)
        second_half = self._make_keeper_events_on_side("right", 5, start_offset=600.0)

        all_events = first_half + second_half

        # A wrong-side event in first half (right side)
        wrong_first = _make_event(
            EventType.CATCH, start=100.0, end=101.0, bbox_center_x=0.8,
        )
        # A wrong-side event in second half (left side)
        wrong_second = _make_event(
            EventType.CATCH, start=700.0, end=701.0, bbox_center_x=0.2,
        )
        all_events_with_wrong = all_events + [wrong_first, wrong_second]

        result = _filter_wrong_side_events(
            [wrong_first, wrong_second], all_events_with_wrong, video_dur,
        )
        assert result == []

    def test_ambiguous_split_keeps_all(self):
        """When votes are 50/50, no events are removed."""
        # 3 left + 3 right in first half — neither reaches 60%
        left = self._make_keeper_events_on_side("left", 3, start_offset=10.0)
        right = self._make_keeper_events_on_side("right", 3, start_offset=100.0)
        all_events = left + right
        result = _filter_wrong_side_events(all_events, all_events, 1000.0)
        assert len(result) == 6

    def test_no_bbox_events_always_kept(self):
        """Events without bounding_box are never filtered."""
        # Set up a majority on the left side
        voters = self._make_keeper_events_on_side("left", 5, start_offset=10.0)
        # Event without bbox at "wrong" position doesn't matter — kept
        no_bbox_event = _make_event(EventType.CATCH, start=100.0, end=101.0)
        assert no_bbox_event.bounding_box is None

        all_events = voters + [no_bbox_event]
        result = _filter_wrong_side_events([no_bbox_event], all_events, 1000.0)
        assert len(result) == 1

    def test_single_event_no_filtering(self):
        """With only 1 voter event, no side can be established (all kept)."""
        left = self._make_keeper_events_on_side("left", 1, start_offset=10.0)
        all_events = left
        result = _filter_wrong_side_events(left, all_events, 1000.0)
        assert len(result) == 1

    def test_midfield_events_always_rejected(self):
        """Events in the middle band (0.35-0.65) are rejected outright."""
        # Set up a majority on the left side (to satisfy voter requirements)
        voters = self._make_keeper_events_on_side("left", 5, start_offset=10.0)
        midfield_event = _make_event(
            EventType.CATCH, start=100.0, end=101.0, bbox_center_x=0.50,
        )
        all_events = voters + [midfield_event]
        result = _filter_wrong_side_events([midfield_event], all_events, 1000.0)
        assert result == []

    def test_midfield_events_not_used_for_voting(self):
        """Events in the middle band are excluded from the majority vote."""
        # 3 events at midfield (should be ignored for voting) + 2 on left
        midfield = [
            _make_event(EventType.CATCH, start=10.0 + i * 10, end=11.0 + i * 10, bbox_center_x=0.50)
            for i in range(3)
        ]
        left = self._make_keeper_events_on_side("left", 2, start_offset=100.0)
        right_outlier = _make_event(
            EventType.CATCH, start=200.0, end=201.0, bbox_center_x=0.8,
        )
        all_events = midfield + left + [right_outlier]
        # 2 voters on left, 1 on right → left majority → right outlier removed
        result = _filter_wrong_side_events([right_outlier], all_events, 1000.0)
        assert result == []

    def test_filter_integrates_with_plugin(self, ctx):
        """Spatial filter works end-to-end through a plugin's select_events."""
        # 5 saves on left side, 1 save on right side (opponent GK false positive)
        left_saves = [
            _make_event(
                EventType.CATCH,
                start=10.0 + i * 20,
                bbox_center_x=0.2,
            )
            for i in range(5)
        ]
        wrong_save = _make_event(
            EventType.CATCH, start=200.0, bbox_center_x=0.8,
        )
        events = left_saves + [wrong_save]

        selected = KeeperSavesPlugin().select_events(events, ctx)
        assert len(selected) == 5
        assert wrong_save not in selected
