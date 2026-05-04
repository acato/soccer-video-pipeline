"""Unit tests for src/detection/temporal_fusion.py."""
from __future__ import annotations

import pytest

from src.detection.models import Event, EventType
from src.detection.temporal_fusion import apply_temporal_fusion


def _ev(etype: EventType, t_start: float, t_end: float | None = None,
        confidence: float = 0.7, reel_targets: list[str] | None = None) -> Event:
    return Event(
        job_id="j1",
        source_file="m.mp4",
        event_type=etype,
        timestamp_start=t_start,
        timestamp_end=t_end if t_end is not None else t_start + 1.0,
        confidence=confidence,
        reel_targets=reel_targets if reel_targets is not None else ["highlights"],
        frame_start=int(t_start * 30),
        frame_end=int((t_end if t_end is not None else t_start + 1.0) * 30),
    )


@pytest.mark.unit
class TestApplyTemporalFusion:

    def test_promotes_shot_followed_by_dead_time(self):
        # Shot at t=100; nothing happens for 45s after — celebration pattern.
        events = [_ev(EventType.SHOT_ON_TARGET, 100.0, 101.0)]
        out, stats = apply_temporal_fusion(
            events, min_dead_time_sec=25.0, lookahead_sec=45.0,
        )
        assert stats["shots_promoted"] == 1
        promoted = [e for e in out if e.event_type == EventType.GOAL]
        assert len(promoted) == 1
        assert promoted[0].timestamp_start == 100.0
        assert promoted[0].metadata["promoted_from_shot"] is True
        assert promoted[0].metadata["promotion_method"] == "temporal_dead_time"

    def test_blocks_promotion_when_catch_follows(self):
        # Shot at 100, GK catches at 105 — saved, NOT a goal.
        events = [
            _ev(EventType.SHOT_ON_TARGET, 100.0, 101.0),
            _ev(EventType.CATCH, 105.0, 106.0, reel_targets=["keeper"]),
        ]
        out, stats = apply_temporal_fusion(events)
        assert stats["shots_promoted"] == 0
        assert stats["promotion_blocked_activity"] == 1
        assert sum(1 for e in out if e.event_type == EventType.GOAL) == 0

    def test_blocks_promotion_when_goal_kick_follows(self):
        # Shot wide, ball out, opposing GK takes goal kick within 15s.
        events = [
            _ev(EventType.SHOT_ON_TARGET, 100.0, 101.0),
            _ev(EventType.GOAL_KICK, 115.0, 116.0, reel_targets=["keeper"]),
        ]
        _, stats = apply_temporal_fusion(events)
        assert stats["shots_promoted"] == 0
        assert stats["promotion_blocked_activity"] == 1

    def test_skips_when_existing_goal_covers_shot(self):
        # Already a goal at the shot timestamp — don't double-promote.
        events = [
            _ev(EventType.SHOT_ON_TARGET, 100.0, 101.0),
            _ev(EventType.GOAL, 100.0, 115.0),
        ]
        _, stats = apply_temporal_fusion(events)
        assert stats["shots_promoted"] == 0
        assert stats["promotion_blocked_existing_goal"] == 1

    def test_dead_time_threshold_respected(self):
        # Shot followed by a CORNER_KICK at +20s; min_dead_time=25 → not enough
        # gap so we DON'T promote (an event lives in the window).
        events = [
            _ev(EventType.SHOT_ON_TARGET, 100.0, 101.0),
            _ev(EventType.CORNER_KICK, 120.0, 121.0, reel_targets=["keeper"]),
        ]
        _, stats = apply_temporal_fusion(events, min_dead_time_sec=25.0)
        assert stats["shots_promoted"] == 0

    def test_lookahead_window_caps_search(self):
        # Shot at 100, next event at +60s. lookahead=45 → window only scans up
        # to 145, which is empty → promote. (The +60s event is OUTSIDE the
        # window.)
        events = [
            _ev(EventType.SHOT_ON_TARGET, 100.0, 101.0),
            _ev(EventType.THROW_IN, 160.0, 161.0, reel_targets=["keeper"]),
        ]
        _, stats = apply_temporal_fusion(
            events, min_dead_time_sec=25.0, lookahead_sec=45.0,
        )
        assert stats["shots_promoted"] == 1

    def test_multiple_shots_some_promote_some_dont(self):
        events = [
            _ev(EventType.SHOT_ON_TARGET, 100.0, 101.0),                            # promotes
            _ev(EventType.SHOT_ON_TARGET, 200.0, 201.0),                            # blocked by catch
            _ev(EventType.CATCH, 205.0, 206.0, reel_targets=["keeper"]),
            _ev(EventType.SHOT_ON_TARGET, 300.0, 301.0),                            # promotes
        ]
        out, stats = apply_temporal_fusion(events)
        assert stats["shots_examined"] == 3
        assert stats["shots_promoted"] == 2
        assert stats["promotion_blocked_activity"] == 1

    def test_promoted_goal_inherits_reel_targets(self):
        events = [_ev(EventType.SHOT_ON_TARGET, 100.0, 101.0,
                      reel_targets=["highlights", "keeper"])]
        out, _ = apply_temporal_fusion(events)
        promoted = [e for e in out if e.event_type == EventType.GOAL]
        assert promoted[0].reel_targets == ["highlights", "keeper"]

    def test_returns_originals_plus_new_goals(self):
        events = [_ev(EventType.SHOT_ON_TARGET, 100.0, 101.0)]
        out, _ = apply_temporal_fusion(events)
        # Original shot is preserved AND goal is appended
        assert len(out) == 2
        assert any(e.event_type == EventType.SHOT_ON_TARGET for e in out)
        assert any(e.event_type == EventType.GOAL for e in out)
