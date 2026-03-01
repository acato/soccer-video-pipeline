"""
Unit tests for HighlightsShotsPlugin.

Tests are pure event filtering — no video, FFmpeg, or ML models needed.
"""
from __future__ import annotations

import pytest

from src.detection.models import Event, EventType
from src.reel_plugins.base import PipelineContext
from src.reel_plugins.highlights import HighlightsShotsPlugin
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
    is_gk: bool = False,
) -> Event:
    return Event(
        event_id=f"evt-{event_type.value}-{start}",
        job_id="test-job",
        source_file="match.mp4",
        event_type=event_type,
        timestamp_start=start,
        timestamp_end=end,
        confidence=confidence,
        reel_targets=reel_targets or ["highlights"],
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
    """Mix of highlights and non-highlights events."""
    return [
        _make_event(EventType.SHOT_ON_TARGET, start=10.0),
        _make_event(EventType.SHOT_OFF_TARGET, start=30.0),
        _make_event(EventType.GOAL, start=50.0, confidence=0.92),
        _make_event(EventType.NEAR_MISS, start=70.0),
        _make_event(EventType.PENALTY, start=90.0),
        _make_event(EventType.FREE_KICK_SHOT, start=110.0),
        # Non-highlights
        _make_event(EventType.DRIBBLE_SEQUENCE, start=130.0, reel_targets=[]),
        _make_event(EventType.TACKLE, start=150.0, reel_targets=[]),
        _make_event(EventType.CATCH, start=170.0, reel_targets=["keeper"], is_gk=True),
    ]


# ===========================================================================
# HighlightsShotsPlugin
# ===========================================================================

@pytest.mark.unit
class TestHighlightsShotsPlugin:
    def test_name_and_reel(self):
        p = HighlightsShotsPlugin()
        assert p.name == "highlights_shots"
        assert p.reel_name == "highlights"

    def test_clip_params(self):
        p = HighlightsShotsPlugin()
        assert p.clip_params.pre_pad_sec == 3.0
        assert p.clip_params.post_pad_sec == 5.0
        assert p.clip_params.max_clip_duration_sec == 90.0

    def test_selects_shot_and_goal_types(self, mixed_events, ctx):
        p = HighlightsShotsPlugin()
        selected = p.select_events(mixed_events, ctx)
        types = {e.event_type for e in selected}
        assert types == {
            EventType.SHOT_ON_TARGET,
            EventType.SHOT_OFF_TARGET,
            EventType.GOAL,
            EventType.NEAR_MISS,
            EventType.PENALTY,
            EventType.FREE_KICK_SHOT,
        }

    def test_ignores_dribble_and_tackle(self, mixed_events, ctx):
        p = HighlightsShotsPlugin()
        selected = p.select_events(mixed_events, ctx)
        excluded = {EventType.DRIBBLE_SEQUENCE, EventType.TACKLE}
        assert not any(e.event_type in excluded for e in selected)

    def test_ignores_keeper_events(self, mixed_events, ctx):
        p = HighlightsShotsPlugin()
        selected = p.select_events(mixed_events, ctx)
        assert not any(e.event_type == EventType.CATCH for e in selected)

    def test_filters_low_confidence_goal(self, ctx):
        """GOAL has confidence threshold 0.85 — low confidence should be filtered."""
        event = _make_event(EventType.GOAL, confidence=0.50)
        selected = HighlightsShotsPlugin().select_events([event], ctx)
        assert selected == []

    def test_empty_events(self, ctx):
        assert HighlightsShotsPlugin().select_events([], ctx) == []

    def test_count_matches_expected(self, mixed_events, ctx):
        p = HighlightsShotsPlugin()
        selected = p.select_events(mixed_events, ctx)
        assert len(selected) == 6
