"""
Unit tests for the composable reels architecture.

Covers:
  - ReelSpec model and presets
  - Job.get_reel_specs() backward compat
  - compute_clips_v2 with per-event-type padding
  - Event type filtering by ReelSpec
  - EVENT_TYPE_CONFIG completeness
"""
from __future__ import annotations

import pytest

from src.detection.models import (
    Event, EventType, EVENT_TYPE_CONFIG, EventTypeConfig,
)
from src.ingestion.models import (
    ReelSpec, REEL_PRESETS, reel_types_to_specs,
    Job, VideoFile,
)
from src.segmentation.clipper import compute_clips_v2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_type: EventType,
    start: float,
    end: float | None = None,
    confidence: float = 0.85,
    is_gk: bool = False,
    sim_team_gk: float = 0.90,
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
        reel_targets=[],
        is_goalkeeper_event=is_gk,
        frame_start=int(start * 30),
        frame_end=int(end * 30),
        metadata={"sim_team_gk": sim_team_gk} if is_gk else {},
    )


def _make_video_file():
    return VideoFile(
        path="/mnt/nas/match.mp4", filename="match.mp4",
        duration_sec=5400.0, fps=30.0, width=3840, height=2160,
        codec="h264", size_bytes=20_000_000_000, sha256="abc123",
    )


# ===========================================================================
# EventTypeConfig
# ===========================================================================

@pytest.mark.unit
class TestEventTypeConfig:
    def test_every_event_type_has_config(self):
        """Every EventType enum member has an entry in EVENT_TYPE_CONFIG."""
        for et in EventType:
            assert et in EVENT_TYPE_CONFIG, f"Missing config for {et.value}"

    def test_config_values_are_frozen_dataclass(self):
        for et, cfg in EVENT_TYPE_CONFIG.items():
            assert isinstance(cfg, EventTypeConfig)

    def test_goalkeeper_events_have_is_gk_true(self):
        gk_types = {
            EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING,
            EventType.PUNCH, EventType.CATCH, EventType.GOAL_KICK,
            EventType.DISTRIBUTION_SHORT, EventType.DISTRIBUTION_LONG,
            EventType.ONE_ON_ONE, EventType.CORNER_KICK, EventType.PENALTY,
        }
        for et in gk_types:
            assert EVENT_TYPE_CONFIG[et].is_gk_event is True, f"{et.value} should be GK"

    def test_highlights_events_have_is_gk_false(self):
        hl_types = {
            EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
            EventType.GOAL, EventType.NEAR_MISS,
            EventType.DRIBBLE_SEQUENCE, EventType.TACKLE,
            EventType.FREE_KICK_SHOT,
        }
        for et in hl_types:
            assert EVENT_TYPE_CONFIG[et].is_gk_event is False, f"{et.value} should not be GK"

    def test_saves_have_correct_padding(self):
        for et in [EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING,
                   EventType.PUNCH, EventType.CATCH]:
            cfg = EVENT_TYPE_CONFIG[et]
            assert cfg.pre_pad_sec == 8.0
            assert cfg.post_pad_sec == 2.0
            assert cfg.max_clip_sec == 25.0

    def test_goal_kick_has_short_padding(self):
        cfg = EVENT_TYPE_CONFIG[EventType.GOAL_KICK]
        assert cfg.pre_pad_sec == 1.0
        assert cfg.post_pad_sec == 2.0
        assert cfg.max_clip_sec == 15.0


# ===========================================================================
# ReelSpec
# ===========================================================================

@pytest.mark.unit
class TestReelSpec:
    def test_basic_creation(self):
        spec = ReelSpec(name="my_reel", event_types=["catch", "punch"])
        assert spec.name == "my_reel"
        assert spec.event_types == ["catch", "punch"]
        assert spec.max_reel_duration_sec == 1200.0

    def test_custom_max_duration(self):
        spec = ReelSpec(name="short", event_types=["goal"], max_reel_duration_sec=300.0)
        assert spec.max_reel_duration_sec == 300.0

    def test_presets_exist(self):
        assert "keeper" in REEL_PRESETS
        assert "highlights" in REEL_PRESETS

    def test_keeper_preset_has_all_gk_types(self):
        keeper = REEL_PRESETS["keeper"]
        assert "shot_stop_diving" in keeper.event_types
        assert "corner_kick" in keeper.event_types
        assert "penalty" in keeper.event_types

    def test_highlights_preset_has_shot_types(self):
        hl = REEL_PRESETS["highlights"]
        assert "shot_on_target" in hl.event_types
        assert "goal" in hl.event_types


# ===========================================================================
# reel_types_to_specs conversion
# ===========================================================================

@pytest.mark.unit
class TestReelTypesConversion:
    def test_keeper_converts_to_preset(self):
        specs = reel_types_to_specs(["keeper"])
        assert len(specs) == 1
        assert specs[0].name == "keeper"
        assert "shot_stop_diving" in specs[0].event_types

    def test_highlights_converts_to_preset(self):
        specs = reel_types_to_specs(["highlights"])
        assert len(specs) == 1
        assert specs[0].name == "highlights"

    def test_both_types_convert(self):
        specs = reel_types_to_specs(["keeper", "highlights"])
        assert len(specs) == 2
        names = {s.name for s in specs}
        assert names == {"keeper", "highlights"}

    def test_unknown_type_gets_empty_spec(self):
        specs = reel_types_to_specs(["unknown"])
        assert len(specs) == 1
        assert specs[0].name == "unknown"
        assert specs[0].event_types == []


# ===========================================================================
# Job.get_reel_specs backward compat
# ===========================================================================

@pytest.mark.unit
class TestJobGetReelSpecs:
    def test_legacy_job_converts_reel_types(self):
        job = Job(
            video_file=_make_video_file(),
            reel_types=["keeper", "highlights"],
        )
        specs = job.get_reel_specs()
        assert len(specs) == 2
        names = {s.name for s in specs}
        assert names == {"keeper", "highlights"}

    def test_new_job_uses_reels_field(self):
        custom = ReelSpec(name="saves", event_types=["shot_stop_diving"])
        job = Job(
            video_file=_make_video_file(),
            reels=[custom],
        )
        specs = job.get_reel_specs()
        assert len(specs) == 1
        assert specs[0].name == "saves"

    def test_reels_takes_priority_over_reel_types(self):
        custom = ReelSpec(name="custom", event_types=["goal"])
        job = Job(
            video_file=_make_video_file(),
            reel_types=["keeper"],
            reels=[custom],
        )
        specs = job.get_reel_specs()
        assert len(specs) == 1
        assert specs[0].name == "custom"


# ===========================================================================
# compute_clips_v2
# ===========================================================================

@pytest.mark.unit
class TestComputeClipsV2:
    def test_single_save_uses_per_type_padding(self):
        event = _make_event(EventType.SHOT_STOP_DIVING, start=30.0, is_gk=True)
        clips = compute_clips_v2([event], video_duration=5400.0, reel_name="keeper")
        assert len(clips) == 1
        # pre_pad=8.0, post_pad=2.0 for saves
        assert clips[0].start_sec == pytest.approx(22.0)  # 30 - 8
        assert clips[0].end_sec == pytest.approx(33.0)     # 31 + 2

    def test_goal_kick_uses_short_padding(self):
        event = _make_event(EventType.GOAL_KICK, start=60.0, is_gk=True)
        clips = compute_clips_v2([event], video_duration=5400.0, reel_name="keeper")
        assert len(clips) == 1
        assert clips[0].start_sec == pytest.approx(59.0)  # 60 - 1
        assert clips[0].end_sec == pytest.approx(63.0)     # 61 + 2

    def test_goal_uses_wide_padding(self):
        event = _make_event(EventType.GOAL, start=100.0)
        clips = compute_clips_v2([event], video_duration=5400.0, reel_name="highlights")
        assert len(clips) == 1
        assert clips[0].start_sec == pytest.approx(95.0)   # 100 - 5
        assert clips[0].end_sec == pytest.approx(109.0)     # 101 + 8

    def test_empty_events_returns_empty(self):
        clips = compute_clips_v2([], video_duration=5400.0)
        assert clips == []

    def test_low_confidence_filtered_out(self):
        event = _make_event(EventType.CATCH, start=30.0, confidence=0.10)
        clips = compute_clips_v2([event], video_duration=5400.0)
        assert clips == []

    def test_two_distant_events_two_clips(self):
        e1 = _make_event(EventType.CATCH, start=30.0, is_gk=True)
        e2 = _make_event(EventType.CATCH, start=300.0, is_gk=True)
        clips = compute_clips_v2([e1, e2], video_duration=5400.0)
        assert len(clips) == 2

    def test_mixed_types_use_respective_padding(self):
        save = _make_event(EventType.SHOT_STOP_DIVING, start=30.0, is_gk=True)
        gk = _make_event(EventType.GOAL_KICK, start=300.0, is_gk=True)
        clips = compute_clips_v2([save, gk], video_duration=5400.0)
        assert len(clips) == 2
        # Save clip
        assert clips[0].start_sec == pytest.approx(22.0)
        # Goal kick clip
        assert clips[1].start_sec == pytest.approx(299.0)

    def test_reel_name_propagated(self):
        event = _make_event(EventType.CATCH, start=30.0, is_gk=True)
        clips = compute_clips_v2([event], video_duration=5400.0, reel_name="deflections")
        assert clips[0].reel_type == "deflections"

    def test_clips_clamped_to_zero(self):
        event = _make_event(EventType.SHOT_STOP_DIVING, start=3.0, is_gk=True)
        clips = compute_clips_v2([event], video_duration=5400.0)
        assert clips[0].start_sec == 0.0  # 3.0 - 8.0 = -5.0, clamped to 0

    def test_clips_clamped_to_duration(self):
        event = _make_event(EventType.GOAL, start=5398.0)
        clips = compute_clips_v2([event], video_duration=5400.0)
        assert clips[0].end_sec == 5400.0  # 5399 + 8 = 5407, clamped to 5400
