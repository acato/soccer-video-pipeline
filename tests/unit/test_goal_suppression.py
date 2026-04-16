"""Tests for save-language goal suppression in _post_filter_events."""
import pytest

from src.detection.dual_pass_detector import (
    DualPassConfig,
    DualPassDetector,
    _has_save_language,
)
from src.detection.models import Event, EventType


def _make_detector(**overrides) -> DualPassDetector:
    cfg = DualPassConfig(**overrides)
    det = DualPassDetector.__new__(DualPassDetector)
    det._cfg = cfg
    return det


def _make_event(
    event_type: str = "goal",
    start: float = 100.0,
    reasoning: str = "",
    confidence: float = 0.9,
) -> Event:
    return Event(
        event_id="test",
        job_id="test",
        source_file="test.mp4",
        event_type=EventType(event_type),
        timestamp_start=start,
        timestamp_end=start + 3.0,
        confidence=confidence,
        reel_targets=[],
        is_goalkeeper_event=False,
        frame_start=int(start * 30),
        frame_end=int((start + 3) * 30),
        reviewed=False,
        review_override=None,
        metadata={"vlm_reasoning": reasoning},
    )


# ── _has_save_language unit tests ──────────────────────────────────────


@pytest.mark.unit
class TestHasSaveLanguage:
    def test_keeper_dives(self):
        assert _has_save_language("Goalkeeper dives to their right to make a save")

    def test_keeper_saves(self):
        assert _has_save_language("The keeper saves the shot from close range")

    def test_ball_deflects_away(self):
        assert _has_save_language("the ball deflects away from the goal")

    def test_ball_rebounds_away(self):
        assert _has_save_language("the ball rebounds away after hitting the post")

    def test_does_not_enter_net(self):
        assert _has_save_language("The ball does not enter the net")

    def test_ball_deflected_away_from_goal(self):
        assert _has_save_language(
            "ball is deflected away from the goal by the keeper"
        )

    def test_gk_blocks(self):
        assert _has_save_language("GK blocks the shot at close range")

    def test_clean_goal_no_save_language(self):
        assert not _has_save_language(
            "The ball enters the goal and players celebrate with arms raised"
        )

    def test_celebration_only(self):
        assert not _has_save_language(
            "Players are celebrating near the center circle after scoring"
        )

    def test_empty_string(self):
        assert not _has_save_language("")

    def test_shot_toward_goal_no_save(self):
        assert not _has_save_language(
            "A player strikes the ball toward the goal from outside the box"
        )


# ── _post_filter_events integration ───────────────────────────────────


@pytest.mark.unit
class TestGoalSaveLanguageSuppression:
    """Goal events with save language in reasoning get demoted to shot_on_target."""

    def test_goal_with_save_reasoning_demoted(self):
        det = _make_detector(single_pass=True)
        events = [
            _make_event(
                "goal",
                start=500.0,
                reasoning="Goalkeeper dives to their right; ball deflects away from the goal",
            ),
        ]
        result = det._post_filter_events(events)
        assert len(result) == 1
        assert result[0].event_type == EventType.SHOT_ON_TARGET
        assert result[0].metadata["goal_suppressed"] is True
        assert result[0].metadata["suppression_reason"] == "save_language"

    def test_goal_with_clean_reasoning_kept(self):
        det = _make_detector(single_pass=True)
        events = [
            _make_event(
                "goal",
                start=800.0,
                reasoning="Ball enters the goal, players celebrate with arms raised",
            ),
        ]
        result = det._post_filter_events(events)
        assert len(result) == 1
        assert result[0].event_type == EventType.GOAL

    def test_goal_no_reasoning_kept(self):
        """Goals without reasoning metadata should not be suppressed."""
        det = _make_detector(single_pass=True)
        event = _make_event("goal", start=100.0, reasoning="")
        result = det._post_filter_events([event])
        assert result[0].event_type == EventType.GOAL

    def test_non_goal_events_unaffected(self):
        det = _make_detector(single_pass=True)
        events = [
            _make_event("shot_on_target", start=100.0,
                        reasoning="Goalkeeper dives to save"),
            _make_event("throw_in", start=200.0, reasoning=""),
            _make_event("goal_kick", start=300.0, reasoning=""),
        ]
        result = det._post_filter_events(events)
        types = [e.event_type for e in result]
        assert types == [
            EventType.SHOT_ON_TARGET,
            EventType.THROW_IN,
            EventType.GOAL_KICK,
        ]

    def test_mixed_goals_only_save_language_demoted(self):
        """Only goals with save language get demoted; clean goals survive."""
        det = _make_detector(single_pass=True)
        events = [
            _make_event(
                "goal", start=800.0,
                reasoning="Players in white celebrating near the goal",
            ),
            _make_event(
                "goal", start=810.0,
                reasoning="Goalkeeper dives to make a save; ball rebounds away",
            ),
        ]
        result = det._post_filter_events(events)
        goal_events = [e for e in result if e.event_type == EventType.GOAL]
        shot_events = [e for e in result
                       if e.event_type == EventType.SHOT_ON_TARGET]
        assert len(goal_events) == 1
        assert goal_events[0].timestamp_start == 800.0
        assert len(shot_events) == 1
        assert shot_events[0].timestamp_start == 810.0

    def test_works_in_dual_pass_mode(self):
        """Save-language gate applies in dual-pass mode too.

        In dual-pass, the keyword gate may fire first for goals without
        positive evidence.  This test uses reasoning that passes the keyword
        gate (has celebration language) but also contains save language.
        """
        det = _make_detector(single_pass=False)
        events = [
            _make_event(
                "goal", start=500.0,
                reasoning=(
                    "Players are celebrating with arms raised, but the "
                    "goalkeeper dives to their right and the ball rebounds away"
                ),
            ),
        ]
        result = det._post_filter_events(events)
        assert result[0].event_type == EventType.SHOT_ON_TARGET
        assert result[0].metadata["goal_suppressed"] is True
