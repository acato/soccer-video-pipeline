"""
Unit tests for Gemini 2.5 Flash dead-ball restart classifier.

All tests mock the Google GenAI client — no real API calls.
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.detection.gemini_classifier import (
    ClassificationResult,
    GapCandidate,
    GeminiClassifier,
)
from src.detection.models import BoundingBox, Event, EventType
from src.ingestion.models import KitConfig, MatchConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def match_config():
    return MatchConfig(
        team=KitConfig(team_name="Rush", outfield_color="white", gk_color="teal"),
        opponent=KitConfig(team_name="GA 2008", outfield_color="blue", gk_color="purple"),
    )


@pytest.fixture
def classifier(match_config):
    return GeminiClassifier(
        api_key="test-key-123",
        model="gemini-2.5-flash",
        source_file="/tmp/test-match.mp4",
        match_config=match_config,
        job_id="test-job",
        clip_pre_sec=5.0,
        clip_post_sec=8.0,
        min_confidence=0.5,
    )


def _make_gap(
    start_ts: float = 60.0,
    end_ts: float = 63.0,
    fps: float = 30.0,
    pos_before: tuple = (0.1, 0.5),
    pos_after: tuple = (0.15, 0.6),
) -> GapCandidate:
    return GapCandidate(
        gap_start_frame=int(start_ts * fps),
        gap_end_frame=int(end_ts * fps),
        gap_start_ts=start_ts,
        gap_end_ts=end_ts,
        gap_duration_sec=end_ts - start_ts,
        ball_pos_before=pos_before,
        ball_pos_after=pos_after,
    )


def _make_gemini_response(
    event_type: str = "goal_kick",
    confidence: float = 0.85,
    reasoning: str = "GK places ball on 6-yard box",
    kick_offset: float = 7.0,
) -> SimpleNamespace:
    """Build a mock Gemini response."""
    body = json.dumps({
        "event_type": event_type,
        "confidence": confidence,
        "reasoning": reasoning,
        "kick_timestamp_offset_sec": kick_offset,
    })
    return SimpleNamespace(text=body)


# ---------------------------------------------------------------------------
# Tests: classify_gaps
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyGaps:
    """Tests for the main classify_gaps entry point."""

    def test_empty_gaps_returns_empty(self, classifier):
        assert classifier.classify_gaps([], fps=30.0) == []

    @patch.object(GeminiClassifier, "_extract_clip")
    @patch.object(GeminiClassifier, "_classify_clip")
    def test_classify_goal_kick(self, mock_classify, mock_extract, classifier):
        mock_extract.return_value = Path("/tmp/fake_clip.mp4")
        mock_classify.return_value = ClassificationResult(
            event_type="goal_kick", confidence=0.9,
            reasoning="GK places ball", kick_timestamp_offset_sec=7.0,
        )
        gap = _make_gap()
        events = classifier.classify_gaps([gap], fps=30.0)

        assert len(events) == 1
        assert events[0].event_type == EventType.GOAL_KICK
        assert events[0].metadata["gemini_event_type"] == "goal_kick"
        assert events[0].metadata["gemini_confidence"] == 0.9
        assert events[0].is_goalkeeper_event is True

    @patch.object(GeminiClassifier, "_extract_clip")
    @patch.object(GeminiClassifier, "_classify_clip")
    def test_classify_corner_kick(self, mock_classify, mock_extract, classifier):
        mock_extract.return_value = Path("/tmp/fake_clip.mp4")
        mock_classify.return_value = ClassificationResult(
            event_type="corner_kick", confidence=0.8,
            reasoning="Ball at corner flag", kick_timestamp_offset_sec=6.0,
        )
        gap = _make_gap()
        events = classifier.classify_gaps([gap], fps=30.0)

        assert len(events) == 1
        assert events[0].event_type == EventType.CORNER_KICK

    @patch.object(GeminiClassifier, "_extract_clip")
    @patch.object(GeminiClassifier, "_classify_clip")
    def test_filter_out_throw_in(self, mock_classify, mock_extract, classifier):
        """Throw-ins are not in target_types by default, so filtered out."""
        mock_extract.return_value = Path("/tmp/fake_clip.mp4")
        mock_classify.return_value = ClassificationResult(
            event_type="throw_in", confidence=0.9,
            reasoning="Player throws from sideline", kick_timestamp_offset_sec=5.0,
        )
        gap = _make_gap()
        events = classifier.classify_gaps([gap], fps=30.0)

        assert len(events) == 0

    @patch.object(GeminiClassifier, "_extract_clip")
    @patch.object(GeminiClassifier, "_classify_clip")
    def test_filter_low_confidence(self, mock_classify, mock_extract, classifier):
        """Events below min_confidence are filtered out."""
        mock_extract.return_value = Path("/tmp/fake_clip.mp4")
        mock_classify.return_value = ClassificationResult(
            event_type="goal_kick", confidence=0.3,
            reasoning="Uncertain", kick_timestamp_offset_sec=7.0,
        )
        gap = _make_gap()
        events = classifier.classify_gaps([gap], fps=30.0)

        assert len(events) == 0

    @patch.object(GeminiClassifier, "_extract_clip", return_value=None)
    def test_skip_on_extraction_failure(self, mock_extract, classifier):
        """Gaps with failed clip extraction are skipped (not crashed)."""
        gap = _make_gap()
        events = classifier.classify_gaps([gap], fps=30.0)

        assert len(events) == 0

    @patch.object(GeminiClassifier, "_extract_clip")
    @patch.object(GeminiClassifier, "_classify_clip", return_value=None)
    def test_skip_on_classification_failure(self, mock_classify, mock_extract, classifier):
        mock_extract.return_value = Path("/tmp/fake_clip.mp4")
        gap = _make_gap()
        events = classifier.classify_gaps([gap], fps=30.0)

        assert len(events) == 0

    @patch.object(GeminiClassifier, "_extract_clip")
    @patch.object(GeminiClassifier, "_classify_clip", side_effect=Exception("API timeout"))
    def test_fail_open_on_api_error(self, mock_classify, mock_extract, classifier):
        """Pipeline doesn't crash on API errors — gap is skipped."""
        mock_extract.return_value = Path("/tmp/fake_clip.mp4")
        gap = _make_gap()
        events = classifier.classify_gaps([gap], fps=30.0)

        assert len(events) == 0  # skipped, not crashed

    @patch.object(GeminiClassifier, "_extract_clip")
    @patch.object(GeminiClassifier, "_classify_clip")
    def test_multiple_gaps(self, mock_classify, mock_extract, classifier):
        """Classify multiple gaps in sequence."""
        mock_extract.return_value = Path("/tmp/fake_clip.mp4")
        mock_classify.side_effect = [
            ClassificationResult("goal_kick", 0.9, "GK kick", 7.0),
            ClassificationResult("throw_in", 0.85, "Throw", 5.0),
            ClassificationResult("corner_kick", 0.8, "Corner", 6.0),
        ]
        gaps = [_make_gap(start_ts=t) for t in [60.0, 120.0, 180.0]]
        events = classifier.classify_gaps(gaps, fps=30.0)

        # throw_in filtered out
        assert len(events) == 2
        assert events[0].event_type == EventType.GOAL_KICK
        assert events[1].event_type == EventType.CORNER_KICK


# ---------------------------------------------------------------------------
# Tests: clip extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClipExtraction:
    """Tests for _extract_clip FFmpeg command."""

    @patch("subprocess.run")
    def test_extract_clip_success(self, mock_run, classifier, tmp_path):
        clip_path = tmp_path / "test_clip.mp4"
        clip_path.write_bytes(b"\x00" * 100)

        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"", stderr=b"",
        )

        with patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_file = MagicMock()
            mock_file.name = str(clip_path)
            mock_tmp.return_value = mock_file

            gap = _make_gap(start_ts=60.0, end_ts=63.0)
            result = classifier._extract_clip(gap)

            assert result is not None
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "ffmpeg"
            assert "-ss" in call_args
            # start_ts=60, pre_sec=5 → ss=55.0
            assert "55.000" in call_args[call_args.index("-ss") + 1]
            # duration = 5 + 3 + 8 = 16
            assert "16.000" in call_args[call_args.index("-t") + 1]
            assert "scale=640:-2" in call_args[call_args.index("-vf") + 1]

    @patch("subprocess.run")
    def test_extract_clip_failure(self, mock_run, classifier):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"error",
        )
        gap = _make_gap()
        result = classifier._extract_clip(gap)

        assert result is None

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 30))
    def test_extract_clip_timeout(self, mock_run, classifier):
        gap = _make_gap()
        result = classifier._extract_clip(gap)

        assert result is None


# ---------------------------------------------------------------------------
# Tests: prompt generation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPromptGeneration:
    """Tests for _build_prompt."""

    def test_prompt_contains_team_names(self, classifier):
        gap = _make_gap()
        prompt = classifier._build_prompt(gap)

        assert "Rush" in prompt
        assert "GA 2008" in prompt

    def test_prompt_contains_jersey_colors(self, classifier):
        gap = _make_gap()
        prompt = classifier._build_prompt(gap)

        assert "teal" in prompt
        assert "purple" in prompt
        assert "white" in prompt
        assert "blue" in prompt

    def test_prompt_contains_gap_duration(self, classifier):
        gap = _make_gap(start_ts=60.0, end_ts=63.5)
        prompt = classifier._build_prompt(gap)

        assert "3.5" in prompt

    def test_prompt_contains_ball_positions(self, classifier):
        gap = _make_gap(pos_before=(0.10, 0.50), pos_after=(0.15, 0.60))
        prompt = classifier._build_prompt(gap)

        assert "x=0.10" in prompt
        assert "y=0.50" in prompt
        assert "x=0.15" in prompt
        assert "y=0.60" in prompt

    def test_prompt_handles_unknown_positions(self, classifier):
        gap = _make_gap(pos_before=None, pos_after=None)
        prompt = classifier._build_prompt(gap)

        assert "unknown" in prompt

    def test_prompt_asks_for_json(self, classifier):
        gap = _make_gap()
        prompt = classifier._build_prompt(gap)

        assert "event_type" in prompt
        assert "confidence" in prompt
        assert "reasoning" in prompt
        assert "goal_kick" in prompt
        assert "corner_kick" in prompt

    def test_prompt_contains_visual_cues(self, classifier):
        gap = _make_gap()
        prompt = classifier._build_prompt(gap)

        assert "6-yard box" in prompt
        assert "corner flag" in prompt
        assert "both hands" in prompt  # throw-in


# ---------------------------------------------------------------------------
# Tests: response parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResponseParsing:
    """Tests for _parse_response."""

    def test_parse_clean_json(self, classifier):
        resp = _make_gemini_response("goal_kick", 0.9, "GK kick")
        result = classifier._parse_response(resp)

        assert result is not None
        assert result.event_type == "goal_kick"
        assert result.confidence == 0.9
        assert result.reasoning == "GK kick"

    def test_parse_corner_kick(self, classifier):
        resp = _make_gemini_response("corner_kick", 0.8, "Corner flag")
        result = classifier._parse_response(resp)

        assert result.event_type == "corner_kick"
        assert result.confidence == 0.8

    def test_parse_markdown_fenced_json(self, classifier):
        """Handle JSON wrapped in ```json ... ``` blocks."""
        body = '```json\n{"event_type": "goal_kick", "confidence": 0.85, "reasoning": "GK kick", "kick_timestamp_offset_sec": 7.0}\n```'
        resp = SimpleNamespace(text=body)
        result = classifier._parse_response(resp)

        assert result is not None
        assert result.event_type == "goal_kick"
        assert result.confidence == 0.85

    def test_parse_json_with_surrounding_text(self, classifier):
        """Handle JSON with text before/after."""
        body = 'Here is my analysis:\n{"event_type": "goal_kick", "confidence": 0.8, "reasoning": "GK kick"}\nThat is my answer.'
        resp = SimpleNamespace(text=body)
        result = classifier._parse_response(resp)

        assert result is not None
        assert result.event_type == "goal_kick"

    def test_parse_clamps_confidence(self, classifier):
        resp = _make_gemini_response("goal_kick", 1.5, "test")
        result = classifier._parse_response(resp)
        assert result.confidence == 1.0

        resp = _make_gemini_response("goal_kick", -0.5, "test")
        result = classifier._parse_response(resp)
        assert result.confidence == 0.0

    def test_parse_malformed_json(self, classifier):
        resp = SimpleNamespace(text="I think this is a goal kick")
        result = classifier._parse_response(resp)
        assert result is None

    def test_parse_empty_response(self, classifier):
        resp = SimpleNamespace(text="")
        result = classifier._parse_response(resp)
        assert result is None

    def test_parse_missing_fields(self, classifier):
        """Missing optional fields get defaults."""
        body = json.dumps({"event_type": "goal_kick"})
        resp = SimpleNamespace(text=body)
        result = classifier._parse_response(resp)

        assert result is not None
        assert result.event_type == "goal_kick"
        assert result.confidence == 0.0
        assert result.reasoning == ""
        assert result.kick_timestamp_offset_sec is None


# ---------------------------------------------------------------------------
# Tests: event creation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEventCreation:
    """Tests for _make_event."""

    def test_goal_kick_event(self, classifier):
        gap = _make_gap(start_ts=60.0, end_ts=63.0, pos_after=(0.15, 0.6))
        result = ClassificationResult(
            event_type="goal_kick", confidence=0.9,
            reasoning="GK kick", kick_timestamp_offset_sec=7.0,
        )
        event = classifier._make_event(gap, result, fps=30.0)

        assert event.event_type == EventType.GOAL_KICK
        assert event.confidence == 0.9
        assert event.is_goalkeeper_event is True
        assert event.metadata["gemini_event_type"] == "goal_kick"
        assert event.metadata["gemini_model"] == "gemini-2.5-flash"
        assert event.metadata["gap_duration_sec"] == 3.0
        assert event.bounding_box is not None

    def test_corner_kick_event(self, classifier):
        gap = _make_gap()
        result = ClassificationResult(
            event_type="corner_kick", confidence=0.8,
            reasoning="Corner", kick_timestamp_offset_sec=6.0,
        )
        event = classifier._make_event(gap, result, fps=30.0)

        assert event.event_type == EventType.CORNER_KICK
        assert event.is_goalkeeper_event is True

    def test_goal_event(self, classifier):
        gap = _make_gap()
        result = ClassificationResult(
            event_type="goal", confidence=0.95,
            reasoning="Ball in net", kick_timestamp_offset_sec=5.0,
        )
        event = classifier._make_event(gap, result, fps=30.0)

        assert event.event_type == EventType.GOAL
        assert event.is_goalkeeper_event is False

    def test_event_without_kick_offset(self, classifier):
        """When no kick offset, uses gap center for timestamp."""
        gap = _make_gap(start_ts=60.0, end_ts=66.0)
        result = ClassificationResult(
            event_type="goal_kick", confidence=0.8,
            reasoning="GK kick", kick_timestamp_offset_sec=None,
        )
        event = classifier._make_event(gap, result, fps=30.0)

        # Gap center = (60 + 66) / 2 = 63
        assert abs(event.timestamp_start - 62.5) < 0.01
        assert abs(event.timestamp_end - 63.5) < 0.01

    def test_event_with_kick_offset(self, classifier):
        """Kick offset is relative to clip start (gap_start - pre_sec)."""
        gap = _make_gap(start_ts=60.0, end_ts=63.0)
        result = ClassificationResult(
            event_type="goal_kick", confidence=0.85,
            reasoning="GK kick", kick_timestamp_offset_sec=7.0,
        )
        event = classifier._make_event(gap, result, fps=30.0)

        # Clip start = 60 - 5.0 = 55.0
        # Kick ts = 55.0 + 7.0 = 62.0
        assert abs(event.timestamp_start - 61.5) < 0.01
        assert abs(event.timestamp_end - 62.5) < 0.01

    def test_event_no_ball_pos_after(self, classifier):
        """No bounding box when ball_pos_after is None."""
        gap = _make_gap(pos_after=None)
        result = ClassificationResult(
            event_type="goal_kick", confidence=0.8,
            reasoning="GK kick", kick_timestamp_offset_sec=7.0,
        )
        event = classifier._make_event(gap, result, fps=30.0)

        assert event.bounding_box is None


# ---------------------------------------------------------------------------
# Tests: lazy client init
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClientInit:
    """Tests for lazy Google GenAI client initialization."""

    def test_client_not_created_on_init(self, classifier):
        assert classifier._client is None

    def test_client_reused_on_subsequent_calls(self, classifier):
        mock_client = MagicMock()
        classifier._client = mock_client

        client1 = classifier._get_client()
        client2 = classifier._get_client()

        assert client1 is client2
        assert client1 is mock_client
