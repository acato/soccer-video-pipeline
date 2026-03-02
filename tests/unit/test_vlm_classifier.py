"""
Unit tests for VLM (Vision Language Model) classifier.

All tests mock the Anthropic client — no real API calls.
"""
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.detection.models import BoundingBox, Event, EventType
from src.detection.vlm_classifier import VLMClassifier
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
def vlm(match_config):
    return VLMClassifier(
        api_key="test-key-123",
        model="claude-sonnet-4-20250514",
        source_file="/tmp/test-match.mp4",
        match_config=match_config,
        frame_width=640,
        min_confidence=0.6,
    )


def _make_event(
    event_type=EventType.SHOT_STOP_DIVING,
    ts_start=10.0,
    ts_end=11.0,
    confidence=0.85,
    bbox_cx=0.1,
    bbox_cy=0.5,
    metadata=None,
) -> Event:
    return Event(
        event_id="test-event-001",
        job_id="test-job",
        source_file="match.mp4",
        event_type=event_type,
        timestamp_start=ts_start,
        timestamp_end=ts_end,
        confidence=confidence,
        reel_targets=["keeper"],
        is_goalkeeper_event=True,
        frame_start=int(ts_start * 30),
        frame_end=int(ts_end * 30),
        bounding_box=BoundingBox(x=bbox_cx - 0.025, y=bbox_cy - 0.075, width=0.05, height=0.15),
        metadata=metadata or {},
    )


def _make_api_response(is_gk_save: bool, confidence: float, reasoning: str):
    """Build a mock Anthropic message response."""
    body = json.dumps({
        "is_gk_save": is_gk_save,
        "confidence": confidence,
        "reasoning": reasoning,
    })
    content_block = SimpleNamespace(text=body)
    return SimpleNamespace(content=[content_block])


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # minimal JPEG-like bytes


# ---------------------------------------------------------------------------
# Tests: filter_events
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFilterEvents:
    """Tests for the main filter_events entry point."""

    def test_empty_list_returns_empty(self, vlm):
        assert vlm.filter_events([]) == []

    @patch.object(VLMClassifier, "_extract_frames", return_value=[FAKE_JPEG] * 3)
    @patch.object(VLMClassifier, "_classify_event")
    def test_keep_confirmed_save(self, mock_classify, mock_extract, vlm):
        mock_classify.return_value = (True, 0.9, "Clear diving save by teal GK")
        event = _make_event()
        result = vlm.filter_events([event])

        assert len(result) == 1
        assert result[0].metadata["vlm_is_save"] is True
        assert result[0].metadata["vlm_confidence"] == 0.9
        assert result[0].metadata["vlm_reasoning"] == "Clear diving save by teal GK"
        assert result[0].metadata["vlm_model"] == "claude-sonnet-4-20250514"

    @patch.object(VLMClassifier, "_extract_frames", return_value=[FAKE_JPEG] * 3)
    @patch.object(VLMClassifier, "_classify_event")
    def test_reject_false_positive(self, mock_classify, mock_extract, vlm):
        mock_classify.return_value = (False, 0.85, "Blue outfield player, not GK")
        event = _make_event()
        result = vlm.filter_events([event])

        assert len(result) == 0
        assert event.metadata["vlm_is_save"] is False

    @patch.object(VLMClassifier, "_extract_frames", return_value=[FAKE_JPEG] * 3)
    @patch.object(VLMClassifier, "_classify_event")
    def test_reject_low_confidence_save(self, mock_classify, mock_extract, vlm):
        """Even if is_gk_save=True, low confidence means rejection."""
        mock_classify.return_value = (True, 0.3, "Possibly a save but unclear")
        event = _make_event()
        result = vlm.filter_events([event])

        assert len(result) == 0
        assert event.metadata["vlm_confidence"] == 0.3

    @patch.object(VLMClassifier, "_extract_frames", return_value=[FAKE_JPEG] * 3)
    @patch.object(VLMClassifier, "_classify_event")
    def test_mixed_events(self, mock_classify, mock_extract, vlm):
        """Some kept, some rejected."""
        events = [
            _make_event(ts_start=10.0, ts_end=11.0),
            _make_event(ts_start=25.0, ts_end=26.0),
            _make_event(ts_start=40.0, ts_end=41.0),
        ]
        # Assign unique IDs
        events[0].event_id = "ev-1"
        events[1].event_id = "ev-2"
        events[2].event_id = "ev-3"

        mock_classify.side_effect = [
            (True, 0.9, "Real save"),
            (False, 0.8, "Outfield clearance"),
            (True, 0.75, "Standing save"),
        ]
        result = vlm.filter_events(events)

        assert len(result) == 2
        assert result[0].event_id == "ev-1"
        assert result[1].event_id == "ev-3"

    @patch.object(VLMClassifier, "_extract_frames", return_value=[])
    def test_fail_open_on_extraction_failure(self, mock_extract, vlm):
        """Events are kept when frame extraction fails (fail-open)."""
        event = _make_event()
        result = vlm.filter_events([event])

        assert len(result) == 1

    @patch.object(VLMClassifier, "_extract_frames", return_value=[FAKE_JPEG] * 3)
    @patch.object(VLMClassifier, "_classify_event", side_effect=Exception("API timeout"))
    def test_fail_open_on_api_error(self, mock_classify, mock_extract, vlm):
        """Events are kept when API call fails (fail-open)."""
        event = _make_event()
        result = vlm.filter_events([event])

        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: frame extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFrameExtraction:
    """Tests for _extract_frames and _extract_single_frame."""

    @patch("subprocess.run")
    def test_extract_single_frame_success(self, mock_run, vlm):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=FAKE_JPEG, stderr=b""
        )
        result = vlm._extract_single_frame(10.5)
        assert result == FAKE_JPEG

        # Verify FFmpeg command structure
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ffmpeg"
        assert "-ss" in call_args
        assert "10.500" in call_args[call_args.index("-ss") + 1]
        assert "-i" in call_args
        assert "/tmp/test-match.mp4" in call_args
        assert "-vframes" in call_args
        assert "1" in call_args
        assert "scale=640:-1" in call_args[call_args.index("-vf") + 1]

    @patch("subprocess.run")
    def test_extract_single_frame_failure(self, mock_run, vlm):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"error"
        )
        result = vlm._extract_single_frame(10.5)
        assert result is None

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 10))
    def test_extract_single_frame_timeout(self, mock_run, vlm):
        result = vlm._extract_single_frame(10.5)
        assert result is None

    @patch.object(VLMClassifier, "_extract_single_frame")
    def test_extract_frames_three_timestamps(self, mock_single, vlm):
        """Extracts 3 frames at -0.5s, 0.0s, +0.5s around event center."""
        mock_single.return_value = FAKE_JPEG
        event = _make_event(ts_start=10.0, ts_end=11.0)

        frames = vlm._extract_frames(event)
        assert len(frames) == 3
        assert mock_single.call_count == 3

        # Event center is (10.0 + 11.0) / 2 = 10.5
        calls = [c[0][0] for c in mock_single.call_args_list]
        assert abs(calls[0] - 10.0) < 0.01   # 10.5 - 0.5
        assert abs(calls[1] - 10.5) < 0.01   # 10.5 + 0.0
        assert abs(calls[2] - 11.0) < 0.01   # 10.5 + 0.5

    @patch.object(VLMClassifier, "_extract_single_frame")
    def test_extract_frames_clamps_to_zero(self, mock_single, vlm):
        """Timestamps never go negative."""
        mock_single.return_value = FAKE_JPEG
        event = _make_event(ts_start=0.0, ts_end=0.2)

        frames = vlm._extract_frames(event)
        calls = [c[0][0] for c in mock_single.call_args_list]
        assert all(t >= 0.0 for t in calls)

    @patch.object(VLMClassifier, "_extract_single_frame")
    def test_extract_frames_partial_failure(self, mock_single, vlm):
        """Returns only successfully extracted frames."""
        mock_single.side_effect = [FAKE_JPEG, None, FAKE_JPEG]
        event = _make_event()

        frames = vlm._extract_frames(event)
        assert len(frames) == 2


# ---------------------------------------------------------------------------
# Tests: prompt generation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPromptGeneration:
    """Tests for _build_prompt."""

    def test_prompt_contains_team_names(self, vlm):
        event = _make_event()
        prompt = vlm._build_prompt(event)

        assert "Rush" in prompt
        assert "GA 2008" in prompt

    def test_prompt_contains_jersey_colors(self, vlm):
        event = _make_event()
        prompt = vlm._build_prompt(event)

        assert "teal" in prompt
        assert "purple" in prompt
        assert "white" in prompt
        assert "blue" in prompt

    def test_prompt_contains_ball_position(self, vlm):
        event = _make_event(bbox_cx=0.1, bbox_cy=0.5)
        prompt = vlm._build_prompt(event)

        assert "x=0.10" in prompt
        assert "y=0.50" in prompt

    def test_prompt_contains_event_type(self, vlm):
        event = _make_event(event_type=EventType.CATCH)
        prompt = vlm._build_prompt(event)

        assert "catch" in prompt

    def test_prompt_contains_speed_metadata(self, vlm):
        event = _make_event(metadata={"ball_pre_speed": 0.85, "ball_post_speed": 0.12})
        prompt = vlm._build_prompt(event)

        assert "0.85" in prompt
        assert "0.12" in prompt

    def test_prompt_without_bbox(self, vlm):
        event = _make_event()
        event.bounding_box = None
        prompt = vlm._build_prompt(event)

        assert "x=unknown" in prompt
        assert "y=unknown" in prompt

    def test_prompt_asks_for_json(self, vlm):
        event = _make_event()
        prompt = vlm._build_prompt(event)

        assert "is_gk_save" in prompt
        assert "confidence" in prompt
        assert "reasoning" in prompt
        assert "Respond ONLY with JSON" in prompt


# ---------------------------------------------------------------------------
# Tests: response parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResponseParsing:
    """Tests for _parse_response."""

    def test_parse_clean_json(self, vlm):
        resp = _make_api_response(True, 0.9, "Diving save by teal GK")
        is_save, conf, reason = vlm._parse_response(resp)

        assert is_save is True
        assert conf == 0.9
        assert reason == "Diving save by teal GK"

    def test_parse_false_response(self, vlm):
        resp = _make_api_response(False, 0.85, "Blue outfield player")
        is_save, conf, reason = vlm._parse_response(resp)

        assert is_save is False
        assert conf == 0.85

    def test_parse_markdown_fenced_json(self, vlm):
        """Handle JSON wrapped in ```json ... ``` blocks."""
        body = '```json\n{"is_gk_save": true, "confidence": 0.8, "reasoning": "GK save"}\n```'
        content_block = SimpleNamespace(text=body)
        resp = SimpleNamespace(content=[content_block])

        is_save, conf, reason = vlm._parse_response(resp)
        assert is_save is True
        assert conf == 0.8

    def test_parse_clamps_confidence(self, vlm):
        """Confidence values outside 0-1 are clamped."""
        body = json.dumps({"is_gk_save": True, "confidence": 1.5, "reasoning": "test"})
        resp = SimpleNamespace(content=[SimpleNamespace(text=body)])

        _, conf, _ = vlm._parse_response(resp)
        assert conf == 1.0

        body = json.dumps({"is_gk_save": True, "confidence": -0.5, "reasoning": "test"})
        resp = SimpleNamespace(content=[SimpleNamespace(text=body)])

        _, conf, _ = vlm._parse_response(resp)
        assert conf == 0.0

    def test_parse_malformed_json(self, vlm):
        """Fail-open on unparseable response."""
        resp = SimpleNamespace(content=[SimpleNamespace(text="I think this is a save")])
        is_save, conf, reason = vlm._parse_response(resp)

        assert is_save is True  # fail-open
        assert conf == 0.0
        assert "parse_error" in reason

    def test_parse_missing_fields(self, vlm):
        """Missing fields get defaults."""
        body = json.dumps({"is_gk_save": True})
        resp = SimpleNamespace(content=[SimpleNamespace(text=body)])

        is_save, conf, reason = vlm._parse_response(resp)
        assert is_save is True
        assert conf == 0.0
        assert reason == ""


# ---------------------------------------------------------------------------
# Tests: classify_event (integration with mocked client)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyEvent:
    """Tests for _classify_event with mocked Anthropic client."""

    def test_classify_sends_images_and_prompt(self, vlm):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_api_response(
            True, 0.9, "Real save"
        )
        vlm._client = mock_client

        event = _make_event()
        frames = [FAKE_JPEG, FAKE_JPEG, FAKE_JPEG]

        is_save, conf, reason = vlm._classify_event(event, frames)

        assert is_save is True
        assert conf == 0.9

        # Verify API was called with correct structure
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 256

        content = call_kwargs["messages"][0]["content"]
        # 3 images + 1 text prompt
        assert len(content) == 4
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "image"
        assert content[3]["type"] == "text"

        # Images are base64 encoded
        assert content[0]["source"]["type"] == "base64"
        assert content[0]["source"]["media_type"] == "image/jpeg"

    def test_classify_with_single_frame(self, vlm):
        """Works with fewer than 3 frames."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_api_response(
            True, 0.7, "Partial frames"
        )
        vlm._client = mock_client

        event = _make_event()
        frames = [FAKE_JPEG]

        is_save, conf, reason = vlm._classify_event(event, frames)
        assert is_save is True

        content = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        assert len(content) == 2  # 1 image + 1 text


# ---------------------------------------------------------------------------
# Tests: lazy client init
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClientInit:
    """Tests for lazy Anthropic client initialization."""

    def test_client_not_created_on_init(self, vlm):
        assert vlm._client is None

    def test_client_created_on_first_use(self, vlm):
        mock_anthropic = MagicMock()
        mock_client_instance = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client_instance

        import sys
        original = sys.modules.get("anthropic")
        sys.modules["anthropic"] = mock_anthropic
        try:
            client = vlm._get_client()
            mock_anthropic.Anthropic.assert_called_once_with(api_key="test-key-123")
            assert client is mock_client_instance
        finally:
            vlm._client = None  # reset for other tests
            if original is not None:
                sys.modules["anthropic"] = original
            else:
                sys.modules.pop("anthropic", None)

    def test_client_reused_on_subsequent_calls(self, vlm):
        mock_client = MagicMock()
        vlm._client = mock_client

        client1 = vlm._get_client()
        client2 = vlm._get_client()

        assert client1 is client2
        assert client1 is mock_client


# ---------------------------------------------------------------------------
# Tests: VLM metadata stored on events
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMetadataStorage:
    """Verify VLM verdict metadata is stored on events."""

    @patch.object(VLMClassifier, "_extract_frames", return_value=[FAKE_JPEG] * 3)
    @patch.object(VLMClassifier, "_classify_event")
    def test_kept_event_has_metadata(self, mock_classify, mock_extract, vlm):
        mock_classify.return_value = (True, 0.92, "Clear save")
        event = _make_event()
        vlm.filter_events([event])

        assert event.metadata["vlm_is_save"] is True
        assert event.metadata["vlm_confidence"] == 0.92
        assert event.metadata["vlm_reasoning"] == "Clear save"
        assert event.metadata["vlm_model"] == "claude-sonnet-4-20250514"

    @patch.object(VLMClassifier, "_extract_frames", return_value=[FAKE_JPEG] * 3)
    @patch.object(VLMClassifier, "_classify_event")
    def test_rejected_event_has_metadata(self, mock_classify, mock_extract, vlm):
        mock_classify.return_value = (False, 0.88, "Not a save")
        event = _make_event()
        vlm.filter_events([event])

        assert event.metadata["vlm_is_save"] is False
        assert event.metadata["vlm_confidence"] == 0.88
