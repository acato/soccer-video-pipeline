"""
Unit tests for src/detection/scene_analyzer.py

All tests mock the Anthropic client — no real API calls.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.detection.frame_sampler import SampledFrame
from src.detection.models import EventBoundary, GameState, SceneLabel
from src.detection.scene_analyzer import COARSE_BATCH_SIZE, SceneAnalyzer


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


@pytest.fixture
def analyzer():
    return SceneAnalyzer(
        api_key="test-key-123",
        model="claude-sonnet-4-20250514",
        source_file="/tmp/test-match.mp4",
        event_types=["corner_kick", "goal_kick"],
    )


def _make_frames(timestamps: list[float]) -> list[SampledFrame]:
    return [SampledFrame(timestamp_sec=ts, jpeg_bytes=FAKE_JPEG) for ts in timestamps]


def _make_scan_response(items: list[dict]):
    """Build a mock Anthropic message response with JSON array."""
    body = json.dumps(items)
    return SimpleNamespace(content=[SimpleNamespace(text=body)])


def _make_refine_response(confirmed: bool, start: float, end: float, reasoning: str = ""):
    body = json.dumps({
        "confirmed": confirmed,
        "clip_start_sec": start,
        "clip_end_sec": end,
        "reasoning": reasoning,
    })
    return SimpleNamespace(content=[SimpleNamespace(text=body)])


# ---------------------------------------------------------------------------
# Tests: scan()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScan:

    def test_scan_classifies_frames(self, analyzer):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_scan_response([
            {"frame_index": 0, "timestamp": 0.0, "state": "active_play"},
            {"frame_index": 1, "timestamp": 3.0, "state": "corner_kick"},
            {"frame_index": 2, "timestamp": 6.0, "state": "active_play"},
        ])
        analyzer._client = mock_client

        frames = _make_frames([0.0, 3.0, 6.0])
        labels = analyzer.scan(frames)

        assert len(labels) == 3
        assert labels[0].game_state == GameState.ACTIVE_PLAY
        assert labels[1].game_state == GameState.CORNER_KICK
        assert labels[2].game_state == GameState.ACTIVE_PLAY

    def test_scan_batches_frames(self, analyzer):
        """Frames are split into batches of COARSE_BATCH_SIZE."""
        mock_client = MagicMock()
        # Return valid responses for each batch
        batch1_response = _make_scan_response([
            {"frame_index": i, "timestamp": float(i * 3), "state": "active_play"}
            for i in range(COARSE_BATCH_SIZE)
        ])
        batch2_response = _make_scan_response([
            {"frame_index": i, "timestamp": float((COARSE_BATCH_SIZE + i) * 3), "state": "goal_kick"}
            for i in range(3)
        ])
        mock_client.messages.create.side_effect = [batch1_response, batch2_response]
        analyzer._client = mock_client

        frames = _make_frames([float(i * 3) for i in range(COARSE_BATCH_SIZE + 3)])
        labels = analyzer.scan(frames)

        assert len(labels) == COARSE_BATCH_SIZE + 3
        assert mock_client.messages.create.call_count == 2

    def test_scan_fail_open_on_api_error(self, analyzer):
        """On API error, all frames in the batch are marked ACTIVE_PLAY."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API timeout")
        analyzer._client = mock_client

        frames = _make_frames([0.0, 3.0, 6.0])
        labels = analyzer.scan(frames)

        assert len(labels) == 3
        assert all(l.game_state == GameState.ACTIVE_PLAY for l in labels)

    def test_scan_empty_frames(self, analyzer):
        labels = analyzer.scan([])
        assert labels == []


# ---------------------------------------------------------------------------
# Tests: _parse_scan_response()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseScanResponse:

    def test_parse_clean_json(self, analyzer):
        batch = _make_frames([0.0, 3.0])
        response = _make_scan_response([
            {"frame_index": 0, "timestamp": 0.0, "state": "corner_kick"},
            {"frame_index": 1, "timestamp": 3.0, "state": "goal_kick"},
        ])
        labels = analyzer._parse_scan_response(response, batch)

        assert len(labels) == 2
        assert labels[0].game_state == GameState.CORNER_KICK
        assert labels[0].timestamp_sec == 0.0
        assert labels[1].game_state == GameState.GOAL_KICK
        assert labels[1].timestamp_sec == 3.0

    def test_parse_markdown_fenced_json(self, analyzer):
        batch = _make_frames([0.0])
        body = '```json\n[{"frame_index": 0, "timestamp": 0.0, "state": "replay"}]\n```'
        response = SimpleNamespace(content=[SimpleNamespace(text=body)])
        labels = analyzer._parse_scan_response(response, batch)

        assert len(labels) == 1
        assert labels[0].game_state == GameState.REPLAY

    def test_parse_malformed_json_fail_open(self, analyzer):
        batch = _make_frames([0.0, 3.0])
        response = SimpleNamespace(content=[SimpleNamespace(text="Not valid JSON")])
        labels = analyzer._parse_scan_response(response, batch)

        assert len(labels) == 2
        assert all(l.game_state == GameState.ACTIVE_PLAY for l in labels)

    def test_parse_unknown_state_defaults_to_active(self, analyzer):
        batch = _make_frames([0.0])
        response = _make_scan_response([
            {"frame_index": 0, "timestamp": 0.0, "state": "unknown_state"},
        ])
        labels = analyzer._parse_scan_response(response, batch)

        assert labels[0].game_state == GameState.ACTIVE_PLAY

    def test_parse_uses_batch_timestamps(self, analyzer):
        """Frame timestamps come from our batch, not the VLM echo."""
        batch = _make_frames([10.0, 13.0])
        response = _make_scan_response([
            {"frame_index": 0, "timestamp": 999.0, "state": "active_play"},
            {"frame_index": 1, "timestamp": 999.0, "state": "active_play"},
        ])
        labels = analyzer._parse_scan_response(response, batch)

        assert labels[0].timestamp_sec == 10.0
        assert labels[1].timestamp_sec == 13.0


# ---------------------------------------------------------------------------
# Tests: refine_event()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRefineEvent:

    def test_refine_confirmed_event(self, analyzer):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_refine_response(
            confirmed=True, start=25.0, end=40.0, reasoning="Clear corner kick"
        )
        analyzer._client = mock_client

        frames = _make_frames([25.0, 26.0, 27.0, 28.0, 29.0, 30.0])
        boundary = analyzer.refine_event(frames, "corner_kick", 30.0)

        assert boundary is not None
        assert boundary.confirmed is True
        assert boundary.clip_start_sec == 25.0
        assert boundary.clip_end_sec == 40.0
        assert boundary.event_type == "corner_kick"
        assert boundary.reasoning == "Clear corner kick"

    def test_refine_unconfirmed_event(self, analyzer):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_refine_response(
            confirmed=False, start=25.0, end=40.0, reasoning="This is a throw-in"
        )
        analyzer._client = mock_client

        frames = _make_frames([25.0, 26.0, 27.0])
        boundary = analyzer.refine_event(frames, "corner_kick", 30.0)

        assert boundary is not None
        assert boundary.confirmed is False
        assert boundary.reasoning == "This is a throw-in"

    def test_refine_fail_open_on_api_error(self, analyzer):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API timeout")
        analyzer._client = mock_client

        frames = _make_frames([25.0, 26.0])
        boundary = analyzer.refine_event(frames, "corner_kick", 30.0)

        # Fail-open: returns None (caller handles it)
        assert boundary is None

    def test_refine_prompt_contains_event_type(self, analyzer):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_refine_response(
            confirmed=True, start=10.0, end=25.0,
        )
        analyzer._client = mock_client

        frames = _make_frames([10.0, 11.0])
        analyzer.refine_event(frames, "goal_kick", 15.0)

        call_kwargs = mock_client.messages.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        # Last text block is the prompt
        prompt_text = content[-1]["text"]
        assert "goal kick" in prompt_text

    def test_refine_sends_images(self, analyzer):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_refine_response(
            confirmed=True, start=10.0, end=25.0,
        )
        analyzer._client = mock_client

        frames = _make_frames([10.0, 11.0, 12.0])
        analyzer.refine_event(frames, "corner_kick", 11.0)

        call_kwargs = mock_client.messages.create.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        image_blocks = [c for c in content if c.get("type") == "image"]
        assert len(image_blocks) == 3


# ---------------------------------------------------------------------------
# Tests: _parse_refine_response()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseRefineResponse:

    def test_parse_clean_json(self, analyzer):
        response = _make_refine_response(True, 25.0, 40.0, "Good corner kick")
        boundary = analyzer._parse_refine_response(response, "corner_kick", 30.0)

        assert boundary.confirmed is True
        assert boundary.clip_start_sec == 25.0
        assert boundary.clip_end_sec == 40.0

    def test_parse_malformed_json_fail_closed(self, analyzer):
        response = SimpleNamespace(content=[SimpleNamespace(text="Not JSON")])
        boundary = analyzer._parse_refine_response(response, "corner_kick", 30.0)

        # Fail-closed: non-JSON means VLM couldn't confirm → unconfirmed
        assert boundary.confirmed is False
        assert boundary.clip_start_sec == 20.0  # center - 10
        assert boundary.clip_end_sec == 40.0     # center + 10
        assert "parse_error" in boundary.reasoning

    def test_parse_markdown_fenced_response(self, analyzer):
        body = '```json\n{"confirmed": true, "clip_start_sec": 10.0, "clip_end_sec": 20.0, "reasoning": "ok"}\n```'
        response = SimpleNamespace(content=[SimpleNamespace(text=body)])
        boundary = analyzer._parse_refine_response(response, "goal_kick", 15.0)

        assert boundary.confirmed is True
        assert boundary.clip_start_sec == 10.0


# ---------------------------------------------------------------------------
# Tests: lazy client init
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClientInit:

    def test_client_not_created_on_init(self, analyzer):
        assert analyzer._client is None

    def test_client_reused(self, analyzer):
        mock_client = MagicMock()
        analyzer._client = mock_client
        assert analyzer._get_client() is mock_client
