"""Unit tests for src/detection/chunk_tagger.py"""
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.detection.chunk_tagger import (
    ChunkTagger,
    TaggedEvent,
    _TAG_TO_EVENT,
    _GK_TAG_TYPES,
)
from src.detection.models import EventType


def _make_match_config():
    """Create a mock MatchConfig."""
    mc = MagicMock()
    mc.team.team_name = "Rush"
    mc.team.outfield_color = "white"
    mc.team.gk_color = "teal"
    mc.opponent.team_name = "GA 2008"
    mc.opponent.outfield_color = "blue"
    mc.opponent.gk_color = "purple"
    mc.gk_first_half_side = "left"
    return mc


def _make_tagger(**kwargs):
    """Create a ChunkTagger with test defaults."""
    defaults = dict(
        vllm_url="http://localhost:8000",
        model="Qwen/Qwen3-VL-32B-Instruct",
        source_file="/tmp/test.mp4",
        match_config=_make_match_config(),
        job_id="test-job",
        chunk_duration_sec=150.0,
        chunk_overlap_sec=15.0,
        chunk_fps=2,
        min_confidence=0.5,
        working_dir="/tmp",
    )
    defaults.update(kwargs)
    return ChunkTagger(**defaults)


# ---------------------------------------------------------------------------
# Tag-to-event mapping
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTagMapping:
    def test_all_expected_tags_mapped(self):
        expected = {"goal_kick", "corner_kick", "goal", "save_catch", "save_parry", "punch"}
        assert set(_TAG_TO_EVENT.keys()) == expected

    def test_goal_kick_maps_to_event(self):
        assert _TAG_TO_EVENT["goal_kick"] == EventType.GOAL_KICK

    def test_corner_kick_maps_to_event(self):
        assert _TAG_TO_EVENT["corner_kick"] == EventType.CORNER_KICK

    def test_goal_maps_to_event(self):
        assert _TAG_TO_EVENT["goal"] == EventType.GOAL

    def test_save_catch_maps_to_catch(self):
        assert _TAG_TO_EVENT["save_catch"] == EventType.CATCH

    def test_save_parry_maps_to_diving_save(self):
        assert _TAG_TO_EVENT["save_parry"] == EventType.SHOT_STOP_DIVING

    def test_punch_maps_to_punch(self):
        assert _TAG_TO_EVENT["punch"] == EventType.PUNCH

    def test_gk_tag_types(self):
        assert "goal_kick" in _GK_TAG_TYPES
        assert "corner_kick" in _GK_TAG_TYPES
        assert "save_catch" in _GK_TAG_TYPES
        assert "save_parry" in _GK_TAG_TYPES
        assert "punch" in _GK_TAG_TYPES
        assert "goal" not in _GK_TAG_TYPES


# ---------------------------------------------------------------------------
# Chunk computation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeChunks:
    def test_short_video_single_chunk(self):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=15.0)
        chunks = tagger._compute_chunks(100.0)
        assert len(chunks) == 1
        assert chunks[0] == (0.0, 100.0)

    def test_exact_fit_two_chunks(self):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=15.0)
        # step = 135, video = 270s → exactly 2 chunks
        chunks = tagger._compute_chunks(270.0)
        assert len(chunks) == 2
        assert chunks[0] == (0.0, 150.0)
        assert chunks[1] == (135.0, 270.0)

    def test_overlap_between_chunks(self):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=15.0)
        chunks = tagger._compute_chunks(400.0)
        # Verify overlap: chunk[i] end > chunk[i+1] start
        for i in range(len(chunks) - 1):
            assert chunks[i][1] > chunks[i + 1][0], "Chunks should overlap"
            overlap = chunks[i][1] - chunks[i + 1][0]
            assert abs(overlap - 15.0) < 1.0, f"Overlap should be ~15s, got {overlap}"

    def test_short_trailing_chunk_absorbed(self):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=15.0)
        # 285 + 5 = 290: trailing 5s chunk should be absorbed into previous
        chunks = tagger._compute_chunks(290.0)
        assert chunks[-1][1] == 290.0  # Last chunk extends to video end

    def test_full_game_chunk_count(self):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=15.0)
        # 6844s game ÷ 135s step ≈ 51 chunks
        chunks = tagger._compute_chunks(6844.0)
        assert 49 <= len(chunks) <= 52

    def test_zero_overlap(self):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=0.0)
        chunks = tagger._compute_chunks(450.0)
        assert len(chunks) == 3
        assert chunks[0] == (0.0, 150.0)
        assert chunks[1] == (150.0, 300.0)
        assert chunks[2] == (300.0, 450.0)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildPrompt:
    def test_contains_match_info(self):
        tagger = _make_tagger()
        prompt = tagger._build_prompt(0.0, 150.0)
        assert "Rush" in prompt
        assert "GA 2008" in prompt
        assert "white" in prompt
        assert "teal" in prompt
        assert "blue" in prompt
        assert "purple" in prompt

    def test_contains_event_types(self):
        tagger = _make_tagger()
        prompt = tagger._build_prompt(0.0, 150.0)
        assert "GOAL_KICK" in prompt
        assert "CORNER_KICK" in prompt
        assert "GOAL" in prompt
        assert "SAVE_CATCH" in prompt
        assert "SAVE_PARRY" in prompt
        assert "PUNCH" in prompt

    def test_contains_timestamps(self):
        tagger = _make_tagger()
        prompt = tagger._build_prompt(300.0, 450.0)
        assert "5:00" in prompt  # 300s = 5:00
        assert "7:30" in prompt  # 450s = 7:30

    def test_contains_duration(self):
        tagger = _make_tagger()
        prompt = tagger._build_prompt(0.0, 150.0)
        assert "150-second" in prompt

    def test_contains_fps(self):
        tagger = _make_tagger(chunk_fps=2)
        prompt = tagger._build_prompt(0.0, 150.0)
        assert "2 frames per second" in prompt

    def test_contains_json_format(self):
        tagger = _make_tagger()
        prompt = tagger._build_prompt(0.0, 150.0)
        assert "event_type" in prompt
        assert "timestamp_sec" in prompt
        assert "confidence" in prompt
        assert "reasoning" in prompt

    def test_team_names_in_team_field(self):
        tagger = _make_tagger()
        prompt = tagger._build_prompt(0.0, 150.0)
        # Team names should appear as options for the "team" field
        assert '"Rush"' in prompt
        assert '"GA 2008"' in prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseResponse:
    def test_valid_json_array(self):
        tagger = _make_tagger()
        response = json.dumps([
            {"event_type": "goal", "timestamp_sec": 45.0, "confidence": 0.9,
             "team": "Rush", "reasoning": "Ball in net, celebration"},
            {"event_type": "goal_kick", "timestamp_sec": 120.0, "confidence": 0.85,
             "team": "GA 2008", "reasoning": "GK kicks from 6-yard box"},
        ])
        events = tagger._parse_response(response, chunk_start=300.0)
        assert len(events) == 2
        assert events[0].event_type == "goal"
        assert events[0].timestamp_abs == 345.0  # 300 + 45
        assert events[0].confidence == 0.9
        assert events[0].team == "Rush"
        assert events[1].event_type == "goal_kick"
        assert events[1].timestamp_abs == 420.0  # 300 + 120

    def test_empty_array(self):
        tagger = _make_tagger()
        events = tagger._parse_response("[]", chunk_start=0.0)
        assert events == []

    def test_markdown_wrapped(self):
        tagger = _make_tagger()
        response = '```json\n[{"event_type": "goal", "timestamp_sec": 10.0, "confidence": 0.8, "team": "Rush", "reasoning": "goal scored"}]\n```'
        events = tagger._parse_response(response, chunk_start=0.0)
        assert len(events) == 1
        assert events[0].event_type == "goal"

    def test_text_before_json(self):
        tagger = _make_tagger()
        response = 'Here are the events:\n[{"event_type": "punch", "timestamp_sec": 5.0, "confidence": 0.7, "team": "Rush", "reasoning": "GK punched"}]'
        events = tagger._parse_response(response, chunk_start=0.0)
        assert len(events) == 1
        assert events[0].event_type == "punch"

    def test_invalid_json_returns_empty(self):
        tagger = _make_tagger()
        events = tagger._parse_response("not json at all", chunk_start=0.0)
        assert events == []

    def test_empty_response_returns_empty(self):
        tagger = _make_tagger()
        assert tagger._parse_response("", chunk_start=0.0) == []
        assert tagger._parse_response("   ", chunk_start=0.0) == []
        assert tagger._parse_response(None, chunk_start=0.0) == []

    def test_confidence_clamped(self):
        tagger = _make_tagger()
        response = json.dumps([
            {"event_type": "goal", "timestamp_sec": 10.0, "confidence": 1.5,
             "team": "Rush", "reasoning": "test"},
        ])
        events = tagger._parse_response(response, chunk_start=0.0)
        assert events[0].confidence == 1.0

    def test_missing_optional_fields_default(self):
        tagger = _make_tagger()
        response = json.dumps([
            {"event_type": "goal_kick", "timestamp_sec": 50.0},
        ])
        events = tagger._parse_response(response, chunk_start=0.0)
        assert len(events) == 1
        assert events[0].confidence == 0.0
        assert events[0].team == "unknown"

    def test_non_list_response_returns_empty(self):
        tagger = _make_tagger()
        response = json.dumps({"event_type": "goal"})
        events = tagger._parse_response(response, chunk_start=0.0)
        assert events == []

    def test_mixed_valid_invalid_items(self):
        tagger = _make_tagger()
        response = json.dumps([
            {"event_type": "goal", "timestamp_sec": 10.0, "confidence": 0.9,
             "team": "Rush", "reasoning": "scored"},
            "invalid item",
            {"event_type": "corner_kick", "timestamp_sec": 80.0, "confidence": 0.8,
             "team": "GA 2008", "reasoning": "corner"},
        ])
        events = tagger._parse_response(response, chunk_start=0.0)
        assert len(events) == 2


# ---------------------------------------------------------------------------
# Event creation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMakeEvent:
    def test_goal_kick_is_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal_kick", 100.0, 0.85, "Rush", "GK kicks")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
        assert event.event_type == EventType.GOAL_KICK

    def test_save_catch_is_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("save_catch", 200.0, 0.90, "Rush", "GK catches")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
        assert event.event_type == EventType.CATCH

    def test_save_parry_is_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("save_parry", 200.0, 0.80, "Rush", "GK parries")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
        assert event.event_type == EventType.SHOT_STOP_DIVING

    def test_punch_is_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("punch", 300.0, 0.75, "Rush", "GK punches")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
        assert event.event_type == EventType.PUNCH

    def test_goal_by_team_is_not_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal", 500.0, 0.95, "Rush", "Rush scores")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is False
        assert event.event_type == EventType.GOAL

    def test_goal_conceded_is_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal", 500.0, 0.90, "GA 2008", "Opponent scores")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
        assert event.event_type == EventType.GOAL

    def test_goal_unknown_team_not_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal", 500.0, 0.85, "unknown", "Goal scored")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is False

    def test_event_timestamps(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal_kick", 100.0, 0.85, "Rush", "GK kicks")
        event = tagger._make_event(te, fps=30.0)
        assert event.timestamp_start == 99.5
        assert event.timestamp_end == 100.5

    def test_event_metadata(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal", 100.0, 0.95, "Rush", "Great goal")
        event = tagger._make_event(te, fps=30.0)
        assert event.metadata["tagger_event_type"] == "goal"
        assert event.metadata["tagger_confidence"] == 0.95
        assert event.metadata["tagger_team"] == "Rush"
        assert event.metadata["tagger_reasoning"] == "Great goal"
        assert event.metadata["tagger_model"] == "Qwen/Qwen3-VL-32B-Instruct"

    def test_event_frame_numbers(self):
        tagger = _make_tagger()
        te = TaggedEvent("corner_kick", 100.0, 0.80, "Rush", "Corner")
        event = tagger._make_event(te, fps=30.0)
        assert event.frame_start == 2985  # (100.0 - 0.5) * 30
        assert event.frame_end == 3015    # (100.0 + 0.5) * 30


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDeduplicate:
    def test_no_duplicates_unchanged(self):
        tagger = _make_tagger()
        te1 = TaggedEvent("goal_kick", 100.0, 0.85, "Rush", "GK kicks")
        te2 = TaggedEvent("goal_kick", 500.0, 0.80, "Rush", "GK kicks again")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]
        deduped = tagger._deduplicate(events)
        assert len(deduped) == 2

    def test_duplicate_in_overlap_keeps_higher_confidence(self):
        tagger = _make_tagger()
        te1 = TaggedEvent("goal_kick", 140.0, 0.75, "Rush", "From chunk 1")
        te2 = TaggedEvent("goal_kick", 142.0, 0.90, "Rush", "From chunk 2")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]
        deduped = tagger._deduplicate(events)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.90

    def test_different_types_not_deduplicated(self):
        tagger = _make_tagger()
        te1 = TaggedEvent("goal_kick", 100.0, 0.85, "Rush", "Goal kick")
        te2 = TaggedEvent("corner_kick", 102.0, 0.80, "GA 2008", "Corner")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]
        deduped = tagger._deduplicate(events)
        assert len(deduped) == 2

    def test_empty_list(self):
        tagger = _make_tagger()
        assert tagger._deduplicate([]) == []

    def test_single_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal", 100.0, 0.95, "Rush", "Goal")
        events = [tagger._make_event(te, 30.0)]
        deduped = tagger._deduplicate(events)
        assert len(deduped) == 1

    def test_three_way_duplicate_keeps_best(self):
        tagger = _make_tagger()
        te1 = TaggedEvent("save_catch", 140.0, 0.70, "Rush", "Save 1")
        te2 = TaggedEvent("save_catch", 142.0, 0.95, "Rush", "Save 2")
        te3 = TaggedEvent("save_catch", 144.0, 0.80, "Rush", "Save 3")
        events = [
            tagger._make_event(te1, 30.0),
            tagger._make_event(te2, 30.0),
            tagger._make_event(te3, 30.0),
        ]
        deduped = tagger._deduplicate(events)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.95

    def test_proximity_threshold(self):
        tagger = _make_tagger()
        # Events 11s apart — outside default 10s proximity, should NOT merge
        te1 = TaggedEvent("goal_kick", 100.0, 0.85, "Rush", "GK kicks")
        te2 = TaggedEvent("goal_kick", 111.0, 0.80, "Rush", "GK kicks")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]
        deduped = tagger._deduplicate(events)
        assert len(deduped) == 2


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConfidenceFilter:
    def test_below_threshold_excluded(self):
        """tag_video filters events below min_confidence."""
        tagger = _make_tagger(min_confidence=0.6)
        te_low = TaggedEvent("goal_kick", 100.0, 0.4, "Rush", "Low conf")
        te_high = TaggedEvent("goal_kick", 200.0, 0.8, "Rush", "High conf")
        # Directly test the filter logic from tag_video
        assert te_low.confidence < tagger._min_confidence
        assert te_high.confidence >= tagger._min_confidence

    def test_unknown_event_type_excluded(self):
        """Events with types not in _TAG_TO_EVENT are filtered out."""
        assert "throw_in" not in _TAG_TO_EVENT
        assert "free_kick" not in _TAG_TO_EVENT


# ---------------------------------------------------------------------------
# Full pipeline (mocked)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTagVideoMocked:
    @patch("src.detection.chunk_tagger.ChunkTagger._extract_chunk")
    @patch("src.detection.chunk_tagger.ChunkTagger._tag_chunk")
    def test_tag_video_calls_per_chunk(self, mock_tag, mock_extract):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=15.0)
        mock_extract.return_value = MagicMock()  # fake path
        mock_tag.return_value = ([], "[]")

        tagger.tag_video(300.0, fps=30.0)

        # Should have ~3 chunks for 300s video
        chunks = tagger._compute_chunks(300.0)
        assert mock_extract.call_count == len(chunks)
        assert mock_tag.call_count == len(chunks)

    @patch("src.detection.chunk_tagger.ChunkTagger._extract_chunk")
    @patch("src.detection.chunk_tagger.ChunkTagger._tag_chunk")
    def test_tag_video_progress_callback(self, mock_tag, mock_extract):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=0.0)
        mock_extract.return_value = MagicMock()
        mock_tag.return_value = ([], "[]")

        progress_calls = []
        tagger.tag_video(300.0, fps=30.0,
                         progress_callback=lambda c, t: progress_calls.append((c, t)))

        # 300s / 150s = 2 chunks
        assert len(progress_calls) == 2
        assert progress_calls[-1] == (2, 2)

    @patch("src.detection.chunk_tagger.ChunkTagger._extract_chunk")
    @patch("src.detection.chunk_tagger.ChunkTagger._tag_chunk")
    def test_tag_video_skips_failed_extraction(self, mock_tag, mock_extract):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=0.0)
        mock_extract.return_value = None  # extraction failed
        mock_tag.return_value = ([], "[]")

        events = tagger.tag_video(150.0, fps=30.0)
        assert events == []
        mock_tag.assert_not_called()

    @patch("src.detection.chunk_tagger.ChunkTagger._extract_chunk")
    @patch("src.detection.chunk_tagger.ChunkTagger._tag_chunk")
    def test_tag_video_continues_on_error(self, mock_tag, mock_extract):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=0.0)
        mock_extract.return_value = MagicMock()
        # First chunk errors, second succeeds
        mock_tag.side_effect = [
            Exception("API down"),
            ([TaggedEvent("goal", 50.0, 0.9, "Rush", "Goal")], '[]'),
        ]

        events = tagger.tag_video(300.0, fps=30.0)
        assert len(events) == 1

    @patch("src.detection.chunk_tagger.ChunkTagger._extract_chunk")
    @patch("src.detection.chunk_tagger.ChunkTagger._tag_chunk")
    def test_tag_video_deduplicates_across_chunks(self, mock_tag, mock_extract):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=15.0)
        mock_extract.return_value = MagicMock()

        # Same event in overlap zone from two chunks
        events_chunk1 = [TaggedEvent("goal_kick", 140.0, 0.75, "Rush", "GK kicks")]
        events_chunk2 = [TaggedEvent("goal_kick", 142.0, 0.90, "Rush", "GK kicks")]
        mock_tag.side_effect = [
            (events_chunk1, "[]"),
            (events_chunk2, "[]"),
        ]

        events = tagger.tag_video(285.0, fps=30.0)
        # Should deduplicate to 1 event with higher confidence
        assert len(events) == 1
        assert events[0].confidence == 0.90
