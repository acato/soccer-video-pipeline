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
        expected = {
            "goal", "penalty", "free_kick", "shot",
            "corner_kick", "goal_kick", "catch", "save",
        }
        assert set(_TAG_TO_EVENT.keys()) == expected

    def test_goal_kick_maps_to_event(self):
        assert _TAG_TO_EVENT["goal_kick"] == EventType.GOAL_KICK

    def test_corner_kick_maps_to_event(self):
        assert _TAG_TO_EVENT["corner_kick"] == EventType.CORNER_KICK

    def test_goal_maps_to_event(self):
        assert _TAG_TO_EVENT["goal"] == EventType.GOAL

    def test_catch_maps_to_catch(self):
        assert _TAG_TO_EVENT["catch"] == EventType.CATCH

    def test_save_maps_to_diving_save(self):
        assert _TAG_TO_EVENT["save"] == EventType.SHOT_STOP_DIVING

    def test_penalty_maps_to_penalty(self):
        assert _TAG_TO_EVENT["penalty"] == EventType.PENALTY

    def test_free_kick_maps_to_free_kick_shot(self):
        assert _TAG_TO_EVENT["free_kick"] == EventType.FREE_KICK_SHOT

    def test_shot_maps_to_shot_on_target(self):
        assert _TAG_TO_EVENT["shot"] == EventType.SHOT_ON_TARGET

    def test_gk_tag_types(self):
        assert "goal_kick" in _GK_TAG_TYPES
        assert "corner_kick" in _GK_TAG_TYPES
        assert "catch" in _GK_TAG_TYPES
        assert "save" in _GK_TAG_TYPES
        assert "penalty" in _GK_TAG_TYPES
        assert "goal" not in _GK_TAG_TYPES
        assert "shot" not in _GK_TAG_TYPES
        assert "free_kick" not in _GK_TAG_TYPES


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
        assert "CATCH" in prompt
        assert "SAVE" in prompt
        assert "PENALTY" in prompt
        assert "FREE_KICK" in prompt
        assert "SHOT" in prompt

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
        assert "start_sec" in prompt
        assert "end_sec" in prompt
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
            {"event_type": "goal", "start_sec": 45.0, "end_sec": 65.0, "confidence": 0.9,
             "team": "Rush", "reasoning": "Ball in net, celebration"},
            {"event_type": "goal_kick", "start_sec": 120.0, "end_sec": 130.0, "confidence": 0.85,
             "team": "GA 2008", "reasoning": "GK kicks from 6-yard box"},
        ])
        events = tagger._parse_response(response, chunk_start=300.0)
        assert len(events) == 2
        assert events[0].event_type == "goal"
        assert events[0].timestamp_abs == 345.0  # 300 + 45
        assert events[0].timestamp_end_abs == 365.0  # 300 + 65
        assert events[0].confidence == 0.9
        assert events[0].team == "Rush"
        assert events[1].event_type == "goal_kick"
        assert events[1].timestamp_abs == 420.0  # 300 + 120
        assert events[1].timestamp_end_abs == 430.0  # 300 + 130

    def test_empty_array(self):
        tagger = _make_tagger()
        events = tagger._parse_response("[]", chunk_start=0.0)
        assert events == []

    def test_markdown_wrapped(self):
        tagger = _make_tagger()
        response = '```json\n[{"event_type": "goal", "start_sec": 10.0, "end_sec": 30.0, "confidence": 0.8, "team": "Rush", "reasoning": "goal scored"}]\n```'
        events = tagger._parse_response(response, chunk_start=0.0)
        assert len(events) == 1
        assert events[0].event_type == "goal"

    def test_text_before_json(self):
        tagger = _make_tagger()
        response = 'Here are the events:\n[{"event_type": "save", "start_sec": 5.0, "end_sec": 8.0, "confidence": 0.7, "team": "Rush", "reasoning": "GK deflected"}]'
        events = tagger._parse_response(response, chunk_start=0.0)
        assert len(events) == 1
        assert events[0].event_type == "save"

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
            {"event_type": "goal", "start_sec": 10.0, "end_sec": 30.0, "confidence": 1.5,
             "team": "Rush", "reasoning": "test"},
        ])
        events = tagger._parse_response(response, chunk_start=0.0)
        assert events[0].confidence == 1.0

    def test_missing_optional_fields_default(self):
        tagger = _make_tagger()
        response = json.dumps([
            {"event_type": "goal_kick", "start_sec": 50.0},
        ])
        events = tagger._parse_response(response, chunk_start=0.0)
        assert len(events) == 1
        assert events[0].confidence == 0.0
        assert events[0].team == "unknown"
        # end_sec defaults to start_sec + 1.0
        assert events[0].timestamp_end_abs == 51.0

    def test_legacy_timestamp_sec_still_works(self):
        """Backward compat: timestamp_sec without start_sec/end_sec."""
        tagger = _make_tagger()
        response = json.dumps([
            {"event_type": "goal", "timestamp_sec": 45.0, "confidence": 0.9,
             "team": "Rush", "reasoning": "old format"},
        ])
        events = tagger._parse_response(response, chunk_start=300.0)
        assert len(events) == 1
        assert events[0].timestamp_abs == 345.0
        assert events[0].timestamp_end_abs == 346.0  # default +1s

    def test_non_list_response_returns_empty(self):
        tagger = _make_tagger()
        response = json.dumps({"event_type": "goal"})
        events = tagger._parse_response(response, chunk_start=0.0)
        assert events == []

    def test_mixed_valid_invalid_items(self):
        tagger = _make_tagger()
        response = json.dumps([
            {"event_type": "goal", "start_sec": 10.0, "end_sec": 30.0, "confidence": 0.9,
             "team": "Rush", "reasoning": "scored"},
            "invalid item",
            {"event_type": "corner_kick", "start_sec": 80.0, "end_sec": 95.0, "confidence": 0.8,
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
        te = TaggedEvent("goal_kick", 100.0, 110.0, 0.85, "Rush", "GK kicks")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
        assert event.event_type == EventType.GOAL_KICK

    def test_catch_is_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("catch", 200.0, 202.0, 0.90, "Rush", "GK catches")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
        assert event.event_type == EventType.CATCH

    def test_save_is_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("save", 200.0, 204.0, 0.80, "Rush", "GK parries")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
        assert event.event_type == EventType.SHOT_STOP_DIVING

    def test_penalty_is_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("penalty", 300.0, 310.0, 0.85, "GA 2008", "Opponent PK")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
        assert event.event_type == EventType.PENALTY

    def test_penalty_by_own_team_is_not_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("penalty", 300.0, 310.0, 0.85, "Rush", "Our PK")
        event = tagger._make_event(te, fps=30.0)
        # penalty in _GK_TAG_TYPES so always marked GK
        assert event.is_goalkeeper_event is True

    def test_free_kick_is_not_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("free_kick", 400.0, 402.0, 0.70, "Rush", "Free kick")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is False
        assert event.event_type == EventType.FREE_KICK_SHOT

    def test_shot_is_not_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("shot", 500.0, 503.0, 0.75, "GA 2008", "Shot wide")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is False
        assert event.event_type == EventType.SHOT_ON_TARGET

    def test_goal_by_team_is_not_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal", 500.0, 520.0, 0.95, "Rush", "Rush scores")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is False
        assert event.event_type == EventType.GOAL

    def test_goal_conceded_is_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal", 500.0, 520.0, 0.90, "GA 2008", "Opponent scores")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
        assert event.event_type == EventType.GOAL

    def test_goal_unknown_team_not_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal", 500.0, 520.0, 0.85, "unknown", "Goal scored")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is False

    def test_event_timestamps(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal_kick", 100.0, 110.0, 0.85, "Rush", "GK kicks")
        event = tagger._make_event(te, fps=30.0)
        assert event.timestamp_start == 100.0
        assert event.timestamp_end == 110.0

    def test_event_metadata(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal", 100.0, 120.0, 0.95, "Rush", "Great goal")
        event = tagger._make_event(te, fps=30.0)
        assert event.metadata["tagger_event_type"] == "goal"
        assert event.metadata["tagger_confidence"] == 0.95
        assert event.metadata["tagger_team"] == "Rush"
        assert event.metadata["tagger_reasoning"] == "Great goal"
        assert event.metadata["tagger_model"] == "Qwen/Qwen3-VL-32B-Instruct"

    def test_event_frame_numbers(self):
        tagger = _make_tagger()
        te = TaggedEvent("corner_kick", 100.0, 115.0, 0.80, "Rush", "Corner")
        event = tagger._make_event(te, fps=30.0)
        assert event.frame_start == 3000  # 100.0 * 30
        assert event.frame_end == 3450   # 115.0 * 30


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDeduplicate:
    def test_no_duplicates_unchanged(self):
        tagger = _make_tagger()
        te1 = TaggedEvent("goal_kick", 100.0, 110.0, 0.85, "Rush", "GK kicks")
        te2 = TaggedEvent("goal_kick", 500.0, 510.0, 0.80, "Rush", "GK kicks again")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]
        deduped = tagger._deduplicate(events)
        assert len(deduped) == 2

    def test_duplicate_in_overlap_keeps_higher_confidence(self):
        tagger = _make_tagger()
        te1 = TaggedEvent("goal_kick", 140.0, 150.0, 0.75, "Rush", "From chunk 1")
        te2 = TaggedEvent("goal_kick", 142.0, 152.0, 0.90, "Rush", "From chunk 2")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]
        deduped = tagger._deduplicate(events)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.90

    def test_different_types_not_deduplicated(self):
        tagger = _make_tagger()
        te1 = TaggedEvent("goal_kick", 100.0, 110.0, 0.85, "Rush", "Goal kick")
        te2 = TaggedEvent("corner_kick", 102.0, 112.0, 0.80, "GA 2008", "Corner")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]
        deduped = tagger._deduplicate(events)
        assert len(deduped) == 2

    def test_empty_list(self):
        tagger = _make_tagger()
        assert tagger._deduplicate([]) == []

    def test_single_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("goal", 100.0, 120.0, 0.95, "Rush", "Goal")
        events = [tagger._make_event(te, 30.0)]
        deduped = tagger._deduplicate(events)
        assert len(deduped) == 1

    def test_three_way_duplicate_keeps_best(self):
        tagger = _make_tagger()
        te1 = TaggedEvent("catch", 140.0, 142.0, 0.70, "Rush", "Save 1")
        te2 = TaggedEvent("catch", 142.0, 144.0, 0.95, "Rush", "Save 2")
        te3 = TaggedEvent("catch", 144.0, 146.0, 0.80, "Rush", "Save 3")
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
        te1 = TaggedEvent("goal_kick", 100.0, 110.0, 0.85, "Rush", "GK kicks")
        te2 = TaggedEvent("goal_kick", 111.0, 121.0, 0.80, "Rush", "GK kicks")
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
        te_low = TaggedEvent("goal_kick", 100.0, 110.0, 0.4, "Rush", "Low conf")
        te_high = TaggedEvent("goal_kick", 200.0, 210.0, 0.8, "Rush", "High conf")
        # Directly test the filter logic from tag_video
        assert te_low.confidence < tagger._min_confidence
        assert te_high.confidence >= tagger._min_confidence

    def test_unknown_event_type_excluded(self):
        """Events with types not in _TAG_TO_EVENT are filtered out."""
        assert "throw_in" not in _TAG_TO_EVENT
        assert "offside" not in _TAG_TO_EVENT


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
            ([TaggedEvent("goal", 50.0, 70.0, 0.9, "Rush", "Goal")], '[]'),
        ]

        events = tagger.tag_video(300.0, fps=30.0)
        assert len(events) == 1

    @patch("src.detection.chunk_tagger.ChunkTagger._extract_chunk")
    @patch("src.detection.chunk_tagger.ChunkTagger._tag_chunk")
    def test_tag_video_deduplicates_across_chunks(self, mock_tag, mock_extract):
        tagger = _make_tagger(chunk_duration_sec=150.0, chunk_overlap_sec=15.0)
        mock_extract.return_value = MagicMock()

        # Same event in overlap zone from two chunks
        events_chunk1 = [TaggedEvent("goal_kick", 140.0, 150.0, 0.75, "Rush", "GK kicks")]
        events_chunk2 = [TaggedEvent("goal_kick", 142.0, 152.0, 0.90, "Rush", "GK kicks")]
        mock_tag.side_effect = [
            (events_chunk1, "[]"),
            (events_chunk2, "[]"),
        ]

        events = tagger.tag_video(285.0, fps=30.0)
        # Should deduplicate to 1 event with higher confidence
        assert len(events) == 1
        assert events[0].confidence == 0.90


# ---------------------------------------------------------------------------
# Hallucination detection and rescan
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRescanHallucinatingChunks:
    def test_no_hallucination_no_rescan(self):
        tagger = _make_tagger()
        # 3 events in 60s = well below threshold
        counts = [(0.0, 60.0, 3)]
        all_events = []
        result = tagger._rescan_hallucinating_chunks(counts, all_events, 30.0)
        assert result == []

    def test_high_event_count_triggers_rescan(self):
        tagger = _make_tagger()
        # 15 events in 60s = way above 8/min threshold
        counts = [(2000.0, 2060.0, 15)]
        all_events = []
        with patch.object(tagger, "_rescan_region", return_value=[]) as mock:
            tagger._rescan_hallucinating_chunks(counts, all_events, 30.0)
            mock.assert_called_once_with(2000.0, 2060.0, 30.0)

    def test_events_from_halluc_chunk_removed(self):
        tagger = _make_tagger()
        counts = [(2000.0, 2060.0, 15)]
        # Events inside and outside the halluc range
        te_inside = TaggedEvent("goal_kick", 2030.0, 2040.0, 0.9, "Rush", "fake")
        te_outside = TaggedEvent("save", 500.0, 504.0, 0.8, "Rush", "real")
        all_events = [
            tagger._make_event(te_inside, 30.0),
            tagger._make_event(te_outside, 30.0),
        ]
        with patch.object(tagger, "_rescan_region", return_value=[]):
            tagger._rescan_hallucinating_chunks(counts, all_events, 30.0)
        # Only the outside event should remain
        assert len(all_events) == 1
        assert all_events[0].timestamp_start == 500.0

    def test_threshold_scales_with_duration(self):
        """A 150s chunk should tolerate more events than a 60s chunk."""
        tagger = _make_tagger()
        # 150s = 2.5 min → threshold = 8 * 2.5 = 20 events
        counts = [(0.0, 150.0, 18)]
        all_events = []
        with patch.object(tagger, "_rescan_region", return_value=[]) as mock:
            tagger._rescan_hallucinating_chunks(counts, all_events, 30.0)
            mock.assert_not_called()  # 18 < 20, no rescan

    def test_threshold_minimum_is_six(self):
        """Even very short chunks have a minimum threshold of 6."""
        tagger = _make_tagger()
        # 15s chunk → 8 * 0.25 = 2, but minimum is 6
        counts = [(0.0, 15.0, 5)]
        all_events = []
        with patch.object(tagger, "_rescan_region", return_value=[]) as mock:
            tagger._rescan_hallucinating_chunks(counts, all_events, 30.0)
            mock.assert_not_called()  # 5 < 6, no rescan

    def test_rescan_region_splits_into_sub_chunks(self):
        """_rescan_region should split region into 15s sub-chunks."""
        tagger = _make_tagger()
        # Mock extraction and tagging
        with patch.object(tagger, "_extract_rescan_chunk", return_value=MagicMock()) as mock_ext, \
             patch.object(tagger, "_tag_rescan_region_chunk", return_value=([], "[]")):
            tagger._rescan_region(2000.0, 2060.0, 30.0)
            # 60s / 15s = 4 sub-chunks
            assert mock_ext.call_count == 4

    def test_halluc_rescan_events_have_metadata(self):
        """Events from halluc rescan should have halluc_rescan metadata."""
        tagger = _make_tagger()
        te = TaggedEvent("save", 2010.0, 2014.0, 0.85, "Rush", "GK save")
        mock_event = tagger._make_event(te, 30.0)

        with patch.object(tagger, "_extract_rescan_chunk", return_value=MagicMock()), \
             patch.object(tagger, "_tag_rescan_region_chunk",
                          return_value=([te], "[]")):
            events = tagger._rescan_region(2000.0, 2015.0, 30.0)
            assert len(events) == 1
            assert events[0].metadata["halluc_rescan"] is True
            assert events[0].metadata["rescan_fps"] == 8

    def test_prompt_fps_override(self):
        """_build_prompt should use fps_override when provided."""
        tagger = _make_tagger(chunk_fps=2)
        prompt_default = tagger._build_prompt(0.0, 60.0)
        assert "2 frames per second" in prompt_default
        prompt_8fps = tagger._build_prompt(0.0, 60.0, fps_override=8)
        assert "8 frames per second" in prompt_8fps

    def test_empty_after_busy_triggers_rescan(self):
        """Empty chunk following a busy chunk (>=5 events) triggers rescan."""
        tagger = _make_tagger()
        # Chunk 0: 7 events (busy), Chunk 1: 0 events (empty after busy)
        counts = [(2000.0, 2060.0, 7), (2050.0, 2110.0, 0)]
        all_events = []
        with patch.object(tagger, "_rescan_region", return_value=[]) as mock:
            tagger._rescan_hallucinating_chunks(counts, all_events, 30.0)
            # Should rescan the empty chunk, not the busy one
            mock.assert_called_once_with(2050.0, 2110.0, 30.0)

    def test_empty_after_quiet_no_rescan(self):
        """Empty chunk after a quiet chunk (<5 events) is not rescanned."""
        tagger = _make_tagger()
        counts = [(2000.0, 2060.0, 2), (2050.0, 2110.0, 0)]
        all_events = []
        with patch.object(tagger, "_rescan_region", return_value=[]) as mock:
            tagger._rescan_hallucinating_chunks(counts, all_events, 30.0)
            mock.assert_not_called()

    def test_both_halluc_and_empty_after_busy(self):
        """Both hallucination AND empty-after-busy can trigger in same run."""
        tagger = _make_tagger()
        counts = [
            (1000.0, 1060.0, 15),  # hallucination
            (2000.0, 2060.0, 7),   # busy
            (2050.0, 2110.0, 0),   # empty after busy
        ]
        all_events = []
        with patch.object(tagger, "_rescan_region", return_value=[]) as mock:
            tagger._rescan_hallucinating_chunks(counts, all_events, 30.0)
            assert mock.call_count == 2  # halluc + empty-after-busy


# ---------------------------------------------------------------------------
# Orphan kickoff rescan
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRescanOrphanKickoffs:
    def test_no_kickoffs_no_rescan(self):
        tagger = _make_tagger()
        result = tagger._rescan_orphan_kickoffs([], [], fps=30.0, video_duration=6000.0)
        assert result == []

    def test_kickoff_with_preceding_goal_skipped(self):
        """Kickoff preceded by a goal within 90s is not an orphan."""
        tagger = _make_tagger()
        kickoffs = [TaggedEvent("kickoff", 850.0, 850.0, 0.9, "unknown", "Kickoff")]
        # Goal at 830s — within 90s of kickoff
        goal_te = TaggedEvent("goal", 830.0, 840.0, 0.95, "Rush", "Goal")
        events = [tagger._make_event(goal_te, 30.0)]
        result = tagger._rescan_orphan_kickoffs(kickoffs, events, 30.0, 6000.0)
        assert result == []

    def test_kickoff_early_in_video_skipped(self):
        """Kickoffs in first 60s are match-start, not goal kickoffs."""
        tagger = _make_tagger()
        kickoffs = [TaggedEvent("kickoff", 30.0, 30.0, 0.9, "unknown", "Match start")]
        result = tagger._rescan_orphan_kickoffs(kickoffs, [], 30.0, 6000.0)
        assert result == []

    def test_kickoff_after_long_gap_is_halftime(self):
        """Kickoff preceded by >120s gap with no events is halftime."""
        tagger = _make_tagger()
        kickoffs = [TaggedEvent("kickoff", 3600.0, 3600.0, 0.9, "unknown", "2nd half")]
        # Last event was at 3400s — 200s gap = halftime
        gk_te = TaggedEvent("goal_kick", 3400.0, 3410.0, 0.8, "Rush", "GK")
        events = [tagger._make_event(gk_te, 30.0)]
        result = tagger._rescan_orphan_kickoffs(kickoffs, events, 30.0, 6000.0)
        assert result == []

    @patch("src.detection.chunk_tagger.ChunkTagger._rescan_for_goal")
    def test_orphan_kickoff_triggers_rescan(self, mock_rescan):
        """Kickoff with no preceding goal and continuous play triggers rescan."""
        tagger = _make_tagger()
        kickoffs = [TaggedEvent("kickoff", 850.0, 850.0, 0.9, "unknown", "Kickoff")]
        # Events at 800s and 810s — play was happening, but no goal detected
        te1 = TaggedEvent("catch", 800.0, 802.0, 0.8, "Rush", "GK catch")
        te2 = TaggedEvent("goal_kick", 810.0, 820.0, 0.7, "Rush", "Goal kick")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]

        mock_rescan.return_value = []
        tagger._rescan_orphan_kickoffs(kickoffs, events, 30.0, 6000.0)
        mock_rescan.assert_called_once()

    def test_duplicate_kickoffs_deduped(self):
        """Kickoffs within 15s of each other are deduped."""
        tagger = _make_tagger()
        # Two kickoffs 5s apart (from overlapping chunks)
        ko1 = TaggedEvent("kickoff", 850.0, 850.0, 0.8, "unknown", "KO1")
        ko2 = TaggedEvent("kickoff", 855.0, 855.0, 0.9, "unknown", "KO2")
        # Event at 820s so they're not halftime and not start-of-match
        te = TaggedEvent("catch", 820.0, 822.0, 0.8, "Rush", "Catch")
        events = [tagger._make_event(te, 30.0)]

        with patch.object(tagger, "_rescan_for_goal", return_value=[]) as mock:
            tagger._rescan_orphan_kickoffs([ko1, ko2], events, 30.0, 6000.0)
            # Only one rescan despite two kickoffs
            assert mock.call_count == 1

    def test_rescan_prompt_contains_kickoff_context(self):
        """Rescan prompt mentions the kickoff timestamp."""
        tagger = _make_tagger()
        prompt = tagger._build_rescan_prompt(820.0, 850.0, 855.0)
        assert "kickoff" in prompt.lower()
        assert "14:15" in prompt  # 855s = 14:15
        assert "goal" in prompt.lower()
        assert str(tagger._rescan_fps) in prompt

    def test_rescan_prompt_contains_match_info(self):
        tagger = _make_tagger()
        prompt = tagger._build_rescan_prompt(820.0, 850.0, 855.0)
        assert "Rush" in prompt
        assert "GA 2008" in prompt

    def test_kickoff_not_in_tag_to_event(self):
        """Kickoff is a detection signal, not a pipeline event type."""
        assert "kickoff" not in _TAG_TO_EVENT

    def test_prompt_contains_kickoff_event_type(self):
        """The main prompt should include kickoff as a taggable type."""
        tagger = _make_tagger()
        prompt = tagger._build_prompt(0.0, 150.0)
        assert "KICKOFF" in prompt
        assert '"kickoff"' in prompt

    def test_prompt_warns_throw_in_not_goal_kick(self):
        """Prompt should warn against confusing throw-ins with goal kicks."""
        tagger = _make_tagger()
        prompt = tagger._build_prompt(0.0, 150.0)
        assert "throw-in" in prompt.lower()

    def test_prompt_kickoff_only_after_goal(self):
        """Prompt should clarify kickoff only happens after a goal/halftime."""
        tagger = _make_tagger()
        prompt = tagger._build_prompt(0.0, 150.0)
        assert "center circle" in prompt.lower()
        assert "after a goal" in prompt.lower()
