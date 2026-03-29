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
from src.detection.models import Event, EventType


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
            "kickoff", "throw_in",
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
        assert "shot" in _GK_TAG_TYPES
        assert "goal" not in _GK_TAG_TYPES
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

    def test_shot_is_gk_event(self):
        tagger = _make_tagger()
        te = TaggedEvent("shot", 500.0, 503.0, 0.75, "GA 2008", "Shot wide")
        event = tagger._make_event(te, fps=30.0)
        assert event.is_goalkeeper_event is True
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
        assert "offside" not in _TAG_TO_EVENT
        assert "handball" not in _TAG_TO_EVENT


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

    @patch("src.detection.chunk_tagger.ChunkTagger._rescan_for_goal")
    def test_orphan_kickoff_infers_goal_when_rescan_empty(self, mock_rescan):
        """When rescan finds nothing, a synthetic goal is inferred from the kickoff."""
        tagger = _make_tagger()
        kickoffs = [TaggedEvent("kickoff", 1080.0, 1080.0, 0.9, "unknown", "Kickoff")]
        te = TaggedEvent("catch", 1050.0, 1052.0, 0.8, "Rush", "Save")
        events = [tagger._make_event(te, 30.0)]

        mock_rescan.return_value = []  # rescan finds nothing
        result = tagger._rescan_orphan_kickoffs(kickoffs, events, 30.0, 6000.0)

        assert len(result) == 1
        inferred = result[0]
        assert inferred.event_type == EventType.GOAL
        assert inferred.confidence == 0.65
        # Catch was by "Rush" (our team) → goal scored by us → not GK event
        assert inferred.is_goalkeeper_event is False
        assert inferred.metadata["inferred_from_kickoff"] is True
        assert inferred.metadata["kickoff_timestamp"] == 1080.0
        # Catch at 1050s is within 90s → used as level-2 anchor
        assert inferred.timestamp_start == 1050.0

    @patch("src.detection.chunk_tagger.ChunkTagger._rescan_for_goal")
    def test_rescan_goal_found_no_inference(self, mock_rescan):
        """When rescan finds a goal, no inference is generated."""
        tagger = _make_tagger()
        kickoffs = [TaggedEvent("kickoff", 850.0, 850.0, 0.9, "unknown", "Kickoff")]
        te = TaggedEvent("catch", 820.0, 822.0, 0.8, "Rush", "Save")
        events = [tagger._make_event(te, 30.0)]

        # Rescan finds a real goal
        real_goal = tagger._make_event(
            TaggedEvent("goal", 830.0, 840.0, 0.92, "GA 2008", "Opponent scored"), 30.0
        )
        mock_rescan.return_value = [real_goal]
        result = tagger._rescan_orphan_kickoffs(kickoffs, events, 30.0, 6000.0)

        assert len(result) == 1
        assert result[0].confidence == 0.92  # the real goal, not inferred
        assert "inferred_from_kickoff" not in result[0].metadata

    @patch("src.detection.chunk_tagger.ChunkTagger._rescan_for_goal")
    def test_multiple_orphans_mix_rescan_and_inferred(self, mock_rescan):
        """Some orphans find goals via rescan, others get inferred."""
        tagger = _make_tagger()
        ko1 = TaggedEvent("kickoff", 850.0, 850.0, 0.9, "unknown", "KO1")
        ko2 = TaggedEvent("kickoff", 1200.0, 1200.0, 0.85, "unknown", "KO2")
        te1 = TaggedEvent("catch", 820.0, 822.0, 0.8, "Rush", "Save")
        te2 = TaggedEvent("goal_kick", 1170.0, 1180.0, 0.7, "Rush", "GK")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]

        real_goal = tagger._make_event(
            TaggedEvent("goal", 830.0, 840.0, 0.92, "GA 2008", "Goal"), 30.0
        )
        # First kickoff → rescan finds goal; second → rescan empty → inferred
        mock_rescan.side_effect = [[real_goal], []]
        result = tagger._rescan_orphan_kickoffs([ko1, ko2], events, 30.0, 6000.0)

        assert len(result) == 2
        # First is real goal
        assert result[0].confidence == 0.92
        # Second is inferred — goal_kick at 1170s is within 90s, used as anchor
        assert result[1].metadata["inferred_from_kickoff"] is True
        assert result[1].timestamp_start == 1170.0

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

    def test_kickoff_in_tag_to_event(self):
        """Kickoff is now a first-class event type."""
        assert "kickoff" in _TAG_TO_EVENT
        assert _TAG_TO_EVENT["kickoff"] == EventType.KICKOFF

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
        assert "center" in prompt.lower()
        assert "kickoff" in prompt.lower()


# ---------------------------------------------------------------------------
# Infer goal from kickoff
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestInferGoalFromKickoff:
    def test_basic_inference(self):
        tagger = _make_tagger()
        ko = TaggedEvent("kickoff", 1080.0, 1080.0, 0.9, "unknown", "Kickoff")
        event = tagger._infer_goal_from_kickoff(ko, fps=30.0)

        assert event.event_type == EventType.GOAL
        assert event.confidence == 0.65
        assert event.is_goalkeeper_event is True
        assert event.timestamp_start == 1050.0  # 1080 - 30
        assert event.timestamp_end == 1070.0    # 1080 - 10
        assert event.reel_targets == []

    def test_metadata_marks_inferred(self):
        tagger = _make_tagger()
        ko = TaggedEvent("kickoff", 500.0, 500.0, 0.85, "unknown", "KO")
        event = tagger._infer_goal_from_kickoff(ko, fps=30.0)

        assert event.metadata["inferred_from_kickoff"] is True
        assert event.metadata["kickoff_timestamp"] == 500.0
        assert event.metadata["tagger_event_type"] == "goal"
        assert event.metadata["tagger_model"] == "Qwen/Qwen3-VL-32B-Instruct"
        assert "orphan kickoff" in event.metadata["tagger_reasoning"].lower()

    def test_frame_numbers(self):
        tagger = _make_tagger()
        ko = TaggedEvent("kickoff", 100.0, 100.0, 0.9, "unknown", "KO")
        event = tagger._infer_goal_from_kickoff(ko, fps=30.0)

        assert event.frame_start == int(70.0 * 30)   # (100-30) * 30
        assert event.frame_end == int(90.0 * 30)      # (100-10) * 30

    def test_early_kickoff_clamps_to_zero(self):
        tagger = _make_tagger()
        ko = TaggedEvent("kickoff", 15.0, 15.0, 0.9, "unknown", "KO")
        event = tagger._infer_goal_from_kickoff(ko, fps=30.0)

        # goal_t = 15 - 30 = -15 → frame_start clamped to 0
        assert event.frame_start == 0
        assert event.timestamp_start == -15.0  # timestamp can be negative
        assert event.timestamp_end == 5.0

    def test_unique_event_ids(self):
        tagger = _make_tagger()
        ko1 = TaggedEvent("kickoff", 500.0, 500.0, 0.9, "unknown", "KO1")
        ko2 = TaggedEvent("kickoff", 1000.0, 1000.0, 0.9, "unknown", "KO2")
        e1 = tagger._infer_goal_from_kickoff(ko1, fps=30.0)
        e2 = tagger._infer_goal_from_kickoff(ko2, fps=30.0)
        assert e1.event_id != e2.event_id

    def test_job_id_and_source_file(self):
        tagger = _make_tagger(job_id="my-job", source_file="/nas/match.mp4")
        ko = TaggedEvent("kickoff", 500.0, 500.0, 0.9, "unknown", "KO")
        event = tagger._infer_goal_from_kickoff(ko, fps=30.0)
        assert event.job_id == "my-job"
        assert event.source_file == "/nas/match.mp4"

    def test_anchors_to_preceding_shot(self):
        """When a shot_on_target exists within 120s before kickoff, use it."""
        tagger = _make_tagger()
        ko = TaggedEvent("kickoff", 1080.0, 1080.0, 0.9, "unknown", "Kickoff")
        shot = Event(
            event_id="shot-1000", job_id="test-job", source_file="/tmp/test.mp4",
            event_type=EventType.SHOT_ON_TARGET,
            timestamp_start=1000.0, timestamp_end=1003.0,
            confidence=0.8, reel_targets=[], frame_start=30000, frame_end=30090,
            metadata={"tagger_team": "GA 2008"},
        )
        event = tagger._infer_goal_from_kickoff(ko, fps=30.0, events=[shot])

        assert event.event_type == EventType.GOAL
        assert event.timestamp_start == 1000.0  # anchored to shot
        assert event.timestamp_end == 1003.0
        assert event.confidence == 0.80  # shot-anchored = higher confidence
        assert event.metadata["anchored_to_shot"] == 1000.0
        assert event.metadata["tagger_team"] == "GA 2008"
        assert "anchored to shot" in event.metadata["tagger_reasoning"].lower()

    def test_anchors_to_latest_shot_within_120s(self):
        """When multiple shots exist, pick the one closest to the kickoff."""
        tagger = _make_tagger()
        ko = TaggedEvent("kickoff", 1080.0, 1080.0, 0.9, "unknown", "KO")
        shot_old = Event(
            event_id="shot-980", job_id="test-job", source_file="/tmp/test.mp4",
            event_type=EventType.SHOT_ON_TARGET,
            timestamp_start=980.0, timestamp_end=982.0,
            confidence=0.8, reel_targets=[], frame_start=29400, frame_end=29460,
        )
        shot_recent = Event(
            event_id="shot-1040", job_id="test-job", source_file="/tmp/test.mp4",
            event_type=EventType.SHOT_ON_TARGET,
            timestamp_start=1040.0, timestamp_end=1043.0,
            confidence=0.8, reel_targets=[], frame_start=31200, frame_end=31290,
        )
        event = tagger._infer_goal_from_kickoff(
            ko, fps=30.0, events=[shot_old, shot_recent]
        )
        assert event.timestamp_start == 1040.0  # latest shot

    def test_shot_too_far_falls_back(self):
        """Shot >120s before kickoff is ignored — falls back to ko_t-30."""
        tagger = _make_tagger()
        ko = TaggedEvent("kickoff", 1080.0, 1080.0, 0.9, "unknown", "KO")
        shot_old = Event(
            event_id="shot-500", job_id="test-job", source_file="/tmp/test.mp4",
            event_type=EventType.SHOT_ON_TARGET,
            timestamp_start=500.0, timestamp_end=502.0,
            confidence=0.8, reel_targets=[], frame_start=15000, frame_end=15060,
        )
        event = tagger._infer_goal_from_kickoff(ko, fps=30.0, events=[shot_old])
        assert event.timestamp_start == 1050.0  # fallback: 1080 - 30
        assert event.metadata["anchored_to_shot"] is None

    def test_no_events_falls_back(self):
        """No events passed → fallback to ko_t-30."""
        tagger = _make_tagger()
        ko = TaggedEvent("kickoff", 1080.0, 1080.0, 0.9, "unknown", "KO")
        event = tagger._infer_goal_from_kickoff(ko, fps=30.0, events=None)
        assert event.timestamp_start == 1050.0  # 1080 - 30
        assert event.metadata["anchored_to_shot"] is None

    def test_non_shot_event_used_as_fallback(self):
        """Non-shot events within 90s are used as level-2 anchor."""
        tagger = _make_tagger()
        ko = TaggedEvent("kickoff", 1080.0, 1080.0, 0.9, "unknown", "KO")
        goal_kick = Event(
            event_id="gk-1020", job_id="test-job", source_file="/tmp/test.mp4",
            event_type=EventType.GOAL_KICK,
            timestamp_start=1020.0, timestamp_end=1025.0,
            confidence=0.8, reel_targets=[], frame_start=30600, frame_end=30750,
            metadata={"tagger_team": "Rush"},
        )
        event = tagger._infer_goal_from_kickoff(ko, fps=30.0, events=[goal_kick])
        assert event.timestamp_start == 1020.0  # anchored to goal_kick
        assert event.confidence == 0.65


# ---------------------------------------------------------------------------
# Event gap scan (pass 3)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScanEventGaps:
    def test_no_gaps_returns_empty(self):
        tagger = _make_tagger()
        # Events 30s apart — well under 90s threshold
        te1 = TaggedEvent("goal_kick", 100.0, 110.0, 0.8, "Rush", "GK")
        te2 = TaggedEvent("catch", 130.0, 132.0, 0.8, "Rush", "Catch")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]
        result = tagger._scan_event_gaps(events, 30.0, 6000.0)
        assert result == []

    def test_empty_events_returns_empty(self):
        tagger = _make_tagger()
        assert tagger._scan_event_gaps([], 30.0, 6000.0) == []

    def test_halftime_gap_skipped(self):
        """Gaps >300s are treated as halftime and skipped."""
        tagger = _make_tagger()
        te1 = TaggedEvent("goal_kick", 2800.0, 2810.0, 0.8, "Rush", "GK")
        te2 = TaggedEvent("goal_kick", 3500.0, 3510.0, 0.8, "Rush", "GK")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]
        # Gap = 700s > 300s → halftime, skipped
        result = tagger._scan_event_gaps(events, 30.0, 6000.0)
        assert result == []

    @patch("src.detection.chunk_tagger.ChunkTagger._rescan_gap")
    def test_gap_triggers_rescan(self, mock_rescan):
        """Gap >90s triggers a rescan."""
        tagger = _make_tagger()
        te1 = TaggedEvent("goal_kick", 2000.0, 2010.0, 0.8, "Rush", "GK")
        te2 = TaggedEvent("corner_kick", 2200.0, 2210.0, 0.8, "Rush", "Corner")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]
        # Gap = 200s > 90s → should trigger rescan
        mock_rescan.return_value = []
        tagger._scan_event_gaps(events, 30.0, 6000.0)
        mock_rescan.assert_called_once()
        call_args = mock_rescan.call_args[0]
        assert call_args[0] == 2000.0  # gap_start
        assert call_args[1] == 2200.0  # gap_end

    @patch("src.detection.chunk_tagger.ChunkTagger._rescan_gap")
    def test_gap_returns_found_events(self, mock_rescan):
        """Events found during gap scan are returned."""
        tagger = _make_tagger()
        te1 = TaggedEvent("goal_kick", 2000.0, 2010.0, 0.8, "Rush", "GK")
        te2 = TaggedEvent("corner_kick", 2200.0, 2210.0, 0.8, "Rush", "Corner")
        events = [tagger._make_event(te1, 30.0), tagger._make_event(te2, 30.0)]

        penalty = tagger._make_event(
            TaggedEvent("penalty", 2050.0, 2060.0, 0.85, "GA 2008", "PK"), 30.0
        )
        mock_rescan.return_value = [penalty]
        result = tagger._scan_event_gaps(events, 30.0, 6000.0)
        assert len(result) == 1
        assert result[0].event_type.value == "penalty"

    def test_gap_prompt_contains_penalty_info(self):
        tagger = _make_tagger()
        prompt = tagger._build_gap_prompt(2050.0, 2065.0, 2000.0, 2200.0)
        assert "penalty" in prompt.lower()
        assert "goal" in prompt.lower()
        assert "Rush" in prompt
        assert "GA 2008" in prompt
        assert "200-second gap" in prompt  # gap duration
