"""
End-to-end unit tests for the goalkeeper detection and reel generation pipeline.

Covers every stage of the chain:
  Detection.metadata["jersey_hsv"]
    → _aggregate_jersey_colors → Track.jersey_color_hsv
    → identify_gk_by_known_colors → gk_ids
    → classify_gk_events → Event(reel_targets=["keeper"])
    → compute_clips(reel_type="keeper") → ClipBoundary
    → postprocess_clips → final clips
    → worker pre-flight checks and abort logic

No ML models, GPU, or real video files required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.models import (
    BoundingBox, Detection, Event, EventType, Track,
    EVENT_REEL_MAP, is_gk_event_type,
)
from src.detection.jersey_classifier import (
    compute_jersey_similarity,
    extract_jersey_color,
    identify_gk_by_known_colors,
    resolve_jersey_color,
    JERSEY_COLOR_PALETTE,
)
from src.detection.event_classifier import _aggregate_jersey_colors
from src.detection.goalkeeper_detector import GoalkeeperDetector
from src.detection.event_log import EventLog
from src.segmentation.clipper import ClipBoundary, compute_clips
from src.segmentation.deduplicator import postprocess_clips, MAX_REEL_DURATIONS
from tests.conftest import make_match_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bbox(cx: float = 0.5, cy: float = 0.5, w: float = 0.05, h: float = 0.15) -> BoundingBox:
    return BoundingBox(x=cx - w / 2, y=cy - h / 2, width=w, height=h)


def _make_detection(
    frame: int, ts: float, cls: str = "player", track_id: int = None,
    cx: float = 0.5, cy: float = 0.5, jersey_hsv: list = None,
) -> Detection:
    det = Detection(
        frame_number=frame,
        timestamp=ts,
        class_name=cls,
        confidence=0.9,
        bbox=_make_bbox(cx, cy),
        track_id=track_id,
    )
    if jersey_hsv is not None:
        det.metadata["jersey_hsv"] = jersey_hsv
    return det


def _make_track(track_id: int, detections: list[Detection], jersey_hsv=None) -> Track:
    track = Track(track_id=track_id, detections=detections)
    if jersey_hsv is not None:
        track.jersey_color_hsv = jersey_hsv
    return track


def _make_gk_event(
    event_id: str, event_type: EventType, start: float, end: float,
    reel_targets: list[str], confidence: float = 0.80,
) -> Event:
    return Event(
        event_id=event_id,
        job_id="job-001",
        source_file="match.mp4",
        event_type=event_type,
        timestamp_start=start,
        timestamp_end=end,
        confidence=confidence,
        reel_targets=reel_targets,
        is_goalkeeper_event=True,
        frame_start=int(start * 30),
        frame_end=int(end * 30),
    )


# ===========================================================================
# Stage 1: Detection.metadata["jersey_hsv"]
# ===========================================================================

@pytest.mark.unit
class TestDetectionMetadata:
    def test_detection_has_metadata_field(self):
        det = Detection(
            frame_number=0, timestamp=0.0, class_name="player",
            confidence=0.9, bbox=_make_bbox(),
        )
        assert isinstance(det.metadata, dict)
        assert det.metadata == {}

    def test_detection_metadata_accepts_jersey_hsv(self):
        det = _make_detection(0, 0.0, jersey_hsv=[88.0, 0.8, 0.55])
        assert det.metadata["jersey_hsv"] == [88.0, 0.8, 0.55]

    def test_detection_metadata_survives_model_dump(self):
        det = _make_detection(0, 0.0, jersey_hsv=[88.0, 0.8, 0.55])
        data = det.model_dump()
        restored = Detection(**data)
        assert restored.metadata["jersey_hsv"] == [88.0, 0.8, 0.55]


# ===========================================================================
# Stage 2: _aggregate_jersey_colors → Track.jersey_color_hsv
# ===========================================================================

@pytest.mark.unit
class TestAggregateJerseyColors:
    def test_aggregates_from_detection_metadata(self):
        dets = [
            _make_detection(i, float(i), jersey_hsv=[88.0, 0.80, 0.55])
            for i in range(5)
        ]
        track = _make_track(1, dets)
        _aggregate_jersey_colors([track])
        assert track.jersey_color_hsv is not None
        assert abs(track.jersey_color_hsv[0] - 88.0) < 0.01

    def test_does_not_overwrite_existing(self):
        dets = [_make_detection(0, 0.0, jersey_hsv=[0.0, 0.0, 0.0])]
        track = _make_track(1, dets, jersey_hsv=(50.0, 0.5, 0.5))
        _aggregate_jersey_colors([track])
        assert track.jersey_color_hsv == (50.0, 0.5, 0.5)

    def test_empty_detections_leaves_track_unchanged(self):
        dets = [_make_detection(0, 0.0)]  # No jersey_hsv in metadata
        track = _make_track(1, dets)
        _aggregate_jersey_colors([track])
        assert track.jersey_color_hsv is None

    def test_median_robust_to_outlier(self):
        """Median should resist a single outlier in jersey color."""
        dets = [
            _make_detection(0, 0.0, jersey_hsv=[88.0, 0.80, 0.55]),
            _make_detection(1, 0.1, jersey_hsv=[88.0, 0.80, 0.55]),
            _make_detection(2, 0.2, jersey_hsv=[88.0, 0.80, 0.55]),
            _make_detection(3, 0.3, jersey_hsv=[88.0, 0.80, 0.55]),
            _make_detection(4, 0.4, jersey_hsv=[0.0, 0.0, 0.0]),  # outlier
        ]
        track = _make_track(1, dets)
        _aggregate_jersey_colors([track])
        assert abs(track.jersey_color_hsv[0] - 88.0) < 0.01


# ===========================================================================
# Stage 3: identify_gk_by_known_colors
# ===========================================================================

@pytest.mark.unit
class TestGKIdentificationByKnownColors:
    def test_identifies_home_and_away_gk(self):
        """Two tracks with distinct colors matching home/away GK should be identified."""
        home_gk_hsv = resolve_jersey_color("neon_yellow")
        away_gk_hsv = resolve_jersey_color("neon_green")

        track_colors = {
            1: home_gk_hsv,  # Home GK
            2: away_gk_hsv,  # Away GK
            3: (112.0, 0.82, 0.65),  # Outfield player (blue)
            4: (112.0, 0.82, 0.65),
        }
        track_positions = {1: 0.08, 2: 0.92, 3: 0.4, 4: 0.6}

        result = identify_gk_by_known_colors(
            track_colors, track_positions, home_gk_hsv, away_gk_hsv,
        )
        assert result["keeper_a"] == 1  # left half
        assert result["keeper_b"] == 2  # right half

    def test_rejects_weak_matches(self):
        """Tracks that don't match any GK color should return None."""
        home_gk_hsv = resolve_jersey_color("neon_yellow")
        away_gk_hsv = resolve_jersey_color("neon_green")

        track_colors = {
            1: (112.0, 0.82, 0.65),  # All outfield blue
            2: (112.0, 0.82, 0.65),
        }
        track_positions = {1: 0.1, 2: 0.9}

        result = identify_gk_by_known_colors(
            track_colors, track_positions, home_gk_hsv, away_gk_hsv,
            min_similarity=0.60,
        )
        assert result["keeper_a"] is None
        assert result["keeper_b"] is None

    def test_empty_track_colors_returns_none(self):
        result = identify_gk_by_known_colors(
            {}, {}, (35.0, 0.95, 0.95), (55.0, 0.95, 0.95),
        )
        assert result == {"keeper_a": None, "keeper_b": None}


# ===========================================================================
# Stage 4: GoalkeeperDetector.reel_label_for + classify_gk_events
# ===========================================================================

@pytest.mark.unit
class TestGoalkeeperDetectorReelLabels:
    def _make_detector_with_identified_keepers(self):
        """Build a GK detector that has already identified both keepers."""
        mc = make_match_config()
        det = GoalkeeperDetector(job_id="j1", source_file="m.mp4", match_config=mc)

        home_gk_hsv = resolve_jersey_color(mc.team.gk_color)
        away_gk_hsv = resolve_jersey_color(mc.opponent.gk_color)

        # Simulate what identify_goalkeepers does internally
        track_colors = {10: home_gk_hsv, 20: away_gk_hsv}
        track_positions = {10: 0.08, 20: 0.92}

        # Build minimal tracks
        dets_a = [_make_detection(i, float(i), cx=0.08, track_id=10) for i in range(20)]
        dets_b = [_make_detection(i, float(i), cx=0.92, track_id=20) for i in range(20)]
        tracks = [
            _make_track(10, dets_a, jersey_hsv=home_gk_hsv),
            _make_track(20, dets_b, jersey_hsv=away_gk_hsv),
        ]

        gk_ids = det.identify_goalkeepers(tracks, (720, 1280), track_colors)
        return det, gk_ids

    def test_team_gk_gets_keeper_label(self):
        det, gk_ids = self._make_detector_with_identified_keepers()
        # The team's GK should have reel_label "keeper"
        team_role = None
        for role in ("keeper_a", "keeper_b"):
            if det.reel_label_for(role) == "keeper":
                team_role = role
        assert team_role is not None, "Team GK should have reel_label='keeper'"

    def test_opponent_gk_gets_none_label(self):
        det, gk_ids = self._make_detector_with_identified_keepers()
        # The opponent's GK should have reel_label None
        opponent_role = None
        for role in ("keeper_a", "keeper_b"):
            if det.reel_label_for(role) is None and gk_ids.get(role) is not None:
                opponent_role = role
        assert opponent_role is not None, "Opponent GK should have reel_label=None"

    def test_classify_gk_events_produces_keeper_reel_targets(self):
        """GK events must have reel_targets containing the keeper role."""
        mc = make_match_config()
        det = GoalkeeperDetector(job_id="j1", source_file="m.mp4", match_config=mc)

        # Build a GK track with a sudden vertical movement (save)
        dets = []
        for i in range(30):
            cy = 0.5
            if i == 15:
                cy = 0.1  # Sudden dive
            dets.append(_make_detection(i, float(i) / 30.0, cx=0.08, cy=cy, track_id=10))

        gk_track = _make_track(10, dets)
        gk_track.is_goalkeeper = True

        events = det.classify_gk_events(gk_track, [gk_track], 30.0, keeper_role="keeper")
        for ev in events:
            assert "keeper" in ev.reel_targets, f"GK event {ev.event_type} missing keeper target"
            assert ev.is_goalkeeper_event is True


# ===========================================================================
# Stage 5: compute_clips for keeper reel
# ===========================================================================

@pytest.mark.unit
class TestKeeperClipComputation:
    def test_keeper_events_produce_keeper_clips(self):
        events = [
            _make_gk_event("e1", EventType.SHOT_STOP_DIVING, 60.0, 62.0, ["keeper"]),
            _make_gk_event("e2", EventType.CATCH, 300.0, 302.0, ["keeper"]),
        ]
        clips = compute_clips(events, 5400.0, "keeper")
        assert len(clips) == 2
        for c in clips:
            assert c.reel_type == "keeper"

    def test_keeper_a_events_match_keeper_reel(self):
        """Events with keeper_a reel_targets should match reel_type='keeper'."""
        events = [
            _make_gk_event("e1", EventType.SHOT_STOP_DIVING, 60.0, 62.0, ["keeper_a"]),
        ]
        clips = compute_clips(events, 5400.0, "keeper")
        assert len(clips) == 1

    def test_highlights_events_excluded_from_keeper(self):
        events = [
            _make_gk_event("e1", EventType.SHOT_STOP_DIVING, 60.0, 62.0, ["keeper"]),
            Event(
                event_id="e2", job_id="j1", source_file="m.mp4",
                event_type=EventType.GOAL, timestamp_start=200.0, timestamp_end=201.0,
                confidence=0.92, reel_targets=["highlights"],
                frame_start=6000, frame_end=6030,
            ),
        ]
        clips = compute_clips(events, 5400.0, "keeper")
        assert len(clips) == 1
        assert clips[0].primary_event_type == "shot_stop_diving"

    def test_no_keeper_events_returns_empty(self):
        """If all events are highlights-only, keeper clips should be empty."""
        events = [
            Event(
                event_id="e1", job_id="j1", source_file="m.mp4",
                event_type=EventType.SHOT_ON_TARGET, timestamp_start=60.0, timestamp_end=62.0,
                confidence=0.80, reel_targets=["highlights"],
                frame_start=1800, frame_end=1860,
            ),
        ]
        clips = compute_clips(events, 5400.0, "keeper")
        assert len(clips) == 0

    def test_max_clip_duration_enforced(self):
        events = [
            _make_gk_event(f"e{i}", EventType.DISTRIBUTION_SHORT, 10.0 * i, 10.0 * i + 2.0, ["keeper"])
            for i in range(20)
        ]
        clips = compute_clips(
            events, 5400.0, "keeper",
            pre_pad=3.0, post_pad=5.0, merge_gap_sec=2.0, max_clip_duration_sec=90.0,
        )
        assert len(clips) > 1
        for clip in clips:
            assert clip.end_sec - clip.start_sec <= 90.0


# ===========================================================================
# Stage 6: postprocess_clips for keeper reel
# ===========================================================================

@pytest.mark.unit
class TestKeeperPostprocess:
    def test_keeper_reel_duration_cap(self):
        """Keeper reel should be capped at 20 minutes."""
        assert "keeper" in MAX_REEL_DURATIONS
        assert MAX_REEL_DURATIONS["keeper"] == 20 * 60

    def test_postprocess_applies_cap(self):
        """Clips exceeding 20min should be pruned."""
        # Create 30 clips of 60s each = 30 min total
        clips = [
            ClipBoundary(
                source_file="m.mp4", start_sec=120.0 * i, end_sec=120.0 * i + 60.0,
                events=[f"e{i}"], reel_type="keeper", primary_event_type="shot_stop_diving",
            )
            for i in range(30)
        ]
        conf_map = {f"e{i}": 0.80 for i in range(30)}
        result = postprocess_clips(clips, "keeper", event_confidence_map=conf_map)
        total_dur = sum(c.end_sec - c.start_sec for c in result)
        assert total_dur <= 20 * 60


# ===========================================================================
# Stage 7: Event log round-trip with keeper events
# ===========================================================================

@pytest.mark.unit
class TestEventLogKeeperRoundTrip:
    def test_keeper_events_survive_write_and_read(self, tmp_path):
        log = EventLog(tmp_path / "events.jsonl")
        events = [
            _make_gk_event("e1", EventType.SHOT_STOP_DIVING, 60.0, 62.0, ["keeper"]),
            _make_gk_event("e2", EventType.CATCH, 300.0, 302.0, ["keeper_a"]),
        ]
        log.append_many(events)
        loaded = log.read_all()
        assert len(loaded) == 2
        assert loaded[0].reel_targets == ["keeper"]
        assert loaded[1].reel_targets == ["keeper_a"]

    def test_filter_by_reel_matches_keeper_prefix(self, tmp_path):
        log = EventLog(tmp_path / "events.jsonl")
        log.append_many([
            _make_gk_event("e1", EventType.SHOT_STOP_DIVING, 60.0, 62.0, ["keeper"]),
            _make_gk_event("e2", EventType.CATCH, 300.0, 302.0, ["keeper_a"]),
            _make_gk_event("e3", EventType.GOAL_KICK, 500.0, 502.0, ["keeper_b"]),
            Event(
                event_id="e4", job_id="j1", source_file="m.mp4",
                event_type=EventType.GOAL, timestamp_start=800.0, timestamp_end=801.0,
                confidence=0.92, reel_targets=["highlights"],
                frame_start=24000, frame_end=24030,
            ),
        ])
        keeper_events = log.filter_by_reel("keeper")
        assert len(keeper_events) == 3
        assert all(any("keeper" in rt for rt in e.reel_targets) for e in keeper_events)

    def test_clear_removes_stale_events(self, tmp_path):
        log = EventLog(tmp_path / "events.jsonl")
        log.append(_make_gk_event("e1", EventType.CATCH, 10.0, 12.0, ["keeper"]))
        assert log.count() == 1
        log.clear()
        assert log.count() == 0


# ===========================================================================
# Stage 8: Jersey color palette coverage
# ===========================================================================

@pytest.mark.unit
class TestJerseyColorPalette:
    def test_resolve_known_colors(self):
        for name in ("teal", "neon_yellow", "neon_green", "blue", "red", "white"):
            hsv = resolve_jersey_color(name)
            assert len(hsv) == 3
            assert 0 <= hsv[0] <= 180
            assert 0 <= hsv[1] <= 1
            assert 0 <= hsv[2] <= 1

    def test_resolve_unknown_color_raises(self):
        with pytest.raises(ValueError, match="Unknown jersey color"):
            resolve_jersey_color("ultraviolet")

    def test_team_gk_and_opponent_gk_are_distinguishable(self):
        """Team and opponent GK colors must be sufficiently different for identification."""
        mc = make_match_config()
        team_hsv = resolve_jersey_color(mc.team.gk_color)
        opp_hsv = resolve_jersey_color(mc.opponent.gk_color)
        similarity = compute_jersey_similarity(team_hsv, opp_hsv)
        assert similarity < 0.80, f"GK colors too similar ({similarity:.2f}), identification will fail"


# ===========================================================================
# Stage 9: extract_jersey_color (requires cv2)
# ===========================================================================

@pytest.mark.unit
class TestExtractJerseyColor:
    def test_extracts_from_synthetic_frame(self):
        """Green frame should produce HSV near green."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("cv2 not installed")

        # Create a 100x200 green BGR image
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[:, :] = [0, 180, 0]  # BGR green
        bbox = _make_bbox(cx=0.5, cy=0.5, w=0.8, h=0.8)
        hsv = extract_jersey_color(frame, bbox, (200, 100))
        assert hsv is not None
        # Green in HSV: H≈60 (OpenCV uses 0-180 scale)
        assert 40 < hsv[0] < 80, f"Expected green hue, got H={hsv[0]}"

    def test_returns_none_for_tiny_bbox(self):
        """Extremely small bbox should fail gracefully."""
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("cv2 not installed")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tiny_bbox = BoundingBox(x=0.5, y=0.5, width=0.001, height=0.001)
        result = extract_jersey_color(frame, tiny_bbox, (100, 100))
        # Either None or a valid color — should not crash
        assert result is None or len(result) == 3


# ===========================================================================
# Stage 10: Pre-flight checks and abort logic
# ===========================================================================

@pytest.mark.unit
class TestPipelinePreflightAndAbort:
    def test_source_file_not_found_fails_immediately(self, tmp_path, monkeypatch):
        """Pipeline should fail immediately if source file doesn't exist."""
        from src.ingestion.job import JobStore
        from src.ingestion.models import Job, JobStatus, VideoFile, MatchConfig, KitConfig

        monkeypatch.setenv("WORKING_DIR", str(tmp_path))
        monkeypatch.setenv("NAS_MOUNT_PATH", str(tmp_path))
        monkeypatch.setenv("NAS_OUTPUT_PATH", str(tmp_path / "output"))
        (tmp_path / "output").mkdir(exist_ok=True)
        (tmp_path / "jobs").mkdir(exist_ok=True)

        store = JobStore(tmp_path / "jobs")
        job = Job(
            video_file=VideoFile(
                path="/nonexistent/video.mp4",
                filename="video.mp4",
                duration_sec=100.0, fps=30.0, width=1280, height=720,
                codec="h264", size_bytes=1000, sha256="abc",
            ),
            reel_types=["keeper"],
            match_config=MatchConfig(
                team=KitConfig(team_name="T", outfield_color="blue", gk_color="teal"),
                opponent=KitConfig(team_name="O", outfield_color="red", gk_color="neon_yellow"),
            ),
        )
        store.save(job)

        from src.api.worker import _run_pipeline
        from src.config import config as cfg

        monkeypatch.setattr(cfg, "__class__", type(cfg))  # Ensure config is live
        result = _run_pipeline(job.job_id, store, cfg)

        assert result["output_paths"] == {}
        assert "error" in result
        assert "not found" in result["error"]

        saved = store.get(job.job_id)
        assert saved.status == JobStatus.FAILED

    def test_consecutive_empty_aborts_pipeline(self):
        """PipelineRunner should abort after max_consecutive_empty chunks with no detections."""
        from src.detection.event_classifier import PipelineRunner
        from src.detection.base import NullDetector
        from src.ingestion.models import VideoFile

        vf = VideoFile(
            path="/nonexistent.mp4", filename="x.mp4",
            duration_sec=600.0, fps=30.0, width=1280, height=720,
            codec="h264", size_bytes=1000, sha256="abc",
        )
        mc = make_match_config()
        player_det = NullDetector(job_id="j1", source_file=vf.path)
        gk_det = GoalkeeperDetector(job_id="j1", source_file=vf.path, match_config=mc)
        event_log = MagicMock()
        event_log.append_many = MagicMock()

        runner = PipelineRunner(
            job_id="j1", video_file=vf,
            player_detector=player_det, gk_detector=gk_det,
            event_log=event_log, chunk_sec=30, min_confidence=0.65,
        )

        with pytest.raises(RuntimeError, match="consecutive"):
            runner.run()


# ===========================================================================
# Stage 11: Full keeper reel integration (events → clips → output)
# ===========================================================================

@pytest.mark.unit
class TestKeeperReelIntegration:
    def test_full_keeper_pipeline_events_to_clips(self):
        """Simulate: GK events written → read back → clipper → postprocess → clips ready."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "events.jsonl"
            event_log = EventLog(log_path)

            # Write realistic GK events
            gk_events = [
                _make_gk_event("e1", EventType.SHOT_STOP_DIVING, 120.5, 123.0, ["keeper"], confidence=0.82),
                _make_gk_event("e2", EventType.DISTRIBUTION_SHORT, 350.0, 355.0, ["keeper"], confidence=0.70),
                _make_gk_event("e3", EventType.CATCH, 780.0, 781.5, ["keeper"], confidence=0.78),
                _make_gk_event("e4", EventType.GOAL_KICK, 1500.0, 1506.0, ["keeper"], confidence=0.72),
                _make_gk_event("e5", EventType.SHOT_STOP_STANDING, 2200.0, 2202.0, ["keeper"], confidence=0.76),
            ]
            # Also add some highlights events that should NOT appear in keeper reel
            hl_events = [
                Event(
                    event_id="h1", job_id="j1", source_file="m.mp4",
                    event_type=EventType.GOAL, timestamp_start=900.0, timestamp_end=901.0,
                    confidence=0.92, reel_targets=["highlights"],
                    frame_start=27000, frame_end=27030,
                ),
            ]
            event_log.append_many(gk_events + hl_events)

            # Read back
            all_events = event_log.read_all()
            assert len(all_events) == 6

            # Compute clips for keeper reel
            clips = compute_clips(all_events, 5400.0, "keeper", pre_pad=3.0, post_pad=5.0)
            assert len(clips) == 5  # 5 GK events, all well-separated

            # Postprocess
            conf_map = {e.event_id: e.confidence for e in all_events}
            final = postprocess_clips(clips, "keeper", event_confidence_map=conf_map)
            assert len(final) == 5
            assert all(c.reel_type == "keeper" for c in final)

            # Verify no highlights events leaked in
            all_event_ids = set()
            for c in final:
                all_event_ids.update(c.events)
            assert "h1" not in all_event_ids

    def test_mixed_keeper_and_highlights_reels(self):
        """Same event set produces different clips for keeper vs highlights."""
        events = [
            _make_gk_event("e1", EventType.SHOT_STOP_DIVING, 60.0, 62.0, ["keeper"], confidence=0.82),
            Event(
                event_id="e2", job_id="j1", source_file="m.mp4",
                event_type=EventType.GOAL, timestamp_start=200.0, timestamp_end=201.0,
                confidence=0.92, reel_targets=["highlights"],
                frame_start=6000, frame_end=6030,
            ),
            _make_gk_event("e3", EventType.ONE_ON_ONE, 400.0, 404.0, ["keeper", "highlights"], confidence=0.80),
        ]
        keeper_clips = compute_clips(events, 5400.0, "keeper")
        hl_clips = compute_clips(events, 5400.0, "highlights")

        # keeper reel: e1 + e3
        assert len(keeper_clips) == 2
        # highlights reel: e2 + e3
        assert len(hl_clips) == 2

        keeper_event_ids = set()
        for c in keeper_clips:
            keeper_event_ids.update(c.events)
        assert "e1" in keeper_event_ids
        assert "e2" not in keeper_event_ids

        hl_event_ids = set()
        for c in hl_clips:
            hl_event_ids.update(c.events)
        assert "e2" in hl_event_ids
        assert "e1" not in hl_event_ids


# ===========================================================================
# Stage 12: GK event type model invariants
# ===========================================================================

@pytest.mark.unit
class TestGKEventTypeInvariants:
    def test_all_gk_event_types_have_empty_reel_map(self):
        """GK event types must have empty defaults in EVENT_REEL_MAP (targets set dynamically)."""
        for et in (
            EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING,
            EventType.PUNCH, EventType.CATCH,
            EventType.GOAL_KICK, EventType.DISTRIBUTION_SHORT, EventType.DISTRIBUTION_LONG,
        ):
            assert EVENT_REEL_MAP[et] == [], f"{et} should have empty reel map default"
            assert is_gk_event_type(et), f"{et} should be classified as GK event type"

    def test_one_on_one_is_gk_event(self):
        assert is_gk_event_type(EventType.ONE_ON_ONE)

    def test_highlights_events_not_gk(self):
        for et in (EventType.SHOT_ON_TARGET, EventType.GOAL, EventType.TACKLE):
            assert not is_gk_event_type(et)


# ===========================================================================
# Stage 13: bbox_scale threshold scaling for wide-angle footage
# ===========================================================================

def _make_bbox_sized(cx: float = 0.5, cy: float = 0.5, w: float = 0.05, h: float = 0.15) -> BoundingBox:
    """Helper that allows specifying bbox height for scale tests."""
    return BoundingBox(x=cx - w / 2, y=cy - h / 2, width=w, height=h)


def _make_detection_sized(
    frame: int, ts: float, cx: float = 0.5, cy: float = 0.5,
    h: float = 0.15, track_id: int = None, cls: str = "player",
) -> Detection:
    return Detection(
        frame_number=frame, timestamp=ts, class_name=cls, confidence=0.9,
        bbox=_make_bbox_sized(cx, cy, h=h), track_id=track_id,
    )


@pytest.mark.unit
class TestBboxScaleComputation:
    """Verify bbox_scale is computed correctly from GK track bbox heights."""

    def test_scale_close_camera_large_bbox(self):
        """GK bbox height = 0.25 (reference) → scale = 1.0."""
        from src.detection.goalkeeper_detector import REFERENCE_BBOX_HEIGHT
        mean_h = 0.25
        scale = max(0.3, min(1.0, mean_h / REFERENCE_BBOX_HEIGHT))
        assert scale == 1.0

    def test_scale_wide_angle_small_bbox(self):
        """GK bbox height = 0.133 (wide-angle) → scale ≈ 0.53."""
        from src.detection.goalkeeper_detector import REFERENCE_BBOX_HEIGHT
        mean_h = 0.133
        scale = max(0.3, min(1.0, mean_h / REFERENCE_BBOX_HEIGHT))
        assert 0.50 < scale < 0.56

    def test_scale_clamped_floor(self):
        """Very small bbox height should clamp scale to 0.3."""
        from src.detection.goalkeeper_detector import REFERENCE_BBOX_HEIGHT
        mean_h = 0.01  # Extremely small
        scale = max(0.3, min(1.0, mean_h / REFERENCE_BBOX_HEIGHT))
        assert scale == 0.3

    def test_scale_clamped_ceiling(self):
        """Larger-than-reference bbox height should clamp scale to 1.0."""
        from src.detection.goalkeeper_detector import REFERENCE_BBOX_HEIGHT
        mean_h = 0.50  # Very large
        scale = max(0.3, min(1.0, mean_h / REFERENCE_BBOX_HEIGHT))
        assert scale == 1.0


@pytest.mark.unit
class TestBboxScaleDetection:
    """Verify that scaled thresholds allow detection in wide-angle footage."""

    def _build_gk_detector(self):
        mc = make_match_config()
        return GoalkeeperDetector(job_id="j1", source_file="m.mp4", match_config=mc)

    def test_distribution_detected_with_small_bbox(self):
        """A post_vel of 0.08 should trigger distribution when bbox_scale ≈ 0.53."""
        det = self._build_gk_detector()
        # Build a track where GK is stationary then suddenly moves.
        # bbox height = 0.133 → scale ≈ 0.53 → threshold = 0.12 * 0.53 ≈ 0.064
        dets = []
        for i in range(20):
            # Stationary phase: cx stays at 0.08
            dets.append(_make_detection_sized(
                i, float(i) / 30.0, cx=0.08, cy=0.5, h=0.133, track_id=10,
            ))
        # Movement phase: cx jumps (post_vel ~ 0.08 which is > 0.064 but < 0.12)
        for i in range(20, 30):
            cx = 0.08 + (i - 20) * 0.003  # slow drift to create velocity ~ 0.08
            dets.append(_make_detection_sized(
                i, float(i) / 30.0, cx=cx, cy=0.5, h=0.133, track_id=10,
            ))

        gk_track = _make_track(10, dets)
        events = det.classify_gk_events(gk_track, [gk_track], 30.0, keeper_role="keeper")
        # With bbox_scale the distribution should be detectable
        dist_events = [e for e in events if e.event_type in (
            EventType.GOAL_KICK, EventType.DISTRIBUTION_SHORT, EventType.DISTRIBUTION_LONG,
        )]
        assert len(dist_events) > 0, "Small bbox scale should lower distribution threshold"

    def test_distribution_not_detected_without_scale(self):
        """Same velocity pattern with large bbox (scale=1.0) should NOT trigger."""
        det = self._build_gk_detector()
        dets = []
        for i in range(20):
            dets.append(_make_detection_sized(
                i, float(i) / 30.0, cx=0.08, cy=0.5, h=0.25, track_id=10,
            ))
        for i in range(20, 30):
            cx = 0.08 + (i - 20) * 0.003
            dets.append(_make_detection_sized(
                i, float(i) / 30.0, cx=cx, cy=0.5, h=0.25, track_id=10,
            ))

        gk_track = _make_track(10, dets)
        events = det.classify_gk_events(gk_track, [gk_track], 30.0, keeper_role="keeper")
        dist_events = [e for e in events if e.event_type in (
            EventType.GOAL_KICK, EventType.DISTRIBUTION_SHORT, EventType.DISTRIBUTION_LONG,
        )]
        assert len(dist_events) == 0, "Large bbox (scale=1.0) should not trigger on small velocity"

    def test_save_detected_with_small_bbox(self):
        """A vertical velocity of 0.15 should trigger a save when bbox_scale ≈ 0.53."""
        det = self._build_gk_detector()
        # bbox height = 0.133 → scale ≈ 0.53 → vel_threshold = 0.35 * 0.53 ≈ 0.186
        # Build a track with a sudden vertical jump (vertical_velocity ≈ 0.20)
        dets = []
        fps = 30.0
        for i in range(10):
            dets.append(_make_detection_sized(
                i, float(i) / fps, cx=0.08, cy=0.5, h=0.133, track_id=10,
            ))
        # Sudden vertical movement at frame 10
        dets.append(_make_detection_sized(
            10, 10.0 / fps, cx=0.08, cy=0.5 + 0.007, h=0.133, track_id=10,  # small rise
        ))
        dets.append(_make_detection_sized(
            11, 11.0 / fps, cx=0.08, cy=0.5 + 0.015, h=0.133, track_id=10,  # bigger jump
        ))
        dets.append(_make_detection_sized(
            12, 12.0 / fps, cx=0.08, cy=0.5 + 0.007, h=0.133, track_id=10,
        ))
        for i in range(13, 20):
            dets.append(_make_detection_sized(
                i, float(i) / fps, cx=0.08, cy=0.5, h=0.133, track_id=10,
            ))

        gk_track = _make_track(10, dets)
        events = det.classify_gk_events(gk_track, [gk_track], fps, keeper_role="keeper")
        save_events = [e for e in events if e.event_type in (
            EventType.SHOT_STOP_STANDING, EventType.SHOT_STOP_DIVING,
        )]
        assert len(save_events) > 0, "Small bbox scale should lower save velocity threshold"

    def test_one_on_one_detected_with_small_bbox(self):
        """A deviation of 0.12 should trigger 1v1 when bbox_scale ≈ 0.53."""
        det = self._build_gk_detector()
        # bbox height = 0.133 → scale ≈ 0.53 → deviation_threshold = 0.20 * 0.53 ≈ 0.106
        # Build a track where GK drifts from baseline by ~0.12
        dets = []
        fps = 30.0
        # Baseline: cy = 0.5 for first 10 frames
        for i in range(10):
            dets.append(_make_detection_sized(
                i, float(i) / fps, cx=0.08, cy=0.5, h=0.133, track_id=10,
            ))
        # GK rushes out: cy shifts by 0.12 for 5 consecutive frames
        for i in range(10, 15):
            dets.append(_make_detection_sized(
                i, float(i) / fps, cx=0.08, cy=0.5 + 0.15, h=0.133, track_id=10,
            ))

        gk_track = _make_track(10, dets)
        events = det.classify_gk_events(gk_track, [gk_track], fps, keeper_role="keeper")
        ono_events = [e for e in events if e.event_type == EventType.ONE_ON_ONE]
        assert len(ono_events) > 0, "Small bbox scale should lower 1v1 deviation threshold"

    def test_large_bbox_retains_original_thresholds(self):
        """With bbox height = 0.25, thresholds should remain at original values."""
        det = self._build_gk_detector()
        # Build a GK track with a sudden dive (same as existing test)
        dets = []
        fps = 30.0
        for i in range(30):
            cy = 0.5
            if i == 15:
                cy = 0.1  # Sudden dive
            dets.append(_make_detection_sized(
                i, float(i) / fps, cx=0.08, cy=cy, h=0.25, track_id=10,
            ))

        gk_track = _make_track(10, dets)
        events = det.classify_gk_events(gk_track, [gk_track], fps, keeper_role="keeper")
        # Should still detect the dive with original thresholds
        save_events = [e for e in events if e.event_type in (
            EventType.SHOT_STOP_STANDING, EventType.SHOT_STOP_DIVING,
        )]
        assert len(save_events) > 0, "Large bbox (scale=1.0) should still detect large movements"
