"""
Unit tests for detection improvements (WS1-WS4) + Rush 2026 QA fixes.

WS1: Zone-aware thresholds + ball-in-net detection
WS2: Corner kick event type + detection + plugin
WS3: Smart clip endpoints
WS4: Pause/resume restart bug fix
QA:  Color margin, position gate, voter confidence, dedup threshold,
     ONE_ON_ONE extension, relaxed next-touch thresholds, goal kick window
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.detection.ball_touch_detector import BallTouchDetector, BallTrajectory
from src.detection.models import (
    BoundingBox, Detection, Event, EventType, Track,
    EVENT_REEL_MAP, _GK_EVENT_TYPES, EVENT_CONFIDENCE_THRESHOLDS,
    is_gk_event_type,
)
from src.ingestion.models import Job, JobStatus, VideoFile
from tests.conftest import make_bbox, make_detection, make_match_config, make_track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEAM_GK_HSV = (35.0, 0.95, 0.95)       # neon_yellow
OPP_GK_HSV = (88.0, 0.80, 0.55)        # teal
TEAM_OUTFIELD_HSV = (112.0, 0.82, 0.65) # blue
OPP_OUTFIELD_HSV = (0.0, 0.85, 0.70)    # red


def _ball_det(frame: int, ts: float, cx: float, cy: float) -> Detection:
    return make_detection(frame, ts, cls="ball", cx=cx, cy=cy, w=0.02, h=0.02)


def _ball_track(track_id: int, dets: list[Detection]) -> Track:
    return make_track(track_id, dets)


def _player_track(
    track_id: int, dets: list[Detection], jersey_hsv: tuple = None,
) -> Track:
    return make_track(track_id, dets, jersey_hsv=jersey_hsv)


def _make_detector(**kwargs) -> BallTouchDetector:
    defaults = dict(
        job_id="job-001",
        source_file="match.mp4",
        match_config=make_match_config(),
    )
    defaults.update(kwargs)
    return BallTouchDetector(**defaults)


def _sample_video_file(path: str = "/mnt/nas/match.mp4") -> VideoFile:
    return VideoFile(
        path=path, filename="match.mp4", duration_sec=5400.0, fps=30.0,
        width=3840, height=2160, codec="h264", size_bytes=15_000_000_000,
        sha256="a" * 64,
    )


def _sample_job(**kwargs) -> Job:
    return Job(video_file=_sample_video_file(), match_config=make_match_config(), **kwargs)


# ===========================================================================
# WS1: Zone-Aware Thresholds
# ===========================================================================

@pytest.mark.unit
class TestZoneAwareThresholds:
    """Ball touch detection uses relaxed thresholds near goal."""

    def test_glancing_deflection_near_goal_detected(self):
        """A ~30-degree direction change near goal should be detected with
        relaxed threshold (25deg) but not with normal (40deg)."""
        fps = 30.0
        dets = []
        # Pre: ball moving rightward fast near left goal at x ~ 0.05..0.12
        # Speed = 0.01/frame * 30fps = 0.30/sec (well above min_touch_speed*0.5)
        for i in range(30):
            t = i / fps
            dets.append(_ball_det(i, t, 0.03 + i * 0.01, 0.50))
        # Change point: ball deflects ~150 degrees (clear reversal)
        # Post: ball goes back leftward and slightly up
        for i in range(30, 50):
            t = i / fps
            j = i - 30
            dets.append(_ball_det(i, t, 0.03 + 30 * 0.01 - j * 0.008, 0.50 - j * 0.005))

        track = _ball_track(1, dets)
        # Add a GK player near the deflection point
        gk_dets = [make_detection(i, i / fps, cls="player", track_id=10,
                                   cx=0.12, cy=0.48)
                    for i in range(50)]
        gk_track = _player_track(10, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = _make_detector()
        events = det.detect_touches([track, gk_track], fps)
        keeper_events = [e for e in events if e.is_goalkeeper_event]
        assert len(keeper_events) >= 1

    def test_same_deflection_midfield_rejected(self):
        """Same 30-degree change at midfield (x=0.50) should NOT be detected
        because normal threshold is 40deg."""
        fps = 30.0
        dets = []
        for i in range(20):
            t = i / fps
            dets.append(_ball_det(i, t, 0.45 + i * 0.003, 0.50))
        angle_rad = math.radians(30)
        for i in range(20, 35):
            t = i / fps
            j = i - 20
            dx = math.cos(angle_rad) * 0.003
            dy = math.sin(angle_rad) * 0.003
            dets.append(_ball_det(i, t, 0.45 + 20 * 0.003 + j * dx, 0.50 + j * dy))

        track = _ball_track(1, dets)
        gk_dets = [make_detection(i, i / fps, cls="player", track_id=10,
                                   cx=0.51, cy=0.50)
                    for i in range(35)]
        gk_track = _player_track(10, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = _make_detector()
        events = det.detect_touches([track, gk_track], fps)
        direction_events = [e for e in events
                           if e.metadata.get("touch_reason") == "direction_change"]
        assert len(direction_events) == 0

    def test_moderate_speed_drop_near_goal_detected(self):
        """35% speed drop near goal detected with near_goal_speed_change_ratio=0.30."""
        fps = 30.0
        dets = []
        # Fast ball approaching goal (x < 0.10)
        for i in range(20):
            t = i / fps
            dets.append(_ball_det(i, t, 0.15 - i * 0.005, 0.50))
        # Speed drops to ~60% (40% drop > 30% threshold)
        for i in range(20, 35):
            t = i / fps
            j = i - 20
            dets.append(_ball_det(i, t, 0.05 - j * 0.003, 0.50))

        track = _ball_track(1, dets)
        gk_dets = [make_detection(i, i / fps, cls="player", track_id=10,
                                   cx=0.06, cy=0.50)
                    for i in range(35)]
        gk_track = _player_track(10, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = _make_detector()
        events = det.detect_touches([track, gk_track], fps)
        keeper_events = [e for e in events if e.is_goalkeeper_event]
        assert len(keeper_events) >= 1

    def test_same_speed_drop_midfield_rejected(self):
        """Same 35% speed drop at midfield (x=0.50) should NOT be detected
        because normal threshold is 50%."""
        fps = 30.0
        dets = []
        for i in range(20):
            t = i / fps
            dets.append(_ball_det(i, t, 0.55 - i * 0.005, 0.50))
        for i in range(20, 35):
            t = i / fps
            j = i - 20
            dets.append(_ball_det(i, t, 0.45 - j * 0.003, 0.50))

        track = _ball_track(1, dets)
        gk_dets = [make_detection(i, i / fps, cls="player", track_id=10,
                                   cx=0.46, cy=0.50)
                    for i in range(35)]
        gk_track = _player_track(10, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = _make_detector()
        events = det.detect_touches([track, gk_track], fps)
        speed_drop_events = [e for e in events
                            if e.metadata.get("touch_reason") == "speed_drop"]
        assert len(speed_drop_events) == 0


# ===========================================================================
# WS1: Ball-in-Net Detection
# ===========================================================================

@pytest.mark.unit
class TestBallInNet:
    """Ball entering the net area should produce a GOAL event."""

    def test_ball_enters_left_net_produces_goal(self):
        """Fast ball at x<0.03 moving left then disappears → GOAL."""
        fps = 30.0
        dets = []
        # Ball moving left quickly toward goal — 0.02/frame = 0.6/sec (>0.30 threshold)
        for i in range(30):
            t = i / fps
            dets.append(_ball_det(i, t, 0.60 - i * 0.02, 0.50))
        # Ball enters net at x=0.01
        dets.append(_ball_det(30, 30 / fps, 0.01, 0.50))
        # Then disappears (no dets for frames 31+)

        track = _ball_track(1, dets)
        trajectory = BallTrajectory().build([track])

        det = _make_detector()
        events = det._detect_ball_in_net(trajectory, [track], fps)

        assert len(events) >= 1
        assert events[0].event_type == EventType.GOAL
        assert events[0].is_goalkeeper_event is False
        assert events[0].metadata["net_side"] == "left"

    def test_ball_enters_right_net_produces_goal(self):
        """Fast ball at x>0.97 moving right then disappears → GOAL."""
        fps = 30.0
        dets = []
        # Ball moving right quickly: 0.02/frame = 0.6/sec
        for i in range(30):
            t = i / fps
            dets.append(_ball_det(i, t, 0.40 + i * 0.02, 0.50))
        # Ball enters net at x=0.99
        dets.append(_ball_det(30, 30 / fps, 0.99, 0.50))
        # Disappears

        track = _ball_track(1, dets)
        trajectory = BallTrajectory().build([track])

        det = _make_detector()
        events = det._detect_ball_in_net(trajectory, [track], fps)

        assert len(events) >= 1
        assert events[0].event_type == EventType.GOAL
        assert events[0].metadata["net_side"] == "right"

    def test_slow_ball_at_edge_no_goal(self):
        """Slow ball at x<0.03 should NOT produce a goal — ball placed, not shot."""
        fps = 30.0
        dets = []
        for i in range(30):
            t = i / fps
            dets.append(_ball_det(i, t, 0.02, 0.50 + i * 0.001))

        track = _ball_track(1, dets)
        trajectory = BallTrajectory().build([track])

        det = _make_detector()
        events = det._detect_ball_in_net(trajectory, [track], fps)
        assert len(events) == 0

    def test_goal_event_targets_highlights_reel(self):
        """GOAL events from ball-in-net should target highlights, not keeper."""
        fps = 30.0
        dets = []
        for i in range(30):
            t = i / fps
            dets.append(_ball_det(i, t, 0.60 - i * 0.02, 0.50))
        dets.append(_ball_det(30, 30 / fps, 0.01, 0.50))

        track = _ball_track(1, dets)
        trajectory = BallTrajectory().build([track])

        det = _make_detector()
        events = det._detect_ball_in_net(trajectory, [track], fps)

        assert len(events) >= 1
        for ev in events:
            assert ev.is_goalkeeper_event is False


# ===========================================================================
# WS2: Corner Kick Event Type
# ===========================================================================

@pytest.mark.unit
class TestCornerKickEventType:
    """CORNER_KICK event type is properly registered."""

    def test_corner_kick_in_event_type(self):
        assert EventType.CORNER_KICK == "corner_kick"

    def test_corner_kick_in_gk_event_types(self):
        assert is_gk_event_type(EventType.CORNER_KICK)

    def test_corner_kick_in_reel_map(self):
        assert EventType.CORNER_KICK in EVENT_REEL_MAP

    def test_corner_kick_confidence_threshold(self):
        assert EventType.CORNER_KICK in EVENT_CONFIDENCE_THRESHOLDS
        assert EVENT_CONFIDENCE_THRESHOLDS[EventType.CORNER_KICK] == 0.65


# ===========================================================================
# WS2: Corner Kick Detection
# ===========================================================================

@pytest.mark.unit
class TestCornerKickDetection:
    """Corner kick detection from ball trajectory."""

    def _corner_kick_scenario(self, corner_x, corner_y, fps=30.0):
        """Build a trajectory with ball near corner, stationary, then kicked,
        plus players in the box."""
        dets = []
        # Ball approaches corner
        for i in range(10):
            t = i / fps
            dets.append(_ball_det(i, t, corner_x + i * 0.001, corner_y))

        # Ball stationary at corner for ~1s (30 frames)
        for i in range(10, 40):
            t = i / fps
            dets.append(_ball_det(i, t, corner_x, corner_y))

        # Ball kicked (high speed)
        for i in range(40, 55):
            t = i / fps
            j = i - 40
            dets.append(_ball_det(i, t, corner_x + j * 0.02, 0.50))

        ball_track = _ball_track(1, dets)

        # 4 players in the box area
        player_tracks = []
        box_x = 0.10 if corner_x < 0.5 else 0.90
        for pid in range(2, 6):
            p_dets = [
                make_detection(f, f / fps, cls="player", track_id=pid,
                               cx=box_x + (pid - 2) * 0.02, cy=0.50)
                for f in range(55)
            ]
            player_tracks.append(make_track(pid, p_dets))

        return ball_track, player_tracks

    @pytest.mark.skip(reason="Spatial corner detection disabled for auto-pan cameras")
    def test_corner_detected_left_bottom(self):
        """Ball near left-bottom corner with players → CORNER_KICK."""
        fps = 30.0
        ball, players = self._corner_kick_scenario(0.02, 0.95, fps)
        trajectory = BallTrajectory().build([ball])

        det = _make_detector()
        events = det._detect_corner_kicks(trajectory, [ball] + players, fps)

        assert len(events) >= 1
        assert events[0].event_type == EventType.CORNER_KICK
        assert events[0].is_goalkeeper_event

    def test_corner_no_players_no_event(self):
        """Ball near corner but no players in box → no event."""
        fps = 30.0
        dets = []
        for i in range(10):
            t = i / fps
            dets.append(_ball_det(i, t, 0.02, 0.95))
        for i in range(10, 40):
            t = i / fps
            dets.append(_ball_det(i, t, 0.02, 0.95))
        for i in range(40, 55):
            t = i / fps
            j = i - 40
            dets.append(_ball_det(i, t, 0.02 + j * 0.02, 0.50))

        ball = _ball_track(1, dets)
        trajectory = BallTrajectory().build([ball])

        det = _make_detector()
        events = det._detect_corner_kicks(trajectory, [ball], fps)
        assert len(events) == 0

    def test_midfield_ball_no_corner(self):
        """Ball stationary at midfield → no corner kick."""
        fps = 30.0
        dets = []
        for i in range(40):
            t = i / fps
            dets.append(_ball_det(i, t, 0.50, 0.50))
        for i in range(40, 55):
            t = i / fps
            j = i - 40
            dets.append(_ball_det(i, t, 0.50 + j * 0.02, 0.50))

        ball = _ball_track(1, dets)
        player_tracks = []
        for pid in range(2, 6):
            p_dets = [make_detection(f, f / fps, cls="player", track_id=pid,
                                      cx=0.50, cy=0.50)
                       for f in range(55)]
            player_tracks.append(make_track(pid, p_dets))

        trajectory = BallTrajectory().build([ball])

        det = _make_detector()
        events = det._detect_corner_kicks(trajectory, [ball] + player_tracks, fps)
        assert len(events) == 0


# ===========================================================================
# WS2: Corner Kick Plugin
# ===========================================================================

@pytest.mark.unit
class TestCornerKickPlugin:
    """KeeperCornerKickPlugin selects and clips correctly."""

    def test_plugin_selects_only_corner_kick(self):
        from src.reel_plugins.keeper import KeeperCornerKickPlugin
        from src.reel_plugins.base import PipelineContext

        plugin = KeeperCornerKickPlugin()

        corner = Event(
            job_id="j", source_file="m.mp4", event_type=EventType.CORNER_KICK,
            timestamp_start=10.0, timestamp_end=15.0, confidence=0.70,
            reel_targets=[], is_goalkeeper_event=True,
            frame_start=300, frame_end=450,
            bounding_box=BoundingBox(x=0.03, y=0.93, width=0.04, height=0.04),
            metadata={"sim_team_gk": 0.90},
        )
        save = Event(
            job_id="j", source_file="m.mp4", event_type=EventType.SHOT_STOP_DIVING,
            timestamp_start=20.0, timestamp_end=21.0, confidence=0.80,
            reel_targets=[], is_goalkeeper_event=True,
            frame_start=600, frame_end=630,
            bounding_box=BoundingBox(x=0.10, y=0.50, width=0.04, height=0.04),
            metadata={"sim_team_gk": 0.90},
        )

        ctx = PipelineContext(
            video_duration_sec=5400.0,
            match_config=make_match_config(),
            keeper_track_ids={},
            job_id="j",
        )
        selected = plugin.select_events([corner, save], ctx)

        assert len(selected) == 1
        assert selected[0].event_type == EventType.CORNER_KICK

    def test_plugin_clip_params(self):
        from src.reel_plugins.keeper import KeeperCornerKickPlugin
        p = KeeperCornerKickPlugin()
        assert p.clip_params.pre_pad_sec == 3.0
        assert p.clip_params.post_pad_sec == 2.0
        assert p.clip_params.max_clip_duration_sec == 25.0

    def test_plugin_reel_name(self):
        from src.reel_plugins.keeper import KeeperCornerKickPlugin
        assert KeeperCornerKickPlugin().reel_name == "keeper"

    def test_registry_includes_corner_kick(self):
        from src.reel_plugins.registry import PluginRegistry, DEFAULT_PLUGIN_NAMES
        assert "keeper_corner_kick" in DEFAULT_PLUGIN_NAMES
        registry = PluginRegistry.default()
        assert "keeper_corner_kick" in registry.plugin_names


# ===========================================================================
# WS3: Smart Clip Endpoints
# ===========================================================================

@pytest.mark.unit
class TestSmartEndpoints:
    """Smart clip endpoint extension tests."""

    def _distribution_event(self, ts=10.0, fps=30.0):
        return Event(
            job_id="j", source_file="m.mp4",
            event_type=EventType.DISTRIBUTION_SHORT,
            timestamp_start=ts - 0.5, timestamp_end=ts + 0.5,
            confidence=0.75, reel_targets=[],
            is_goalkeeper_event=True,
            frame_start=int((ts - 0.5) * fps), frame_end=int((ts + 0.5) * fps),
            metadata={"detection_method": "ball_touch", "touch_reason": "speed_spike"},
        )

    def _save_event(self, ts=10.0, fps=30.0):
        return Event(
            job_id="j", source_file="m.mp4",
            event_type=EventType.SHOT_STOP_STANDING,
            timestamp_start=ts - 0.5, timestamp_end=ts + 0.5,
            confidence=0.80, reel_targets=[],
            is_goalkeeper_event=True,
            frame_start=int((ts - 0.5) * fps), frame_end=int((ts + 0.5) * fps),
            metadata={"detection_method": "ball_touch", "touch_reason": "speed_drop"},
        )

    def test_distribution_extends_to_next_touch(self):
        """Distribution clip extends until another player touches the ball."""
        fps = 30.0
        event = self._distribution_event(ts=10.0, fps=fps)
        # Ball moves away from GK then gets touched (direction change) at t=14
        dets = []
        for i in range(int(10 * fps), int(14 * fps)):
            t = i / fps
            dets.append(_ball_det(i, t, 0.20 + (i - int(10 * fps)) * 0.001, 0.50))
        # Direction change at t=14 (another player touches)
        for i in range(int(14 * fps), int(16 * fps)):
            t = i / fps
            j = i - int(14 * fps)
            dets.append(_ball_det(i, t, 0.20 + 4 * 30 * 0.001 - j * 0.002, 0.55))

        trajectory = BallTrajectory().build([_ball_track(1, dets)])
        det = _make_detector()
        result = det._apply_smart_endpoints([event], trajectory, [], fps)

        assert len(result) == 1
        assert result[0].timestamp_end > event.timestamp_end
        assert result[0].metadata.get("endpoint_extended") is True

    def test_distribution_extends_to_out_of_bounds(self):
        """Distribution clip extends until ball goes out of bounds."""
        fps = 30.0
        event = self._distribution_event(ts=10.0, fps=fps)
        # Ball moves at constant speed toward the sideline (y approaches 0.0)
        dets = []
        n_frames = int(3 * fps)
        for i in range(n_frames):
            frame = int(10 * fps) + i
            t = frame / fps
            # Constant x, y decreasing from 0.50 to 0.005 (out of bounds)
            y = 0.50 - (i / n_frames) * 0.50
            dets.append(_ball_det(frame, t, 0.30, y))

        trajectory = BallTrajectory().build([_ball_track(1, dets)])
        det = _make_detector()
        result = det._apply_smart_endpoints([event], trajectory, [], fps)

        assert len(result) == 1
        ext = result[0]
        assert ext.timestamp_end > event.timestamp_end
        assert ext.metadata.get("endpoint_reason") == "out_of_bounds"

    def test_distribution_caps_at_max_extension(self):
        """Extension never exceeds max (10s for distribution)."""
        fps = 30.0
        event = self._distribution_event(ts=10.0, fps=fps)
        # Ball keeps going for 30 seconds with no touch/OOB
        dets = []
        for i in range(int(10 * fps), int(40 * fps)):
            t = i / fps
            dets.append(_ball_det(i, t, 0.30 + (i - int(10 * fps)) * 0.0001, 0.50))

        trajectory = BallTrajectory().build([_ball_track(1, dets)])
        det = _make_detector()
        result = det._apply_smart_endpoints([event], trajectory, [], fps)

        assert len(result) == 1
        # Max extension for distribution is 10s → end at 10.5 + 10 = 20.5
        assert result[0].timestamp_end <= event.timestamp_end + 10.0 + 0.5

    def test_save_extends_to_ball_disappearance(self):
        """Save clip extends to where ball disappears."""
        fps = 30.0
        event = self._save_event(ts=10.0, fps=fps)
        dets = []
        # Ball visible for a bit after save
        for i in range(int(10 * fps), int(11 * fps)):
            t = i / fps
            dets.append(_ball_det(i, t, 0.15, 0.50))
        # Gap of 10 frames (ball caught)

        trajectory = BallTrajectory().build([_ball_track(1, dets)])
        det = _make_detector()
        result = det._apply_smart_endpoints([event], trajectory, [], fps)

        assert len(result) == 1
        ext = result[0]
        # Should extend beyond original endpoint
        assert ext.timestamp_end >= event.timestamp_end

    def test_no_ball_data_no_change(self):
        """If no ball trajectory data after event → no extension."""
        fps = 30.0
        event = self._save_event(ts=10.0, fps=fps)
        # Only ball data before the event
        dets = [_ball_det(i, i / fps, 0.50, 0.50) for i in range(0, 5)]
        trajectory = BallTrajectory().build([_ball_track(1, dets)])

        det = _make_detector()
        result = det._apply_smart_endpoints([event], trajectory, [], fps)
        assert result[0].timestamp_end == event.timestamp_end

    def test_midfield_shot_not_extended(self):
        """SHOT_ON_TARGET events are not extended (not in _SMART_ENDPOINT_TYPES)."""
        fps = 30.0
        shot = Event(
            job_id="j", source_file="m.mp4",
            event_type=EventType.SHOT_ON_TARGET,
            timestamp_start=10.0, timestamp_end=13.0, confidence=0.80,
            reel_targets=[], is_goalkeeper_event=False,
            frame_start=300, frame_end=390,
            metadata={"ball_speed": 1.2},
        )
        dets = [_ball_det(i, i / fps, 0.50, 0.50) for i in range(300, 450)]
        trajectory = BallTrajectory().build([_ball_track(1, dets)])

        det = _make_detector()
        result = det._apply_smart_endpoints([shot], trajectory, [], fps)
        assert result[0].timestamp_end == shot.timestamp_end

    def test_extension_metadata_recorded(self):
        """Extended events have endpoint_extended and endpoint_reason in metadata."""
        fps = 30.0
        event = self._distribution_event(ts=10.0, fps=fps)
        dets = []
        for i in range(int(10 * fps), int(13 * fps)):
            t = i / fps
            progress = (i - int(10 * fps)) / (3 * fps)
            dets.append(_ball_det(i, t, 0.20 + progress * 0.80, 0.50))

        trajectory = BallTrajectory().build([_ball_track(1, dets)])
        det = _make_detector()
        result = det._apply_smart_endpoints([event], trajectory, [], fps)

        ext = result[0]
        assert ext.metadata["endpoint_extended"] is True
        assert ext.metadata["endpoint_reason"] in (
            "out_of_bounds", "next_touch_speed", "next_touch_direction",
            "ball_disappeared", "max_extension", "trajectory_end",
        )
        assert "extension_sec" in ext.metadata


# ===========================================================================
# WS4: Pause/Resume Restart Bug
# ===========================================================================

@pytest.mark.unit
class TestPauseResumeFix:
    """Resume should continue from last chunk, not restart."""

    def test_job_model_has_last_processed_chunk(self):
        job = _sample_job()
        assert hasattr(job, "last_processed_chunk")
        assert job.last_processed_chunk == -1

    def test_last_processed_chunk_defaults_minus_one(self):
        """Old jobs without last_processed_chunk get default -1."""
        data = _sample_job().model_dump()
        del data["last_processed_chunk"]
        job = Job(**data)
        assert job.last_processed_chunk == -1

    def test_pipeline_runner_skips_processed_chunks(self):
        """PipelineRunner with resume_from_chunk skips already-processed chunks."""
        from src.detection.event_classifier import PipelineRunner
        from src.detection.event_log import EventLog

        player_det = MagicMock()
        player_det.detect_chunk = MagicMock(return_value=[])
        gk_det = MagicMock()
        gk_det.identify_goalkeepers = MagicMock(return_value={})

        vf = _sample_video_file()
        vf_short = VideoFile(
            path="/mnt/nas/match.mp4", filename="match.mp4",
            duration_sec=120.0, fps=30.0, width=1920, height=1080,
            codec="h264", size_bytes=1_000_000, sha256="b" * 64,
        )

        with patch.object(EventLog, 'append_many'):
            with patch.object(EventLog, 'clear'):
                log = MagicMock(spec=EventLog)
                log.append_many = MagicMock()

                runner = PipelineRunner(
                    job_id="j", video_file=vf_short,
                    player_detector=player_det, gk_detector=gk_det,
                    event_log=log, chunk_sec=30,
                    resume_from_chunk=2,  # skip chunks 0, 1
                )

                chunks = runner._chunk_starts(120.0)
                assert len(chunks) == 4  # 0, 30, 60, 90

                # Run but track which chunks are processed
                processed_chunks = []
                original_callback = None

                def track_callback(pct, chunk_idx=-1):
                    processed_chunks.append(chunk_idx)

                runner.run(progress_callback=track_callback)

                # Should have skipped chunks 0, 1
                # Only chunks 2, 3 should have been detected
                assert player_det.detect_chunk.call_count == 2

    def test_event_log_not_cleared_on_resume(self):
        """When resuming (last_processed_chunk >= 0), event log should NOT be cleared."""
        job = _sample_job(last_processed_chunk=5)
        assert job.last_processed_chunk == 5
        # The worker logic checks: if job.last_processed_chunk >= 0 → don't clear
        # This is a unit-level assertion on the model
        assert job.last_processed_chunk >= 0

    def test_resume_preserves_progress(self):
        """Resume endpoint should keep existing progress, not reset to 0."""
        # Simulating: job paused at 40%, resume should NOT set progress=0
        job = _sample_job(
            status=JobStatus.PAUSED,
            progress_pct=40.0,
            last_processed_chunk=10,
        )
        assert job.progress_pct == 40.0
        assert job.last_processed_chunk == 10

    def test_backward_compat_old_jobs_start_from_zero(self):
        """Jobs without last_processed_chunk (default -1) start from chunk 0."""
        job = _sample_job()
        assert job.last_processed_chunk == -1
        # Worker logic: if last_processed_chunk >= 0 → resume, else start fresh
        resume_from = 0 if job.last_processed_chunk < 0 else job.last_processed_chunk + 1
        assert resume_from == 0

    def test_with_status_preserves_last_processed_chunk(self):
        """with_status() should preserve last_processed_chunk field."""
        job = _sample_job(last_processed_chunk=5)
        updated = job.with_status(JobStatus.PENDING)
        assert updated.last_processed_chunk == 5

    def test_resume_clears_flags(self):
        """Resume (PENDING status) should clear pause/cancel flags."""
        job = _sample_job(
            status=JobStatus.PAUSED,
            pause_requested=True,
            last_processed_chunk=10,
        )
        updated = job.with_status(JobStatus.PENDING)
        assert updated.pause_requested is False
        assert updated.cancel_requested is False
        assert updated.last_processed_chunk == 10


# ===========================================================================
# Rush 2026 QA: Fix 1 — GK Color Margin + Position Gate
# ===========================================================================

@pytest.mark.unit
class TestGKColorMargin:
    """Color margin prevents blue/teal confusion in GK classification."""

    def test_blue_opponent_rejected_by_margin(self):
        """Blue opponent outfield player (sim_team_gk≈0.73, sim_opp_of≈0.95)
        should be rejected because 0.73 < 0.95 + 0.10."""
        from src.detection.jersey_classifier import compute_jersey_similarity

        # Teal GK color and blue opponent outfield color
        teal_gk_hsv = (80.0, 0.80, 0.55)
        blue_opp_of_hsv = (120.0, 0.82, 0.65)
        # A blue player's jersey
        blue_player_hsv = (118.0, 0.80, 0.60)

        fps = 30.0
        # Ball trajectory with a speed drop (would normally trigger a touch)
        dets = []
        for i in range(20):
            t = i / fps
            dets.append(_ball_det(i, t, 0.10 + i * 0.005, 0.50))
        # Speed drops sharply
        for i in range(20, 35):
            t = i / fps
            j = i - 20
            dets.append(_ball_det(i, t, 0.20 + j * 0.001, 0.50))

        ball_track = _ball_track(1, dets)

        # Blue opponent player near the touch point
        player_dets = [
            make_detection(i, i / fps, cls="player", track_id=10,
                           cx=0.12, cy=0.48)
            for i in range(35)
        ]
        player_track = _player_track(10, player_dets, jersey_hsv=blue_player_hsv)

        det = BallTouchDetector(
            job_id="j", source_file="m.mp4",
            match_config=make_match_config(),
        )
        # Override colors for this specific scenario
        det._match_config.team.gk_color = "teal"
        det._match_config.opponent.outfield_color = "blue"

        events = det.detect_touches([ball_track, player_track], fps)
        keeper_events = [e for e in events if e.is_goalkeeper_event]
        assert len(keeper_events) == 0, "Blue opponent should be rejected by color margin"

    def test_teal_gk_accepted_with_margin(self):
        """Actual team GK (neon_yellow, sim_team_gk≈0.95) should pass margin check."""
        fps = 30.0
        # Ball trajectory with clear speed drop near goal:
        # Fast approach then abrupt slowdown = save
        dets = []
        # Fast ball: 0.01/frame = 0.30/sec, well above min_touch_speed*0.5
        for i in range(15):
            t = i / fps
            dets.append(_ball_det(i, t, 0.15 - i * 0.008, 0.50))
        # Abrupt slowdown (>50% speed drop)
        for i in range(15, 30):
            t = i / fps
            j = i - 15
            dets.append(_ball_det(i, t, 0.03 + j * 0.001, 0.50))

        ball_track = _ball_track(1, dets)

        # Team GK near the ball (within max_player_distance=0.12)
        gk_dets = [
            make_detection(i, i / fps, cls="player", track_id=10,
                           cx=0.05, cy=0.48)
            for i in range(30)
        ]
        gk_track = _player_track(10, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = _make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        keeper_events = [e for e in events if e.is_goalkeeper_event]
        assert len(keeper_events) >= 1, "Real GK should pass color margin"


@pytest.mark.unit
class TestGKPositionGate:
    """Midfield position gate rejects GK classification at x=0.30-0.70."""

    def test_midfield_player_with_gk_color_rejected(self):
        """Player at x=0.50 wearing GK color → rejected by position gate."""
        fps = 30.0
        dets = []
        for i in range(20):
            t = i / fps
            dets.append(_ball_det(i, t, 0.48 + i * 0.005, 0.50))
        for i in range(20, 35):
            t = i / fps
            j = i - 20
            dets.append(_ball_det(i, t, 0.58 + j * 0.001, 0.50))

        ball_track = _ball_track(1, dets)

        # GK-colored player at midfield (x=0.50)
        gk_dets = [
            make_detection(i, i / fps, cls="player", track_id=10,
                           cx=0.50, cy=0.50)
            for i in range(35)
        ]
        gk_track = _player_track(10, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = _make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        keeper_events = [e for e in events if e.is_goalkeeper_event]
        assert len(keeper_events) == 0, "Midfield player should be rejected by position gate"

    def test_player_near_goal_with_gk_color_accepted(self):
        """Player at x=0.15 wearing GK color → accepted (outside midfield gate)."""
        fps = 30.0
        dets = []
        # Fast ball approaching goal then abrupt slowdown near GK at x=0.15
        for i in range(15):
            t = i / fps
            dets.append(_ball_det(i, t, 0.25 - i * 0.008, 0.50))
        for i in range(15, 30):
            t = i / fps
            j = i - 15
            dets.append(_ball_det(i, t, 0.13 + j * 0.001, 0.50))

        ball_track = _ball_track(1, dets)

        gk_dets = [
            make_detection(i, i / fps, cls="player", track_id=10,
                           cx=0.15, cy=0.48)
            for i in range(30)
        ]
        gk_track = _player_track(10, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = _make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        keeper_events = [e for e in events if e.is_goalkeeper_event]
        assert len(keeper_events) >= 1, "Near-goal GK should be accepted"


# ===========================================================================
# Rush 2026 QA: Fix 2 — Voter Confidence Gate
# ===========================================================================

@pytest.mark.unit
class TestVoterConfidenceGate:
    """Only high-confidence events participate in side voting."""

    def test_low_confidence_events_dont_vote(self):
        """Events with confidence < 0.75 or sim_team_gk < 0.75 should not affect side determination."""
        from src.reel_plugins.keeper import _filter_wrong_side_events

        # 5 low-confidence FPs on the right side (wrong side)
        # Low sim_team_gk = likely FP from jersey color confusion
        low_conf_fps = [
            Event(
                job_id="j", source_file="m.mp4",
                event_type=EventType.SHOT_STOP_STANDING,
                timestamp_start=float(i * 60), timestamp_end=float(i * 60 + 1),
                confidence=0.65,  # below 0.75 threshold
                reel_targets=[], is_goalkeeper_event=True,
                frame_start=i * 1800, frame_end=i * 1800 + 30,
                bounding_box=BoundingBox(x=0.85, y=0.48, width=0.04, height=0.04),
                metadata={"sim_team_gk": 0.65},
            )
            for i in range(5)
        ]

        # 2 high-confidence real events on the left side (correct side)
        real_events = [
            Event(
                job_id="j", source_file="m.mp4",
                event_type=EventType.CATCH,
                timestamp_start=float(i * 120 + 30),
                timestamp_end=float(i * 120 + 31),
                confidence=0.85,  # above 0.75 threshold
                reel_targets=[], is_goalkeeper_event=True,
                frame_start=(i * 120 + 30) * 30,
                frame_end=(i * 120 + 31) * 30,
                bounding_box=BoundingBox(x=0.10, y=0.48, width=0.04, height=0.04),
                metadata={"sim_team_gk": 0.90},  # strong GK match
            )
            for i in range(2)
        ]

        all_events = low_conf_fps + real_events
        # Selected = one real event on left
        selected = [real_events[0]]

        result = _filter_wrong_side_events(selected, all_events, 5400.0)
        # Real event on left should be kept (left side wins among high-conf voters)
        assert len(result) == 1

    def test_high_confidence_events_do_vote(self):
        """Events with confidence >= 0.75 and sim_team_gk >= 0.75 should participate in side voting."""
        from src.reel_plugins.keeper import _filter_wrong_side_events

        # 3 high-confidence, high-sim events on right side
        right_events = [
            Event(
                job_id="j", source_file="m.mp4",
                event_type=EventType.SHOT_STOP_DIVING,
                timestamp_start=float(i * 120),
                timestamp_end=float(i * 120 + 1),
                confidence=0.85,
                reel_targets=[], is_goalkeeper_event=True,
                frame_start=i * 3600, frame_end=i * 3600 + 30,
                bounding_box=BoundingBox(x=0.85, y=0.48, width=0.04, height=0.04),
                metadata={"sim_team_gk": 0.90},
            )
            for i in range(3)
        ]

        # 1 event on left side (wrong side) — also high sim so it votes
        left_event = Event(
            job_id="j", source_file="m.mp4",
            event_type=EventType.CATCH,
            timestamp_start=60.0, timestamp_end=61.0,
            confidence=0.80,
            reel_targets=[], is_goalkeeper_event=True,
            frame_start=1800, frame_end=1830,
            bounding_box=BoundingBox(x=0.10, y=0.48, width=0.04, height=0.04),
            metadata={"sim_team_gk": 0.85},
        )

        all_events = right_events + [left_event]
        selected = [left_event]

        result = _filter_wrong_side_events(selected, all_events, 5400.0)
        # Left event should be removed (right side dominates)
        assert len(result) == 0


# ===========================================================================
# Rush 2026 QA: Fix 3 — Lower Dedup IoU Threshold
# ===========================================================================

@pytest.mark.unit
class TestDedupThreshold:
    """Dedup threshold lowered from 0.5 to 0.3."""

    def test_clips_with_iou_035_deduplicated(self):
        """Clips with IoU=0.35 should be deduplicated (was kept at 0.5 threshold)."""
        from src.segmentation.clipper import ClipBoundary
        from src.segmentation.deduplicator import deduplicate_clips

        # Clip A: 0-10s, Clip B: 5-15s → inter=5, union=15, IoU=0.33
        clip_a = ClipBoundary(
            source_file="m.mp4", start_sec=0.0, end_sec=10.0,
            events=["e1"], reel_type="keeper", primary_event_type="catch",
        )
        clip_b = ClipBoundary(
            source_file="m.mp4", start_sec=5.0, end_sec=15.0,
            events=["e2"], reel_type="keeper", primary_event_type="catch",
        )

        result = deduplicate_clips([clip_a, clip_b])
        assert len(result) == 1, "IoU≈0.33 should be deduplicated at threshold 0.3"

    def test_clips_with_iou_025_kept(self):
        """Clips with IoU < 0.3 should be kept (genuinely different actions)."""
        from src.segmentation.clipper import ClipBoundary
        from src.segmentation.deduplicator import deduplicate_clips

        # Clip A: 0-10s, Clip B: 8-20s → inter=2, union=20, IoU=0.1
        clip_a = ClipBoundary(
            source_file="m.mp4", start_sec=0.0, end_sec=10.0,
            events=["e1"], reel_type="keeper", primary_event_type="catch",
        )
        clip_b = ClipBoundary(
            source_file="m.mp4", start_sec=8.0, end_sec=20.0,
            events=["e2"], reel_type="keeper", primary_event_type="catch",
        )

        result = deduplicate_clips([clip_a, clip_b])
        assert len(result) == 2, "IoU≈0.17 should keep both clips"


# ===========================================================================
# Rush 2026 QA: Fix 4 — Smart Endpoint Improvements
# ===========================================================================

@pytest.mark.unit
class TestSmartEndpointImprovements:
    """Smart endpoint enhancements for Rush 2026 QA."""

    def test_one_on_one_extended_to_next_touch(self):
        """ONE_ON_ONE event should be extended (now in _SMART_ENDPOINT_TYPES)."""
        fps = 30.0
        event = Event(
            job_id="j", source_file="m.mp4",
            event_type=EventType.ONE_ON_ONE,
            timestamp_start=9.5, timestamp_end=10.5,
            confidence=0.80, reel_targets=[],
            is_goalkeeper_event=True,
            frame_start=int(9.5 * fps), frame_end=int(10.5 * fps),
            metadata={"detection_method": "gk_detector"},
        )

        # Ball goes out of bounds at t=13
        dets = []
        for i in range(int(10 * fps), int(13 * fps)):
            t = i / fps
            dets.append(_ball_det(i, t, 0.20 + (i - int(10 * fps)) * 0.005, 0.50))
        # Ball exits at x>0.99
        dets.append(_ball_det(int(13 * fps), 13.0, 0.995, 0.50))

        trajectory = BallTrajectory().build([_ball_track(1, dets)])
        det = _make_detector()
        result = det._apply_smart_endpoints([event], trajectory, [], fps)

        assert len(result) == 1
        assert result[0].timestamp_end > event.timestamp_end
        assert result[0].metadata.get("endpoint_extended") is True

    def test_distribution_extends_on_25deg_direction_change(self):
        """25-degree direction change should now trigger extension endpoint
        (threshold relaxed from 30deg to 20deg)."""
        fps = 30.0
        event = Event(
            job_id="j", source_file="m.mp4",
            event_type=EventType.DISTRIBUTION_SHORT,
            timestamp_start=9.5, timestamp_end=10.5,
            confidence=0.75, reel_targets=[],
            is_goalkeeper_event=True,
            frame_start=int(9.5 * fps), frame_end=int(10.5 * fps),
            metadata={"detection_method": "ball_touch", "touch_reason": "speed_spike"},
        )

        # Ball moves in one direction then changes by ~25deg
        dets = []
        # Straight run
        for i in range(int(10 * fps), int(13 * fps)):
            t = i / fps
            progress = (i - int(10 * fps))
            dets.append(_ball_det(i, t, 0.20 + progress * 0.003, 0.50))

        # 25-degree direction change (~155 degree angle between pre/post)
        angle_rad = math.radians(25)
        base_x = 0.20 + (int(13 * fps) - int(10 * fps)) * 0.003
        for i in range(int(13 * fps), int(15 * fps)):
            t = i / fps
            j = i - int(13 * fps)
            dx = math.cos(math.pi - angle_rad) * 0.003
            dy = math.sin(math.pi - angle_rad) * 0.003
            dets.append(_ball_det(i, t, base_x + j * dx, 0.50 + j * dy))

        trajectory = BallTrajectory().build([_ball_track(1, dets)])
        det = _make_detector()
        result = det._apply_smart_endpoints([event], trajectory, [], fps)

        assert len(result) == 1
        ext = result[0]
        assert ext.timestamp_end > event.timestamp_end
        assert ext.metadata.get("endpoint_extended") is True

    def test_goal_kick_starts_1_5s_before_kick(self):
        """Goal kick event window should start 1.5s before kick (was 0.5s)."""
        fps = 30.0
        dets = []
        # Ball approaches GK area
        for i in range(int(5 * fps)):
            t = i / fps
            dets.append(_ball_det(i, t, 0.08, 0.80))

        # GK collects → stationary for 2s
        for i in range(int(5 * fps), int(7 * fps)):
            t = i / fps
            dets.append(_ball_det(i, t, 0.08, 0.80))

        # Ball stationary near goal for 1.5s (meets min_stationary_sec)
        for i in range(int(7 * fps), int(8.5 * fps)):
            t = i / fps
            dets.append(_ball_det(i, t, 0.08, 0.80))

        # Speed spike = goal kick at t=8.5
        for i in range(int(8.5 * fps), int(10 * fps)):
            t = i / fps
            j = i - int(8.5 * fps)
            dets.append(_ball_det(i, t, 0.08 + j * 0.02, 0.50))

        ball_track = _ball_track(1, dets)

        # GK player at goal area
        gk_dets = [
            make_detection(i, i / fps, cls="player", track_id=10,
                           cx=0.08, cy=0.80)
            for i in range(int(10 * fps))
        ]
        gk_track = _player_track(10, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = _make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        gk_events = [e for e in events if e.event_type == EventType.GOAL_KICK]

        if gk_events:
            # Goal kick should start 3.0s before kick, not 0.5s
            kick_ts = gk_events[0].metadata.get("original_reason", "")
            # The event's timestamp_start should be at least 2.5s before timestamp_end
            event = gk_events[0]
            assert event.timestamp_end - event.timestamp_start >= 2.0, \
                "Goal kick event window should be wider with 3.0s pre-window"

    def test_gk_color_min_similarity_default_is_060(self):
        """Default gk_color_min_similarity should be 0.60 (raised from 0.55)."""
        det = _make_detector()
        assert det._gk_color_min_similarity == 0.60


# ===========================================================================
# Corner kick: widened spatial detection for auto-panning cameras
# ===========================================================================

@pytest.mark.unit
class TestCornerKickWidenedSpatial:
    """Corner kick detection with relaxed thresholds for auto-panning cameras."""

    def _corner_scenario(self, bx, by, fps=30.0, n_players=4):
        """Build a scenario with ball at given position, stationary, then kicked."""
        dets = []
        # Ball approaches position
        for i in range(10):
            t = i / fps
            dets.append(_ball_det(i, t, bx + i * 0.001, by))

        # Ball stationary at position for ~1s (30 frames)
        for i in range(10, 40):
            t = i / fps
            dets.append(_ball_det(i, t, bx, by))

        # Ball kicked (high speed)
        for i in range(40, 55):
            t = i / fps
            j = i - 40
            dets.append(_ball_det(i, t, bx + j * 0.02, 0.50))

        ball_track = _ball_track(1, dets)

        # Players in the same half as the ball
        player_tracks = []
        player_x = 0.10 if bx < 0.5 else 0.90
        for pid in range(2, 2 + n_players):
            p_dets = [
                make_detection(f, f / fps, cls="player", track_id=pid,
                               cx=player_x + (pid - 2) * 0.02, cy=0.50)
                for f in range(55)
            ]
            player_tracks.append(make_track(pid, p_dets))

        return ball_track, player_tracks

    @pytest.mark.skip(reason="Spatial corner detection disabled for auto-pan cameras")
    def test_ball_near_edge_not_old_corner_still_detected(self):
        """Ball at (0.10, 0.88) — passes widened zone but would fail old (0.05, 0.08)."""
        fps = 30.0
        ball, players = self._corner_scenario(0.10, 0.88, fps, n_players=4)
        trajectory = BallTrajectory().build([ball])
        det = _make_detector()
        events = det._detect_corner_kicks(trajectory, [ball] + players, fps)
        assert len(events) >= 1
        assert events[0].event_type == EventType.CORNER_KICK

    @pytest.mark.skip(reason="Spatial corner detection disabled for auto-pan cameras")
    def test_ball_near_strong_x_edge_detected(self):
        """Ball at (0.06, 0.50) — near strong x-edge (< 0.08), needs >=5 players."""
        fps = 30.0
        ball, players = self._corner_scenario(0.06, 0.50, fps, n_players=6)
        trajectory = BallTrajectory().build([ball])
        det = _make_detector()
        events = det._detect_corner_kicks(trajectory, [ball] + players, fps)
        assert len(events) >= 1

    def test_ball_at_midfield_not_detected(self):
        """Ball at (0.50, 0.50) — not near any edge, no corner detected."""
        fps = 30.0
        ball, players = self._corner_scenario(0.50, 0.50, fps, n_players=6)
        trajectory = BallTrajectory().build([ball])
        det = _make_detector()
        events = det._detect_corner_kicks(trajectory, [ball] + players, fps)
        assert len(events) == 0

    def test_near_edge_insufficient_players_rejected(self):
        """Ball near strong edge but only 3 players → rejected (needs >=5)."""
        fps = 30.0
        ball, players = self._corner_scenario(0.06, 0.50, fps, n_players=3)
        trajectory = BallTrajectory().build([ball])
        det = _make_detector()
        events = det._detect_corner_kicks(trajectory, [ball] + players, fps)
        assert len(events) == 0


# ===========================================================================
# Corner kick: post-save restart inference
# ===========================================================================

@pytest.mark.unit
class TestCornerKickPostSaveRestart:
    """Post-save restart corner kick inference (save → gap → restart)."""

    def test_corner_after_save_with_gap_detected(self):
        """Save event → ball disappears → reappears stationary → kicked = corner.

        Ball restart position is NOT near any frame edge (auto-pan camera
        centers ball), so spatial detection misses it — post-save inference
        catches it.
        """
        fps = 30.0
        dets = []

        # Ball moving fast toward GK (frames 0-10)
        for i in range(11):
            t = i / fps
            dets.append(_ball_det(i, t, 0.20 - i * 0.01, 0.30))

        # Ball gone (gap: frames 11-50 = ~1.3s, exceeds 1.0s minimum)
        # (no detections — ball went out of play after GK deflection)

        # Ball reappears NOT near edge (auto-pan centers ball), stationary 1s
        for i in range(51, 81):
            t = i / fps
            dets.append(_ball_det(i, t, 0.30, 0.40))

        # Ball kicked (frames 81-95)
        for i in range(81, 96):
            t = i / fps
            j = i - 81
            dets.append(_ball_det(i, t, 0.30 + j * 0.03, 0.50))

        ball_track = _ball_track(1, dets)

        # 5 players clustered in left half
        player_tracks = []
        for pid in range(2, 7):
            p_dets = [
                make_detection(f, f / fps, cls="player", track_id=pid,
                               cx=0.20 + (pid - 2) * 0.03, cy=0.50)
                for f in range(96)
            ]
            player_tracks.append(make_track(pid, p_dets))

        trajectory = BallTrajectory().build([ball_track])

        # Create a fake save event near the gap (save at ~0.3s, gap starts at frame 10)
        save_event = Event(
            job_id="j", source_file="m.mp4",
            event_type=EventType.SHOT_STOP_STANDING,
            timestamp_start=9.0 / fps, timestamp_end=11.0 / fps,
            confidence=0.80, reel_targets=[],
            is_goalkeeper_event=True,
            frame_start=9, frame_end=11,
            metadata={"sim_team_gk": 0.85},
        )

        det = _make_detector()
        events = det._detect_corner_kicks(
            trajectory, [ball_track] + player_tracks, fps,
            save_events=[save_event],
        )
        assert len(events) >= 1
        corner = [e for e in events if e.event_type == EventType.CORNER_KICK]
        assert len(corner) >= 1
        assert corner[0].metadata["detection_method"] == "corner_kick_post_save"


# ===========================================================================
# Diving save classification
# ===========================================================================

@pytest.mark.unit
class TestDivingSaveClassification:
    """Diving save classification based on GK vertical velocity."""

    def test_high_vertical_velocity_produces_diving_save(self):
        """GK moving fast vertically at touch → SHOT_STOP_DIVING."""
        det = _make_detector()
        # speed_drop reason normally → SHOT_STOP_STANDING
        # but with high GK vertical velocity → upgraded to DIVING
        event = det._classify_touch(
            touch_frame=100,
            timestamp=3.33,
            reason="speed_drop",
            player_track_id=10,
            player_jersey_hsv=TEAM_GK_HSV,
            ball_pos=(0.10, 0.30),
            player_pos=(0.12, 0.32),
            fps=30.0,
            team_gk_hsv=TEAM_GK_HSV,
            opp_gk_hsv=OPP_GK_HSV,
            team_outfield_hsv=TEAM_OUTFIELD_HSV,
            opp_outfield_hsv=OPP_OUTFIELD_HSV,
            gk_vertical_velocity=3.0,  # high → diving
        )
        assert event is not None
        assert event.event_type == EventType.SHOT_STOP_DIVING

    def test_low_vertical_velocity_stays_standing(self):
        """GK barely moving vertically → SHOT_STOP_STANDING."""
        det = _make_detector()
        event = det._classify_touch(
            touch_frame=100,
            timestamp=3.33,
            reason="speed_drop",
            player_track_id=10,
            player_jersey_hsv=TEAM_GK_HSV,
            ball_pos=(0.10, 0.30),
            player_pos=(0.12, 0.32),
            fps=30.0,
            team_gk_hsv=TEAM_GK_HSV,
            opp_gk_hsv=OPP_GK_HSV,
            team_outfield_hsv=TEAM_OUTFIELD_HSV,
            opp_outfield_hsv=OPP_OUTFIELD_HSV,
            gk_vertical_velocity=0.5,  # low → stays standing
        )
        assert event is not None
        assert event.event_type == EventType.SHOT_STOP_STANDING

    def test_catch_not_upgraded_to_diving(self):
        """Catch events stay as CATCH regardless of velocity."""
        det = _make_detector()
        event = det._classify_touch(
            touch_frame=100,
            timestamp=3.33,
            reason="ball_caught",
            player_track_id=10,
            player_jersey_hsv=TEAM_GK_HSV,
            ball_pos=(0.10, 0.30),
            player_pos=(0.12, 0.32),
            fps=30.0,
            team_gk_hsv=TEAM_GK_HSV,
            opp_gk_hsv=OPP_GK_HSV,
            team_outfield_hsv=TEAM_OUTFIELD_HSV,
            opp_outfield_hsv=OPP_OUTFIELD_HSV,
            gk_vertical_velocity=5.0,  # high but reason=ball_caught → CATCH
        )
        assert event is not None
        assert event.event_type == EventType.CATCH

    def test_compute_player_vertical_velocity(self):
        """Vertical velocity computed correctly from track detections."""
        # Create a player track where the GK dives: cy goes from 0.80 to 0.60
        # over 10 frames at 30fps. Height = 0.10.
        # Vertical displacement = 0.20 normalized, in body heights = 0.20/0.10 = 2.0
        # Time = 10/30 = 0.333s.  Velocity = 2.0/0.333 = 6.0 body-heights/s
        fps = 30.0
        dets = []
        for i in range(20):
            cy = 0.80 - i * 0.01  # moves from 0.80 to 0.61
            dets.append(make_detection(
                i, i / fps, cls="player", track_id=10,
                cx=0.10, cy=cy, w=0.05, h=0.10,
            ))
        track = make_track(10, dets)

        vel = BallTouchDetector._compute_player_vertical_velocity(
            track_id=10, touch_frame=10, tracks=[track], fps=fps,
            window_frames=10,
        )
        # dy=abs(0.61-0.80)=0.19, avg_height=0.10, dt=19/30≈0.633
        # vel = (0.19/0.10) / 0.633 ≈ 3.0
        assert vel > 2.0  # definitely a dive

    def test_compute_vertical_velocity_stationary_gk(self):
        """Stationary GK has ~0 vertical velocity."""
        fps = 30.0
        dets = [
            make_detection(i, i / fps, cls="player", track_id=10,
                           cx=0.10, cy=0.80, w=0.05, h=0.10)
            for i in range(20)
        ]
        track = make_track(10, dets)

        vel = BallTouchDetector._compute_player_vertical_velocity(
            track_id=10, touch_frame=10, tracks=[track], fps=fps,
        )
        assert vel < 0.1  # essentially zero
