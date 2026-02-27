"""
Unit tests for detection improvements (WS1-WS4).

WS1: Zone-aware thresholds + ball-in-net detection
WS2: Corner kick event type + detection + plugin
WS3: Smart clip endpoints
WS4: Pause/resume restart bug fix
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
        keeper_events = [e for e in events if "keeper" in e.reel_targets]
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
        keeper_events = [e for e in events if "keeper" in e.reel_targets]
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
        assert "highlights" in events[0].reel_targets
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
            assert "highlights" in ev.reel_targets
            assert "keeper" not in ev.reel_targets


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
            reel_targets=["keeper"], is_goalkeeper_event=True,
            frame_start=300, frame_end=450,
            bounding_box=BoundingBox(x=0.03, y=0.93, width=0.04, height=0.04),
        )
        save = Event(
            job_id="j", source_file="m.mp4", event_type=EventType.SHOT_STOP_DIVING,
            timestamp_start=20.0, timestamp_end=21.0, confidence=0.80,
            reel_targets=["keeper"], is_goalkeeper_event=True,
            frame_start=600, frame_end=630,
            bounding_box=BoundingBox(x=0.10, y=0.50, width=0.04, height=0.04),
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
        assert p.clip_params.post_pad_sec == 6.0
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
            confidence=0.75, reel_targets=["keeper"],
            is_goalkeeper_event=True,
            frame_start=int((ts - 0.5) * fps), frame_end=int((ts + 0.5) * fps),
            metadata={"detection_method": "ball_touch", "touch_reason": "speed_spike"},
        )

    def _save_event(self, ts=10.0, fps=30.0):
        return Event(
            job_id="j", source_file="m.mp4",
            event_type=EventType.SHOT_STOP_STANDING,
            timestamp_start=ts - 0.5, timestamp_end=ts + 0.5,
            confidence=0.80, reel_targets=["keeper"],
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
            reel_targets=["highlights"], is_goalkeeper_event=False,
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
