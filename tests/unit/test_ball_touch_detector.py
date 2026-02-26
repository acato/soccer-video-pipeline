"""
Unit tests for the ball-first touch detector.

Covers:
- BallTrajectory: collection, dedup, smoothing, velocity, gaps
- Touch detection: direction changes, speed drops, catches, NMS
- Player attribution: nearest player, frame tolerance, distance threshold
- GK classification: team GK → keeper, opponent → skip, outfield → skip
- Integration: classic saves, catches, distribution suppression, plugin compat
"""
from __future__ import annotations

import pytest

from src.detection.ball_touch_detector import BallTouchDetector, BallTrajectory
from src.detection.models import (
    BoundingBox, Detection, Event, EventType, Track,
    is_gk_event_type,
)
from tests.conftest import make_bbox, make_detection, make_match_config, make_track


# ---------------------------------------------------------------------------
# HSV colors matching the default test MatchConfig
# ---------------------------------------------------------------------------
# Home FC: outfield=blue (112, 0.82, 0.65), gk=neon_yellow (35, 0.95, 0.95)
# Away United: outfield=red (0, 0.85, 0.70), gk=teal (88, 0.80, 0.55)
TEAM_GK_HSV = (35.0, 0.95, 0.95)      # neon_yellow
OPP_GK_HSV = (88.0, 0.80, 0.55)       # teal
TEAM_OUTFIELD_HSV = (112.0, 0.82, 0.65)  # blue
OPP_OUTFIELD_HSV = (0.0, 0.85, 0.70)    # red


def _ball_det(frame: int, ts: float, cx: float, cy: float, conf: float = 0.9) -> Detection:
    """Shorthand for a ball detection."""
    return make_detection(frame, ts, cls="ball", cx=cx, cy=cy, w=0.02, h=0.02)


def _player_det(
    frame: int, ts: float, track_id: int,
    cx: float = 0.5, cy: float = 0.5,
    jersey_hsv: list = None,
) -> Detection:
    """Shorthand for a player detection."""
    return make_detection(frame, ts, cls="player", track_id=track_id,
                          cx=cx, cy=cy, jersey_hsv=jersey_hsv)


def _ball_track(track_id: int, detections: list[Detection]) -> Track:
    """Create a track containing ball detections."""
    return make_track(track_id, detections)


def _player_track(
    track_id: int, detections: list[Detection],
    jersey_hsv: tuple = None,
) -> Track:
    """Create a player track with jersey color."""
    return make_track(track_id, detections, jersey_hsv=jersey_hsv)


# ===========================================================================
# BallTrajectory tests
# ===========================================================================

@pytest.mark.unit
class TestBallTrajectory:

    def test_build_from_single_track(self):
        """Ball dets from one track are collected."""
        dets = [_ball_det(i, i / 30.0, 0.5 + i * 0.01, 0.5) for i in range(10)]
        track = _ball_track(1, dets)
        traj = BallTrajectory().build([track])
        assert len(traj) == 10
        assert traj.frames == list(range(10))

    def test_build_from_multiple_tracks(self):
        """Ball dets scattered across player tracks are all collected."""
        t1 = _ball_track(1, [_ball_det(0, 0.0, 0.3, 0.5), _ball_det(2, 0.067, 0.32, 0.5)])
        t2 = _ball_track(2, [_ball_det(1, 0.033, 0.31, 0.5), _ball_det(3, 0.1, 0.33, 0.5)])
        traj = BallTrajectory().build([t1, t2])
        assert len(traj) == 4
        assert traj.frames == [0, 1, 2, 3]

    def test_dedup_same_frame_highest_conf(self):
        """When two ball dets on same frame, highest confidence wins."""
        d1 = Detection(
            frame_number=5, timestamp=0.167, class_name="ball",
            confidence=0.7, bbox=make_bbox(0.3, 0.5, 0.02, 0.02),
        )
        d2 = Detection(
            frame_number=5, timestamp=0.167, class_name="ball",
            confidence=0.9, bbox=make_bbox(0.6, 0.5, 0.02, 0.02),
        )
        t1 = make_track(1, [d1])
        t2 = make_track(2, [d2])
        traj = BallTrajectory().build([t1, t2])
        assert len(traj) == 1
        pos = traj.position_at(5)
        assert pos is not None
        assert abs(pos[0] - 0.6) < 0.01  # higher confidence det at x=0.6

    def test_median_smoothing_removes_jitter(self):
        """Median smoothing removes single-frame outliers."""
        dets = [
            _ball_det(0, 0.0, 0.50, 0.50),
            _ball_det(1, 0.033, 0.51, 0.50),  # normal
            _ball_det(2, 0.067, 0.80, 0.50),  # outlier jitter
            _ball_det(3, 0.100, 0.53, 0.50),  # normal
            _ball_det(4, 0.133, 0.54, 0.50),
        ]
        traj = BallTrajectory().build([_ball_track(1, dets)])
        # Smoothed frame 2 should be median of [0.51, 0.80, 0.53] = 0.53
        pos = traj.position_at(2)
        assert pos is not None
        assert abs(pos[0] - 0.53) < 0.01

    def test_velocity_computation(self):
        """Velocity at a frame uses centered window."""
        # Ball moves right at 0.3/s (10 frames at 30fps, 0.01 per frame)
        fps = 30.0
        dets = [_ball_det(i, i / fps, 0.3 + i * 0.01, 0.5) for i in range(20)]
        traj = BallTrajectory().build([_ball_track(1, dets)])
        vel = traj.velocity_at(10, window=5)
        assert vel is not None
        vx, vy = vel
        assert vx > 0.2  # Moving right
        assert abs(vy) < 0.05  # Not moving vertically

    def test_speed_computation(self):
        """Speed is magnitude of velocity."""
        fps = 30.0
        dets = [_ball_det(i, i / fps, 0.3 + i * 0.01, 0.5) for i in range(20)]
        traj = BallTrajectory().build([_ball_track(1, dets)])
        speed = traj.speed_at(10, window=5)
        assert speed is not None
        assert speed > 0.2

    def test_find_gaps(self):
        """Gaps in ball detections are found."""
        dets = (
            [_ball_det(i, i / 30.0, 0.5, 0.5) for i in range(10)]
            + [_ball_det(i, i / 30.0, 0.5, 0.5) for i in range(25, 35)]
        )
        traj = BallTrajectory().build([_ball_track(1, dets)])
        gaps = traj.find_gaps(min_gap_frames=5)
        assert len(gaps) == 1
        assert gaps[0] == (9, 25)

    def test_empty_input(self):
        """Empty tracks produce empty trajectory."""
        traj = BallTrajectory().build([])
        assert len(traj) == 0
        assert traj.position_at(0) is None
        assert traj.velocity_at(0) is None
        assert traj.speed_at(0) is None
        assert traj.find_gaps() == []

    def test_position_at_missing_frame(self):
        """Position at non-existent frame returns None."""
        dets = [_ball_det(0, 0.0, 0.5, 0.5)]
        traj = BallTrajectory().build([_ball_track(1, dets)])
        assert traj.position_at(999) is None

    def test_no_ball_detections(self):
        """Track with only player dets → empty trajectory."""
        dets = [_player_det(i, i / 30.0, track_id=1) for i in range(10)]
        traj = BallTrajectory().build([make_track(1, dets)])
        assert len(traj) == 0


# ===========================================================================
# Touch detection tests
# ===========================================================================

@pytest.mark.unit
class TestTouchDetection:

    def _make_detector(self, **kwargs) -> BallTouchDetector:
        mc = make_match_config()
        defaults = dict(
            job_id="job-001",
            source_file="match.mp4",
            match_config=mc,
        )
        defaults.update(kwargs)
        return BallTouchDetector(**defaults)

    def test_direction_change_detected(self):
        """Ball changing direction >40° is detected as a touch."""
        fps = 30.0
        # Ball moves right then turns sharply upward
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(10)]  # right
            + [_ball_det(10 + i, (10 + i) / fps, 0.30, 0.50 - i * 0.02) for i in range(10)]  # up
        )
        # GK near the bend point — present across multiple frames (realistic)
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.30, cy=0.50) for f in range(7, 14)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        assert len(events) >= 1
        assert events[0].is_goalkeeper_event
        assert "keeper" in events[0].reel_targets

    def test_speed_drop_detected(self):
        """Ball decelerating >50% is detected as a touch."""
        fps = 30.0
        # Ball moves fast then slows
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.03, 0.50) for i in range(10)]  # fast
            + [_ball_det(10 + i, (10 + i) / fps, 0.40 + i * 0.002, 0.50) for i in range(10)]  # slow
        )
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.40, cy=0.50) for f in range(7, 14)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        assert len(events) >= 1

    def test_ball_caught_detected(self):
        """Ball disappearing after fast approach = catch."""
        fps = 30.0
        # Ball moves fast then vanishes (end of trajectory = caught)
        ball_dets = [_ball_det(i, i / fps, 0.10 + i * 0.03, 0.50) for i in range(10)]
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.37, cy=0.50) for f in range(6, 12)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        # Should detect a ball_caught event at the end of the trajectory
        caught = [e for e in events if e.metadata.get("touch_reason") == "ball_caught"]
        assert len(caught) >= 1
        assert caught[0].event_type == EventType.CATCH

    def test_straight_line_no_touch(self):
        """Ball in straight line → no touch detected."""
        fps = 30.0
        ball_dets = [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(30)]
        ball_track = _ball_track(1, ball_dets)

        det = self._make_detector()
        events = det.detect_touches([ball_track], fps)
        assert len(events) == 0

    def test_slow_ball_ignored(self):
        """Stationary/slow ball doesn't trigger touches."""
        fps = 30.0
        # Ball barely moves
        ball_dets = [_ball_det(i, i / fps, 0.50 + i * 0.0005, 0.50) for i in range(30)]
        gk_dets = [_player_det(15, 15 / fps, track_id=99, cx=0.50, cy=0.50)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        assert len(events) == 0

    def test_nms_deduplication(self):
        """Touches within NMS window are deduplicated."""
        fps = 30.0
        # Two sharp direction changes close together (inside NMS window)
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(8)]
            + [_ball_det(8 + i, (8 + i) / fps, 0.26 - i * 0.02, 0.50) for i in range(3)]
            + [_ball_det(11 + i, (11 + i) / fps, 0.20 + i * 0.02, 0.50) for i in range(8)]
        )
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.26, cy=0.50) for f in range(5, 15)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        # Should produce at most 1 event due to NMS
        assert len(events) <= 1


# ===========================================================================
# Player attribution tests
# ===========================================================================

@pytest.mark.unit
class TestPlayerAttribution:

    def _make_detector(self, **kwargs) -> BallTouchDetector:
        mc = make_match_config()
        defaults = dict(
            job_id="job-001",
            source_file="match.mp4",
            match_config=mc,
        )
        defaults.update(kwargs)
        return BallTouchDetector(**defaults)

    def test_nearest_player_found(self):
        """Touch is attributed to the closest player."""
        fps = 30.0
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(10)]
            + [_ball_det(10 + i, (10 + i) / fps, 0.30, 0.50 - i * 0.02) for i in range(10)]
        )
        # Two players near bend: GK closer, outfield farther (both present across frames)
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.30, cy=0.50) for f in range(7, 14)]
        outfield_dets = [_player_det(f, f / fps, track_id=88, cx=0.35, cy=0.55) for f in range(7, 14)]
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)
        of_track = _player_track(88, outfield_dets, jersey_hsv=TEAM_OUTFIELD_HSV)
        ball_track = _ball_track(1, ball_dets)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track, of_track], fps)
        if events:
            assert events[0].player_track_id == 99

    def test_no_nearby_player_no_event(self):
        """If no player within max_player_distance, no event."""
        fps = 30.0
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(10)]
            + [_ball_det(10 + i, (10 + i) / fps, 0.30, 0.50 - i * 0.02) for i in range(10)]
        )
        # Player far from touch point
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.80, cy=0.80) for f in range(7, 14)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector(max_player_distance=0.12)
        events = det.detect_touches([ball_track, gk_track], fps)
        assert len(events) == 0

    def test_frame_tolerance(self):
        """Player detected ±2 frames from touch still gets attributed."""
        fps = 30.0
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(10)]
            + [_ball_det(10 + i, (10 + i) / fps, 0.30, 0.50 - i * 0.02) for i in range(10)]
        )
        # GK only at frames 8-9 — 2 frames before the touch at frame ~10
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.28, cy=0.50) for f in range(8, 10)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        # Should still attribute because of ±2 frame tolerance
        gk_events = [e for e in events if e.player_track_id == 99]
        assert len(gk_events) >= 1

    def test_multiple_players_closest_wins(self):
        """When multiple players near ball, closest one wins."""
        fps = 30.0
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(10)]
            + [_ball_det(10 + i, (10 + i) / fps, 0.30, 0.50 - i * 0.02) for i in range(10)]
        )
        # Two GKs at different distances (both present across frames)
        close_dets = [_player_det(f, f / fps, track_id=99, cx=0.30, cy=0.50) for f in range(7, 14)]
        far_dets = [_player_det(f, f / fps, track_id=88, cx=0.33, cy=0.53) for f in range(7, 14)]
        close_track = _player_track(99, close_dets, jersey_hsv=TEAM_GK_HSV)
        far_track = _player_track(88, far_dets, jersey_hsv=TEAM_GK_HSV)
        ball_track = _ball_track(1, ball_dets)

        det = self._make_detector()
        events = det.detect_touches([ball_track, close_track, far_track], fps)
        if events:
            assert events[0].player_track_id == 99


# ===========================================================================
# GK classification tests
# ===========================================================================

@pytest.mark.unit
class TestGKClassification:

    def _make_detector(self, **kwargs) -> BallTouchDetector:
        mc = make_match_config()
        defaults = dict(
            job_id="job-001",
            source_file="match.mp4",
            match_config=mc,
        )
        defaults.update(kwargs)
        return BallTouchDetector(**defaults)

    def _make_save_scenario(self, player_hsv: tuple, player_tid: int = 99):
        """Create a basic direction-change save scenario with given player."""
        fps = 30.0
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(10)]
            + [_ball_det(10 + i, (10 + i) / fps, 0.30, 0.50 - i * 0.02) for i in range(10)]
        )
        # Player present across multiple frames (realistic tracking)
        gk_dets = [_player_det(f, f / fps, track_id=player_tid, cx=0.30, cy=0.50) for f in range(7, 14)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(player_tid, gk_dets, jersey_hsv=player_hsv)
        return [ball_track, gk_track], fps

    def test_team_gk_produces_keeper_event(self):
        """Team GK jersey → keeper reel event."""
        tracks, fps = self._make_save_scenario(TEAM_GK_HSV)
        det = self._make_detector()
        events = det.detect_touches(tracks, fps)
        keeper_events = [e for e in events if "keeper" in e.reel_targets]
        assert len(keeper_events) >= 1
        assert keeper_events[0].is_goalkeeper_event

    def test_opponent_gk_skipped(self):
        """Opponent GK jersey → no keeper event."""
        tracks, fps = self._make_save_scenario(OPP_GK_HSV)
        det = self._make_detector()
        events = det.detect_touches(tracks, fps)
        keeper_events = [e for e in events if "keeper" in e.reel_targets]
        assert len(keeper_events) == 0

    def test_outfield_player_skipped(self):
        """Outfield player jersey → no keeper event."""
        tracks, fps = self._make_save_scenario(TEAM_OUTFIELD_HSV)
        det = self._make_detector()
        events = det.detect_touches(tracks, fps)
        keeper_events = [e for e in events if "keeper" in e.reel_targets]
        assert len(keeper_events) == 0

    def test_no_jersey_color_skipped(self):
        """No jersey color data → no event."""
        fps = 30.0
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(10)]
            + [_ball_det(10 + i, (10 + i) / fps, 0.30, 0.50 - i * 0.02) for i in range(10)]
        )
        gk_dets = [_player_det(10, 10 / fps, track_id=99, cx=0.30, cy=0.50)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=None)  # No color

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        assert len(events) == 0

    def test_ambiguous_color_skipped(self):
        """Jersey color that doesn't clearly match any role → no event."""
        # Gray: not close to any team color
        ambiguous_hsv = (0.0, 0.06, 0.45)  # gray
        tracks, fps = self._make_save_scenario(ambiguous_hsv)
        det = self._make_detector()
        events = det.detect_touches(tracks, fps)
        keeper_events = [e for e in events if "keeper" in e.reel_targets]
        assert len(keeper_events) == 0


# ===========================================================================
# Integration tests
# ===========================================================================

@pytest.mark.unit
class TestBallTouchIntegration:

    def _make_detector(self, **kwargs) -> BallTouchDetector:
        mc = make_match_config()
        defaults = dict(
            job_id="job-001",
            source_file="match.mp4",
            match_config=mc,
        )
        defaults.update(kwargs)
        return BallTouchDetector(**defaults)

    def test_classic_save_deflection(self):
        """Ball heading toward goal, GK deflects it (direction change)."""
        fps = 30.0
        # Ball approaches from right, GK at left edge deflects upward
        ball_dets = (
            [_ball_det(i, i / fps, 0.30 - i * 0.02, 0.50) for i in range(10)]
            + [_ball_det(10 + i, (10 + i) / fps, 0.10 + i * 0.005, 0.50 - i * 0.02) for i in range(10)]
        )
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.10, cy=0.50) for f in range(7, 14)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        assert len(events) >= 1
        assert events[0].is_goalkeeper_event
        assert "keeper" in events[0].reel_targets
        assert events[0].metadata["detection_method"] == "ball_touch"

    def test_fast_shot_caught(self):
        """Fast ball disappears near GK → CATCH event."""
        fps = 30.0
        ball_dets = [_ball_det(i, i / fps, 0.50 - i * 0.04, 0.50) for i in range(10)]
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.14, cy=0.50) for f in range(6, 12)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        catch_events = [e for e in events if e.event_type == EventType.CATCH]
        assert len(catch_events) >= 1

    def test_slow_backpass_ignored(self):
        """Slow ball near GK → no event (below min_touch_speed)."""
        fps = 30.0
        ball_dets = [_ball_det(i, i / fps, 0.10 + i * 0.001, 0.50) for i in range(30)]
        gk_dets = [_player_det(15, 15 / fps, track_id=99, cx=0.12, cy=0.50)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        assert len(events) == 0

    def test_ball_passing_by_no_event(self):
        """Ball passes near GK without changing trajectory → no event."""
        fps = 30.0
        ball_dets = [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(30)]
        gk_dets = [_player_det(5, 5 / fps, track_id=99, cx=0.20, cy=0.50)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        # Ball is in straight line → no direction/speed change → no touch
        assert len(events) == 0

    def test_opponent_gk_not_in_keeper_reel(self):
        """Opponent GK touch → not in keeper reel."""
        fps = 30.0
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(10)]
            + [_ball_det(10 + i, (10 + i) / fps, 0.30, 0.50 - i * 0.02) for i in range(10)]
        )
        opp_gk_dets = [_player_det(f, f / fps, track_id=88, cx=0.30, cy=0.50) for f in range(7, 14)]
        ball_track = _ball_track(1, ball_dets)
        opp_track = _player_track(88, opp_gk_dets, jersey_hsv=OPP_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, opp_track], fps)
        keeper_events = [e for e in events if "keeper" in e.reel_targets]
        assert len(keeper_events) == 0

    def test_distribution_suppressed_by_default(self):
        """Ball appears near GK (distribution) → suppressed when detect_distributions=False."""
        fps = 30.0
        # Ball appears out of nowhere near GK (throw)
        ball_dets = [_ball_det(20 + i, (20 + i) / fps, 0.10 + i * 0.03, 0.50) for i in range(10)]
        gk_dets = [_player_det(20, 20 / fps, track_id=99, cx=0.10, cy=0.50)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector(detect_distributions=False)
        events = det.detect_touches([ball_track, gk_track], fps)
        dist_events = [
            e for e in events
            if e.event_type in (EventType.DISTRIBUTION_SHORT, EventType.DISTRIBUTION_LONG)
        ]
        assert len(dist_events) == 0

    def test_event_metadata_has_detection_method(self):
        """All events have detection_method='ball_touch' in metadata."""
        fps = 30.0
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(10)]
            + [_ball_det(10 + i, (10 + i) / fps, 0.30, 0.50 - i * 0.02) for i in range(10)]
        )
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.30, cy=0.50) for f in range(7, 14)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        for e in events:
            assert e.metadata["detection_method"] == "ball_touch"

    def test_events_compatible_with_keeper_plugin(self):
        """Events have correct fields for KeeperSavesPlugin selection."""
        fps = 30.0
        ball_dets = (
            [_ball_det(i, i / fps, 0.10 + i * 0.02, 0.50) for i in range(10)]
            + [_ball_det(10 + i, (10 + i) / fps, 0.30, 0.50 - i * 0.02) for i in range(10)]
        )
        gk_dets = [_player_det(f, f / fps, track_id=99, cx=0.30, cy=0.50) for f in range(7, 14)]
        ball_track = _ball_track(1, ball_dets)
        gk_track = _player_track(99, gk_dets, jersey_hsv=TEAM_GK_HSV)

        det = self._make_detector()
        events = det.detect_touches([ball_track, gk_track], fps)
        for e in events:
            assert "keeper" in e.reel_targets
            assert e.is_goalkeeper_event
            assert is_gk_event_type(e.event_type)
            assert e.confidence > 0
            assert e.timestamp_start >= 0
            assert e.timestamp_end > e.timestamp_start

    def test_insufficient_ball_data(self):
        """< 4 ball detections → no events, no crash."""
        fps = 30.0
        ball_dets = [_ball_det(0, 0.0, 0.5, 0.5)]
        ball_track = _ball_track(1, ball_dets)
        det = self._make_detector()
        events = det.detect_touches([ball_track], fps)
        assert events == []

    def test_merge_nearby_events(self):
        """Events within 2s of each other are merged."""
        from src.detection.ball_touch_detector import BallTouchDetector
        e1 = Event(
            job_id="j", source_file="f", event_type=EventType.CATCH,
            timestamp_start=10.0, timestamp_end=10.5, confidence=0.8,
            reel_targets=["keeper"], is_goalkeeper_event=True,
            frame_start=300, frame_end=315,
        )
        e2 = Event(
            job_id="j", source_file="f", event_type=EventType.CATCH,
            timestamp_start=11.0, timestamp_end=11.5, confidence=0.85,
            reel_targets=["keeper"], is_goalkeeper_event=True,
            frame_start=330, frame_end=345,
        )
        merged = BallTouchDetector._merge_nearby_events([e1, e2], min_gap_sec=2.0)
        assert len(merged) == 1
        assert merged[0].confidence == 0.85
        assert merged[0].timestamp_end == 11.5


# ===========================================================================
# Dead-ball reclassification tests
# ===========================================================================

@pytest.mark.unit
class TestDeadBallReclassification:
    """
    Tests for _reclassify_dead_ball_collections:
    GK collects ball → ball stationary >1s → kicked → reclassify as GOAL_KICK.
    """

    def _make_detector(self, **kwargs) -> BallTouchDetector:
        mc = make_match_config()
        defaults = dict(
            job_id="job-001",
            source_file="match.mp4",
            match_config=mc,
        )
        defaults.update(kwargs)
        return BallTouchDetector(**defaults)

    def _make_save_event(self, ts: float = 10.0, fps: float = 30.0,
                         event_type=EventType.CATCH) -> Event:
        """Create a save/catch event at the given timestamp."""
        return Event(
            job_id="job-001",
            source_file="match.mp4",
            event_type=event_type,
            timestamp_start=max(0, ts - 0.5),
            timestamp_end=ts + 0.5,
            confidence=0.80,
            reel_targets=["keeper"],
            is_goalkeeper_event=True,
            frame_start=max(0, int((ts - 0.5) * fps)),
            frame_end=int((ts + 0.5) * fps),
            metadata={
                "detection_method": "ball_touch",
                "touch_reason": "ball_caught",
                "sim_team_gk": 0.75,
                "player_track_id": 99,
            },
        )

    def _build_trajectory_with_stationary(
        self, fps: float = 30.0, touch_ts: float = 10.0,
        stationary_dur: float = 2.0, kick_speed: float = 0.04,
    ) -> BallTrajectory:
        """
        Build a trajectory: ball moving → stationary for stationary_dur → kicked.
        touch_ts is when the GK catches the ball, then it goes stationary.
        """
        dets = []
        # Pre-touch: ball moving fast toward GK (before the catch event)
        for i in range(int(touch_ts * fps) - 30, int(touch_ts * fps)):
            t = i / fps
            dets.append(_ball_det(i, t, 0.30 - (int(touch_ts * fps) - i) * 0.005, 0.50))

        # Post-touch: ball stationary (GK placed it)
        stationary_start = touch_ts + 0.5  # just after the catch event ends
        stationary_end = stationary_start + stationary_dur
        for i in range(int(stationary_start * fps), int(stationary_end * fps)):
            t = i / fps
            dets.append(_ball_det(i, t, 0.10, 0.50))  # ball sitting still

        # Kick: ball starts moving again
        kick_start = stationary_end
        for i in range(int(kick_start * fps), int(kick_start * fps) + 15):
            t = i / fps
            dets.append(_ball_det(i, t, 0.10 + (i - int(kick_start * fps)) * kick_speed, 0.50))

        track = _ball_track(1, dets)
        return BallTrajectory().build([track])

    def test_catch_then_stationary_then_kick_becomes_goal_kick(self):
        """GK catches ball, places it, kicks → GOAL_KICK replaces CATCH."""
        fps = 30.0
        save_event = self._make_save_event(ts=10.0, fps=fps)
        trajectory = self._build_trajectory_with_stationary(
            fps=fps, touch_ts=10.0, stationary_dur=2.0, kick_speed=0.04,
        )

        det = self._make_detector()
        result = det._reclassify_dead_ball_collections([save_event], trajectory, fps)

        assert len(result) == 1
        assert result[0].event_type == EventType.GOAL_KICK
        assert result[0].is_goalkeeper_event
        assert "keeper" in result[0].reel_targets
        assert result[0].metadata["touch_reason"] == "goal_kick"
        assert result[0].metadata["original_reason"] == "ball_caught"

    def test_standing_stop_reclassified_too(self):
        """SHOT_STOP_STANDING followed by stationary+kick → GOAL_KICK."""
        fps = 30.0
        save_event = self._make_save_event(
            ts=10.0, fps=fps, event_type=EventType.SHOT_STOP_STANDING,
        )
        save_event.metadata["touch_reason"] = "speed_drop"
        trajectory = self._build_trajectory_with_stationary(
            fps=fps, touch_ts=10.0, stationary_dur=2.0, kick_speed=0.04,
        )

        det = self._make_detector()
        result = det._reclassify_dead_ball_collections([save_event], trajectory, fps)

        assert len(result) == 1
        assert result[0].event_type == EventType.GOAL_KICK

    def test_no_stationary_period_keeps_save(self):
        """If ball doesn't go stationary after catch, keep original event."""
        fps = 30.0
        save_event = self._make_save_event(ts=10.0, fps=fps)

        # Build trajectory where ball bounces immediately after touch (no stationary)
        dets = []
        for i in range(int(10.0 * fps) - 30, int(10.0 * fps)):
            t = i / fps
            dets.append(_ball_det(i, t, 0.30 - (int(10.0 * fps) - i) * 0.005, 0.50))
        # Ball continues moving after touch (deflection, not placed)
        for i in range(int(10.5 * fps), int(10.5 * fps) + 30):
            t = i / fps
            dets.append(_ball_det(i, t, 0.10 + (i - int(10.5 * fps)) * 0.02, 0.40))
        trajectory = BallTrajectory().build([_ball_track(1, dets)])

        det = self._make_detector()
        result = det._reclassify_dead_ball_collections([save_event], trajectory, fps)

        assert len(result) == 1
        assert result[0].event_type == EventType.CATCH  # preserved

    def test_short_stationary_keeps_save(self):
        """Ball stationary for <1s → not a goal kick, keep original."""
        fps = 30.0
        save_event = self._make_save_event(ts=10.0, fps=fps)
        trajectory = self._build_trajectory_with_stationary(
            fps=fps, touch_ts=10.0, stationary_dur=0.5, kick_speed=0.04,
        )

        det = self._make_detector()
        result = det._reclassify_dead_ball_collections([save_event], trajectory, fps)

        assert len(result) == 1
        assert result[0].event_type == EventType.CATCH  # kept as save

    def test_non_save_events_pass_through(self):
        """Non-save events (distribution, goals) are not reclassified."""
        fps = 30.0
        dist_event = Event(
            job_id="job-001", source_file="match.mp4",
            event_type=EventType.DISTRIBUTION_LONG,
            timestamp_start=10.0, timestamp_end=11.0, confidence=0.80,
            reel_targets=["keeper"], is_goalkeeper_event=True,
            frame_start=300, frame_end=330,
            metadata={"detection_method": "ball_touch", "touch_reason": "speed_spike"},
        )
        trajectory = self._build_trajectory_with_stationary(
            fps=fps, touch_ts=10.0, stationary_dur=2.0,
        )

        det = self._make_detector()
        result = det._reclassify_dead_ball_collections([dist_event], trajectory, fps)

        assert len(result) == 1
        assert result[0].event_type == EventType.DISTRIBUTION_LONG  # unchanged

    def test_mixed_events_only_saves_reclassified(self):
        """Mix of saves and non-saves: only saves with dead-ball pattern change."""
        fps = 30.0
        save = self._make_save_event(ts=10.0, fps=fps)
        shot = Event(
            job_id="job-001", source_file="match.mp4",
            event_type=EventType.SHOT_ON_TARGET,
            timestamp_start=20.0, timestamp_end=21.0, confidence=0.70,
            reel_targets=["highlights"], is_goalkeeper_event=False,
            frame_start=600, frame_end=630,
            metadata={"detection_method": "ball_touch", "touch_reason": "speed_drop"},
        )
        trajectory = self._build_trajectory_with_stationary(
            fps=fps, touch_ts=10.0, stationary_dur=2.0,
        )

        det = self._make_detector()
        result = det._reclassify_dead_ball_collections([save, shot], trajectory, fps)

        assert len(result) == 2
        assert result[0].event_type == EventType.GOAL_KICK  # reclassified
        assert result[1].event_type == EventType.SHOT_ON_TARGET  # unchanged

    def test_goal_kick_event_compatible_with_distribution_plugin(self):
        """Reclassified GOAL_KICK events have correct fields for KeeperDistributionPlugin."""
        fps = 30.0
        save_event = self._make_save_event(ts=10.0, fps=fps)
        trajectory = self._build_trajectory_with_stationary(
            fps=fps, touch_ts=10.0, stationary_dur=2.0,
        )

        det = self._make_detector()
        result = det._reclassify_dead_ball_collections([save_event], trajectory, fps)

        assert len(result) == 1
        gk_event = result[0]
        assert gk_event.event_type == EventType.GOAL_KICK
        assert gk_event.is_goalkeeper_event
        assert "keeper" in gk_event.reel_targets
        # Verify it would be picked up by KeeperDistributionPlugin
        from src.reel_plugins.keeper import KeeperDistributionPlugin, _DISTRIBUTION_TYPES
        assert gk_event.event_type in _DISTRIBUTION_TYPES

    def test_empty_events_returns_empty(self):
        """Empty event list → empty result."""
        fps = 30.0
        trajectory = self._build_trajectory_with_stationary(fps=fps, touch_ts=10.0)
        det = self._make_detector()
        result = det._reclassify_dead_ball_collections([], trajectory, fps)
        assert result == []
