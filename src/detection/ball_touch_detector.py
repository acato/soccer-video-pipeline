"""
Ball-first touch detector for goalkeeper event detection.

Inverts the traditional GK-first approach: instead of finding the GK then
looking for ball contacts, this detector builds a ball trajectory first,
finds moments where the ball changes behavior (direction, speed, disappearance),
then looks up who touched it.  If the toucher wears a GK jersey, it's a
keeper event.

This eliminates the GK identification bottleneck — the ball drives all decisions.
"""
from __future__ import annotations

import bisect
import math
from typing import Optional

import numpy as np
import structlog

from src.detection.models import (
    BoundingBox, Detection, Event, EventType, Track,
)
from src.ingestion.models import MatchConfig

log = structlog.get_logger(__name__)


class BallTrajectory:
    """
    Collect all ball detections across all tracks, deduplicate by frame
    (highest confidence wins), sort by frame, and apply median smoothing
    to remove YOLO jitter while preserving real direction changes.

    No interpolation of gaps — gaps are themselves a signal (ball caught/occluded).
    """

    def __init__(self) -> None:
        # frame → (timestamp, x, y, confidence)
        self._positions: dict[int, tuple[float, float, float, float]] = {}
        self._sorted_frames: list[int] = []
        self._smoothed_x: dict[int, float] = {}
        self._smoothed_y: dict[int, float] = {}

    def build(self, all_tracks: list[Track]) -> "BallTrajectory":
        """Collect ball detections from all tracks, deduplicate, sort, smooth."""
        # Collect: ByteTrack is class-agnostic so ball dets scatter across tracks
        raw: dict[int, tuple[float, float, float, float]] = {}
        for track in all_tracks:
            for det in track.detections:
                if det.class_name != "ball":
                    continue
                frame = det.frame_number
                existing = raw.get(frame)
                if existing is None or det.confidence > existing[3]:
                    raw[frame] = (
                        det.timestamp,
                        det.bbox.center_x,
                        det.bbox.center_y,
                        det.confidence,
                    )

        self._positions = raw
        self._sorted_frames = sorted(raw.keys())

        # Median smoothing (window=3) on x, y — removes single-frame jitter
        self._smooth(window=3)
        return self

    def _smooth(self, window: int = 3) -> None:
        """Apply median smoothing to x, y positions."""
        frames = self._sorted_frames
        n = len(frames)
        half = window // 2

        for i, frame in enumerate(frames):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            neighbor_frames = frames[lo:hi]
            xs = [self._positions[f][1] for f in neighbor_frames]
            ys = [self._positions[f][2] for f in neighbor_frames]
            self._smoothed_x[frame] = float(np.median(xs))
            self._smoothed_y[frame] = float(np.median(ys))

    def position_at(self, frame: int) -> Optional[tuple[float, float]]:
        """Return smoothed (x, y) at frame, or None if no detection."""
        if frame not in self._smoothed_x:
            return None
        return (self._smoothed_x[frame], self._smoothed_y[frame])

    def timestamp_at(self, frame: int) -> Optional[float]:
        """Return timestamp at frame, or None if no detection."""
        pos = self._positions.get(frame)
        return pos[0] if pos is not None else None

    def velocity_at(self, frame: int, window: int = 5) -> Optional[tuple[float, float]]:
        """
        Compute velocity (vx, vy) at frame using a centered window.
        Returns None if insufficient data.
        """
        idx = bisect.bisect_left(self._sorted_frames, frame)
        if idx >= len(self._sorted_frames) or self._sorted_frames[idx] != frame:
            return None

        half = window // 2
        lo = max(0, idx - half)
        hi = min(len(self._sorted_frames), idx + half + 1)

        if hi - lo < 2:
            return None

        f_start = self._sorted_frames[lo]
        f_end = self._sorted_frames[hi - 1]
        t_start = self._positions[f_start][0]
        t_end = self._positions[f_end][0]
        dt = t_end - t_start

        if dt <= 0:
            return None

        x_start = self._smoothed_x[f_start]
        x_end = self._smoothed_x[f_end]
        y_start = self._smoothed_y[f_start]
        y_end = self._smoothed_y[f_end]

        return ((x_end - x_start) / dt, (y_end - y_start) / dt)

    def speed_at(self, frame: int, window: int = 5) -> Optional[float]:
        """Compute speed (magnitude of velocity) at frame."""
        vel = self.velocity_at(frame, window)
        if vel is None:
            return None
        return (vel[0] ** 2 + vel[1] ** 2) ** 0.5

    def find_gaps(self, min_gap_frames: int = 5) -> list[tuple[int, int]]:
        """
        Find gaps in ball detections.
        Returns list of (last_seen_frame, first_reappear_frame) tuples.
        """
        gaps = []
        frames = self._sorted_frames
        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] >= min_gap_frames:
                gaps.append((frames[i - 1], frames[i]))
        return gaps

    @property
    def frames(self) -> list[int]:
        """Sorted list of frames with ball detections."""
        return self._sorted_frames

    def __len__(self) -> int:
        return len(self._sorted_frames)


class BallTouchDetector:
    """
    Ball-first GK event detector.

    Pipeline:
    1. Build ball trajectory from all tracks
    2. Find touch moments: direction changes, speed changes, disappearances
    3. Attribute each touch to nearest player track
    4. Classify toucher by jersey color → keeper event or highlights
    5. Merge nearby events
    """

    def __init__(
        self,
        job_id: str,
        source_file: str,
        match_config: Optional[MatchConfig] = None,
        detect_distributions: bool = False,
        min_touch_speed: float = 0.20,
        direction_change_deg: float = 40.0,
        speed_change_ratio: float = 0.50,
        max_player_distance: float = 0.12,
        gk_color_min_similarity: float = 0.60,
        # WS1: Near-goal relaxed thresholds
        near_goal_x_threshold: float = 0.15,
        near_goal_direction_change_deg: float = 25.0,
        near_goal_speed_change_ratio: float = 0.30,
    ):
        self.job_id = job_id
        self.source_file = source_file
        self._match_config = match_config
        self._detect_distributions = detect_distributions
        self._min_touch_speed = min_touch_speed
        self._direction_change_deg = direction_change_deg
        self._speed_change_ratio = speed_change_ratio
        self._max_player_distance = max_player_distance
        self._gk_color_min_similarity = gk_color_min_similarity

        # Near-goal zone thresholds (relaxed for deflections/saves)
        self._near_goal_x = near_goal_x_threshold
        self._near_goal_speed_change_ratio = near_goal_speed_change_ratio

        # Pre-compute direction change thresholds as cosine
        self._cos_threshold = math.cos(math.radians(direction_change_deg))
        self._near_goal_cos_threshold = math.cos(math.radians(near_goal_direction_change_deg))

    def detect_touches(self, tracks: list[Track], fps: float) -> list[Event]:
        """
        Main entry point. Detect ball touch events from tracked data.

        Returns list of Events with reel_targets=["keeper"] for GK touches,
        or reel_targets=["highlights"] for outfield shots.
        """
        # Step 1: Build ball trajectory
        trajectory = BallTrajectory().build(tracks)
        if len(trajectory) < 4:
            log.debug(
                "ball_touch.insufficient_ball_data",
                ball_frames=len(trajectory),
            )
            return []

        # Step 2: Find touch moments
        touch_frames = self._find_touch_frames(trajectory, fps)
        if not touch_frames:
            log.debug("ball_touch.no_touches_found")
            return []

        # Step 3 + 4: Attribute and classify
        events = self._attribute_and_classify(
            touch_frames, trajectory, tracks, fps,
        )

        # Step 5: Smart endpoint extension (WS3)
        events = self._apply_smart_endpoints(events, trajectory, tracks, fps)

        # Step 6: Ball-in-net detection (WS1)
        goal_events = self._detect_ball_in_net(trajectory, tracks, fps)
        events.extend(goal_events)

        # Step 7: Corner kick detection (WS2)
        # Pass save events to enable post-save restart inference
        save_events = [
            e for e in events
            if e.event_type in (
                EventType.SHOT_STOP_STANDING,
                EventType.SHOT_STOP_DIVING,
                EventType.CATCH,
            )
        ]
        corner_events = self._detect_corner_kicks(
            trajectory, tracks, fps, save_events=save_events,
        )
        events.extend(corner_events)

        # Step 8: Merge nearby
        events = self._merge_nearby_events(events, min_gap_sec=2.0)

        log.info(
            "ball_touch.detection_complete",
            ball_frames=len(trajectory),
            touch_candidates=len(touch_frames),
            events_produced=len(events),
        )
        return events

    def _find_touch_frames(
        self, trajectory: BallTrajectory, fps: float,
    ) -> list[tuple[int, str]]:
        """
        Scan trajectory for moments where ball behavior changes.

        Uses per-frame velocity comparison: for each frame, compare a short
        pre-window velocity to a short post-window velocity.  Only the frame
        with the sharpest change in each neighborhood is kept (NMS).

        Returns list of (frame, reason) tuples.
        """
        candidates: list[tuple[int, str, float]] = []  # (frame, reason, score)
        frames = trajectory.frames

        # Direction and speed changes — compare local pre/post velocities
        half_win = 3  # frames on each side for velocity estimation
        for i, frame in enumerate(frames):
            # Need at least half_win frames on each side
            if i < half_win or i >= len(frames) - half_win:
                continue
            reason, score = self._check_touch_at_frame(
                frame, i, trajectory, half_win,
            )
            if reason is not None:
                candidates.append((frame, reason, score))

        # Gap-based: ball caught (disappears) or distribution (appears)
        for last_seen, first_reappear in trajectory.find_gaps(min_gap_frames=5):
            speed_before = trajectory.speed_at(last_seen, window=5)
            if speed_before is not None and speed_before > self._min_touch_speed:
                candidates.append((last_seen, "ball_caught", speed_before))

            if self._detect_distributions:
                speed_after = trajectory.speed_at(first_reappear, window=5)
                if speed_after is not None and speed_after > self._min_touch_speed:
                    candidates.append((first_reappear, "ball_released", speed_after))

        # End-of-trajectory catch: ball moving fast then trajectory ends
        # (no reappearance within chunk = caught / held)
        if len(frames) >= 4:
            last_frame = frames[-1]
            speed_end = trajectory.speed_at(last_frame, window=min(5, len(frames)))
            if speed_end is not None and speed_end > self._min_touch_speed:
                # Check if the ball was actually moving fast in the last few frames
                # (not just a static ball at the end)
                if not any(f == last_frame and r == "ball_caught" for f, r, _ in candidates):
                    candidates.append((last_frame, "ball_caught", speed_end))

        # Sort by frame
        candidates.sort(key=lambda c: c[0])

        # NMS: greedy, keep highest score within each window
        nms_window = max(10, int(fps / 3))
        candidates_by_score = sorted(candidates, key=lambda c: c[2], reverse=True)
        suppressed_frames: set[int] = set()
        result: list[tuple[int, str]] = []
        for frame, reason, score in candidates_by_score:
            if any(abs(frame - sf) < nms_window for sf in suppressed_frames):
                continue
            result.append((frame, reason))
            suppressed_frames.add(frame)
        result.sort(key=lambda c: c[0])

        return result

    def _check_touch_at_frame(
        self, frame: int, idx: int, trajectory: BallTrajectory, half_win: int,
    ) -> tuple[Optional[str], float]:
        """
        Check if ball trajectory changes at this frame by comparing
        local pre-window velocity to local post-window velocity.

        Uses zone-aware thresholds: near the goal (x < 0.15 or x > 0.85)
        thresholds are relaxed to catch deflections and saves on slower balls.
        """
        frames = trajectory.frames

        # Pre-velocity: from frames[idx-half_win] to frames[idx-1]
        pre_start = frames[idx - half_win]
        pre_end = frames[idx - 1] if idx >= 1 else frames[idx]
        # Post-velocity: from frames[idx+1] to frames[idx+half_win]
        post_start = frames[idx + 1] if idx + 1 < len(frames) else frames[idx]
        post_end = frames[idx + half_win] if idx + half_win < len(frames) else frames[-1]

        pre_vel = self._velocity_between(trajectory, pre_start, pre_end)
        post_vel = self._velocity_between(trajectory, post_start, post_end)

        if pre_vel is None or post_vel is None:
            return None, 0.0

        pre_vx, pre_vy = pre_vel
        post_vx, post_vy = post_vel
        pre_speed = (pre_vx ** 2 + pre_vy ** 2) ** 0.5
        post_speed = (post_vx ** 2 + post_vy ** 2) ** 0.5

        # Determine if ball is near goal — use relaxed thresholds
        ball_pos = trajectory.position_at(frame)
        near_goal = False
        if ball_pos is not None:
            bx = ball_pos[0]
            near_goal = bx < self._near_goal_x or bx > (1.0 - self._near_goal_x)

        # Select thresholds based on zone
        speed_ratio = self._near_goal_speed_change_ratio if near_goal else self._speed_change_ratio
        cos_thresh = self._near_goal_cos_threshold if near_goal else self._cos_threshold
        min_speed = self._min_touch_speed * 0.5 if near_goal else self._min_touch_speed

        # Ignore if ball is too slow before contact
        if pre_speed < min_speed:
            return None, 0.0

        # Speed drop: post speed < (1 - ratio) * pre speed
        if pre_speed > 0 and post_speed < pre_speed * (1.0 - speed_ratio):
            return "speed_drop", pre_speed

        # Speed spike: post >> pre (kick)
        if post_speed > pre_speed * (1.0 + speed_ratio) and post_speed > min_speed:
            if self._detect_distributions:
                return "speed_spike", post_speed
            return None, 0.0

        # Direction change
        if pre_speed > min_speed * 0.5 and post_speed > min_speed * 0.5:
            dot = pre_vx * post_vx + pre_vy * post_vy
            denom = pre_speed * post_speed
            if denom > 0:
                cos_angle = max(-1.0, min(1.0, dot / denom))
                if cos_angle < cos_thresh:
                    return "direction_change", pre_speed

        return None, 0.0

    @staticmethod
    def _velocity_between(
        trajectory: BallTrajectory, f_start: int, f_end: int,
    ) -> Optional[tuple[float, float]]:
        """Compute velocity vector between two frames."""
        if f_start == f_end:
            return None
        t_start = trajectory.timestamp_at(f_start)
        t_end = trajectory.timestamp_at(f_end)
        if t_start is None or t_end is None:
            return None
        dt = t_end - t_start
        if dt <= 0:
            return None
        pos_start = trajectory.position_at(f_start)
        pos_end = trajectory.position_at(f_end)
        if pos_start is None or pos_end is None:
            return None
        return (
            (pos_end[0] - pos_start[0]) / dt,
            (pos_end[1] - pos_start[1]) / dt,
        )

    def _attribute_and_classify(
        self,
        touch_frames: list[tuple[int, str]],
        trajectory: BallTrajectory,
        tracks: list[Track],
        fps: float,
    ) -> list[Event]:
        """
        For each touch frame, find the nearest player and classify by jersey.
        Then post-filter: reclassify dead-ball collections as goal kicks.
        """
        from src.detection.jersey_classifier import (
            compute_jersey_similarity,
            resolve_jersey_color,
        )

        events: list[Event] = []

        # Pre-resolve GK colors
        team_gk_hsv = None
        opp_gk_hsv = None
        team_outfield_hsv = None
        opp_outfield_hsv = None
        if self._match_config is not None:
            team_gk_hsv = resolve_jersey_color(self._match_config.team.gk_color)
            opp_gk_hsv = resolve_jersey_color(self._match_config.opponent.gk_color)
            team_outfield_hsv = resolve_jersey_color(self._match_config.team.outfield_color)
            opp_outfield_hsv = resolve_jersey_color(self._match_config.opponent.outfield_color)

        # Build frame-indexed player detection lookup
        # frame → list[(track_id, cx, cy, jersey_color_hsv)]
        player_by_frame: dict[int, list[tuple[int, float, float, Optional[tuple]]]] = {}
        for track in tracks:
            for det in track.detections:
                if det.class_name in ("player", "goalkeeper"):
                    player_by_frame.setdefault(det.frame_number, []).append((
                        track.track_id,
                        det.bbox.center_x,
                        det.bbox.center_y,
                        track.jersey_color_hsv,
                    ))

        for touch_frame, reason in touch_frames:
            ball_pos = trajectory.position_at(touch_frame)
            ball_ts = trajectory.timestamp_at(touch_frame)
            if ball_pos is None or ball_ts is None:
                continue

            bx, by = ball_pos

            # Find nearest player within a frame window around the touch.
            # Search backward more aggressively (10 frames ≈ 0.33s at 30fps)
            # since direction changes are detected after deflection.
            best_player = None
            best_dist = float("inf")
            pre_tolerance = 10   # frames to search backward
            post_tolerance = 2   # frames to search forward

            # Also collect all nearby players for penalty-area GK search
            all_nearby: list[tuple[float, int, float, float, tuple | None]] = []

            for f_offset in range(-pre_tolerance, post_tolerance + 1):
                f = touch_frame + f_offset
                for tid, px, py, jersey_hsv in player_by_frame.get(f, []):
                    dist = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
                    all_nearby.append((dist, tid, px, py, jersey_hsv))
                    if dist < best_dist:
                        best_dist = dist
                        best_player = (tid, px, py, jersey_hsv)

            if best_player is None or best_dist > self._max_player_distance:
                log.debug(
                    "ball_touch.no_nearby_player",
                    frame=touch_frame, reason=reason,
                    best_dist=round(best_dist, 4) if best_dist < float("inf") else None,
                )
                continue

            tid, px, py, jersey_hsv = best_player

            # GK override for save-like touches only: during a parry, the
            # ball deflects away so the nearest player is an attacker.
            # Only for speed_drop/direction_change (save indicators), scan
            # all nearby players for a GK-colored one in the goal area
            # (px < 0.22 or px > 0.78).  Uses PLAYER position, not ball
            # position, because auto-pan cameras center the ball.
            goal_area_override_fired = False
            _GOAL_AREA_X = 0.22
            _OVERRIDE_MIN_SIM = 0.55  # match base gk_color_min_similarity
            _OVERRIDE_MARGIN = 0.03   # GK must barely beat nearest
            _SAVE_LIKE_REASONS = ("speed_drop", "direction_change")
            if team_gk_hsv is not None and reason in _SAVE_LIKE_REASONS:
                best_gk_player = None
                best_gk_sim = 0.0
                # Log all goal-area candidates for diagnostics
                goal_area_candidates: list[tuple[float, int, float]] = []
                for d, t_id, p_x, p_y, j_hsv in all_nearby:
                    if j_hsv is None:
                        continue
                    # Player must be in a goal area (near a goal line)
                    if _GOAL_AREA_X <= p_x <= (1 - _GOAL_AREA_X):
                        continue
                    sim = compute_jersey_similarity(j_hsv, team_gk_hsv)
                    goal_area_candidates.append((sim, t_id, p_x))
                    if sim >= _OVERRIDE_MIN_SIM and sim > best_gk_sim:
                        best_gk_sim = sim
                        best_gk_player = (t_id, p_x, p_y, j_hsv)
                if goal_area_candidates:
                    log.debug(
                        "ball_touch.goal_area_candidates",
                        frame=touch_frame,
                        reason=reason,
                        candidates=[(round(s, 3), t, round(x, 3)) for s, t, x in sorted(goal_area_candidates, reverse=True)[:5]],
                    )
                if best_gk_player is not None:
                    nearest_sim = compute_jersey_similarity(
                        jersey_hsv, team_gk_hsv
                    ) if jersey_hsv is not None else 0.0
                    if best_gk_sim > nearest_sim + _OVERRIDE_MARGIN:
                        old_tid = tid
                        tid, px, py, jersey_hsv = best_gk_player
                        goal_area_override_fired = True
                        log.debug(
                            "ball_touch.goal_area_gk_override",
                            frame=touch_frame,
                            old_track=old_tid,
                            new_track=tid,
                            gk_sim=round(best_gk_sim, 3),
                            nearest_sim=round(nearest_sim, 3),
                            gk_px=round(best_gk_player[1], 3),
                        )

            # Compute GK vertical velocity for diving save detection
            gk_vert_vel = self._compute_player_vertical_velocity(
                tid, touch_frame, tracks, fps,
            )

            # Classify toucher by jersey color.
            # When goal-area override fired, relax color margins — the
            # player was already validated by position + trajectory.
            event = self._classify_touch(
                touch_frame=touch_frame,
                timestamp=ball_ts,
                reason=reason,
                player_track_id=tid,
                player_jersey_hsv=jersey_hsv,
                ball_pos=(bx, by),
                player_pos=(px, py),
                fps=fps,
                team_gk_hsv=team_gk_hsv,
                opp_gk_hsv=opp_gk_hsv,
                team_outfield_hsv=team_outfield_hsv,
                opp_outfield_hsv=opp_outfield_hsv,
                gk_vertical_velocity=gk_vert_vel,
                goal_area_override=goal_area_override_fired,
            )
            if event is not None:
                events.append(event)

        # Post-filter: reclassify dead-ball collections → goal kicks
        events = self._reclassify_dead_ball_collections(events, trajectory, fps)

        return events

    def _reclassify_dead_ball_collections(
        self,
        events: list[Event],
        trajectory: BallTrajectory,
        fps: float,
    ) -> list[Event]:
        """
        Detect dead-ball → goal kick sequences and reclassify.

        Pattern: GK collects ball (save/catch) → ball stationary for >1s →
        ball kicked (speed spike).  The collection is dead time, not a save.
        Replace with a GOAL_KICK event at the kick moment.

        Also suppresses save/catch events where the ball was already slow
        before the GK touched it (dead-ball retrieval, not a shot).
        """
        if not events:
            return events

        result: list[Event] = []
        frames = trajectory.frames

        # Minimum stationary duration to qualify as "placed for goal kick"
        min_stationary_sec = 1.0
        # Maximum speed to be considered stationary (normalized coords/sec)
        stationary_speed = 0.05
        # How far ahead to look for the kick (seconds)
        lookahead_sec = 10.0

        for event in events:
            reason = event.metadata.get("touch_reason", "")

            # Only check save/catch events for dead-ball pattern
            if event.event_type not in (
                EventType.SHOT_STOP_STANDING,
                EventType.SHOT_STOP_DIVING,
                EventType.CATCH,
            ):
                result.append(event)
                continue

            touch_frame = int(event.frame_end)  # approximate touch frame
            touch_ts = event.timestamp_end
            lookahead_end = touch_ts + lookahead_sec

            # Scan trajectory after the touch for stationary period → kick
            goal_kick = self._find_goal_kick_after(
                trajectory, frames, fps, touch_ts, lookahead_end,
                stationary_speed, min_stationary_sec,
            )

            if goal_kick is not None:
                kick_ts, kick_frame = goal_kick

                # Goal kicks require higher GK confidence — the dead-ball→kick
                # pattern can happen anywhere on the pitch (e.g. defender
                # collecting a clearance), so we need a stricter threshold
                # than normal GK classification to prevent FPs.
                _GK_GOAL_KICK_MIN_SIM = 0.78
                orig_sim = event.metadata.get("sim_team_gk", 0)
                if orig_sim < _GK_GOAL_KICK_MIN_SIM:
                    log.debug(
                        "ball_touch.goal_kick_sim_too_low",
                        sim_team_gk=round(orig_sim, 3),
                        threshold=_GK_GOAL_KICK_MIN_SIM,
                        frame=touch_frame,
                    )
                    result.append(event)
                    continue

                log.info(
                    "ball_touch.dead_ball_reclassified",
                    original_type=event.event_type.value,
                    original_frame=touch_frame,
                    kick_frame=kick_frame,
                    kick_ts=round(kick_ts, 1),
                )
                # Replace save/catch with GOAL_KICK at kick moment
                # Pre-window of 1.5s captures ball placement without too much idle time
                result.append(Event(
                    job_id=event.job_id,
                    source_file=event.source_file,
                    event_type=EventType.GOAL_KICK,
                    timestamp_start=max(0, kick_ts - 1.5),
                    timestamp_end=kick_ts + 2.0,
                    confidence=event.confidence,
                    reel_targets=[],
                    player_track_id=event.player_track_id,
                    is_goalkeeper_event=True,
                    frame_start=max(0, int((kick_ts - 1.5) * fps)),
                    frame_end=int((kick_ts + 2.0) * fps),
                    bounding_box=event.bounding_box,
                    metadata={
                        "detection_method": "ball_touch",
                        "touch_reason": "goal_kick",
                        "original_reason": reason,
                        "sim_team_gk": event.metadata.get("sim_team_gk"),
                        "player_track_id": event.player_track_id,
                    },
                ))
            else:
                result.append(event)

        return result

    @staticmethod
    def _find_goal_kick_after(
        trajectory: BallTrajectory,
        frames: list[int],
        fps: float,
        touch_ts: float,
        lookahead_end: float,
        stationary_speed: float,
        min_stationary_sec: float,
    ) -> Optional[tuple[float, int]]:
        """
        Look for a stationary-then-kick pattern after a touch.

        Returns (kick_timestamp, kick_frame) if found, else None.
        """
        # Find frames in the lookahead window
        post_frames = [
            f for f in frames
            if trajectory.timestamp_at(f) is not None
            and touch_ts < trajectory.timestamp_at(f) <= lookahead_end
        ]

        if len(post_frames) < 5:
            return None

        # Find a stationary period: consecutive frames with low speed
        stationary_start_ts = None
        stationary_frames = 0

        for i, frame in enumerate(post_frames):
            speed = trajectory.speed_at(frame, window=3)
            if speed is None:
                continue

            if speed < stationary_speed:
                if stationary_start_ts is None:
                    stationary_start_ts = trajectory.timestamp_at(frame)
                stationary_frames += 1
            else:
                # Ball moving again — was it stationary long enough?
                if (stationary_start_ts is not None
                        and stationary_frames >= 3):
                    current_ts = trajectory.timestamp_at(frame)
                    if current_ts - stationary_start_ts >= min_stationary_sec:
                        # Ball was placed, now kicked — this is the goal kick
                        return (current_ts, frame)
                # Reset
                stationary_start_ts = None
                stationary_frames = 0

        return None

    @staticmethod
    def _compute_player_vertical_velocity(
        track_id: int,
        touch_frame: int,
        tracks: list[Track],
        fps: float,
        window_frames: int = 10,
    ) -> float:
        """
        Compute a player's vertical velocity in body-heights per second.

        Looks at the player's bounding box center_y movement over a window
        around the touch frame. Returns absolute vertical velocity normalized
        by bbox height (body-heights/sec). Returns 0.0 if insufficient data.
        """
        # Find the track
        target_track = None
        for t in tracks:
            if t.track_id == track_id:
                target_track = t
                break
        if target_track is None:
            return 0.0

        # Get detections near the touch frame
        nearby = []
        for det in target_track.detections:
            if abs(det.frame_number - touch_frame) <= window_frames:
                nearby.append(det)

        if len(nearby) < 2:
            return 0.0

        nearby.sort(key=lambda d: d.frame_number)

        # Use the first and last detections in the window
        first = nearby[0]
        last = nearby[-1]

        frame_diff = last.frame_number - first.frame_number
        if frame_diff <= 0 or fps <= 0:
            return 0.0

        dt = frame_diff / fps
        dy = abs(last.bbox.center_y - first.bbox.center_y)

        # Normalize by average bbox height (body-height)
        avg_height = (first.bbox.height + last.bbox.height) / 2
        if avg_height <= 0:
            return 0.0

        return (dy / avg_height) / dt

    def _classify_touch(
        self,
        touch_frame: int,
        timestamp: float,
        reason: str,
        player_track_id: int,
        player_jersey_hsv: Optional[tuple[float, float, float]],
        ball_pos: tuple[float, float],
        player_pos: tuple[float, float],
        fps: float,
        team_gk_hsv: Optional[tuple[float, float, float]],
        opp_gk_hsv: Optional[tuple[float, float, float]],
        team_outfield_hsv: Optional[tuple[float, float, float]],
        opp_outfield_hsv: Optional[tuple[float, float, float]],
        gk_vertical_velocity: float = 0.0,
        goal_area_override: bool = False,
    ) -> Optional[Event]:
        """
        Classify a touch based on the toucher's jersey color.

        Team GK → keeper event
        Opponent GK → skip
        Outfield → skip (highlights shots handled by classify_highlights_events)
        No color data → skip

        When ``goal_area_override`` is True the player was already found in a
        goal area during a save-like ball touch — relax color margins since
        positional + trajectory evidence is strong.
        """
        from src.detection.jersey_classifier import compute_jersey_similarity

        if player_jersey_hsv is None or team_gk_hsv is None:
            log.debug(
                "ball_touch.no_jersey_data",
                frame=touch_frame, reason=reason,
                track_id=player_track_id,
            )
            return None

        sim_team_gk = compute_jersey_similarity(player_jersey_hsv, team_gk_hsv)
        sim_opp_gk = compute_jersey_similarity(player_jersey_hsv, opp_gk_hsv) if opp_gk_hsv else 0.0
        sim_team_of = compute_jersey_similarity(player_jersey_hsv, team_outfield_hsv) if team_outfield_hsv else 0.0
        sim_opp_of = compute_jersey_similarity(player_jersey_hsv, opp_outfield_hsv) if opp_outfield_hsv else 0.0

        # Team GK: highest similarity must be to team GK, above threshold,
        # and with a margin over every other color to prevent blue/teal confusion.
        # Different margins for GK-vs-GK (small — position handles confusion)
        # and GK-vs-outfield (large — critical for blue/teal guard).
        _COLOR_MARGIN_GK = 0.03   # GK vs opponent GK
        _COLOR_MARGIN_OF = 0.12   # GK vs any outfield color

        if goal_area_override:
            # Player validated by position + trajectory — only require
            # sim_team_gk >= threshold and > sim_opp_gk (no outfield margin).
            is_team_gk = (
                sim_team_gk >= self._gk_color_min_similarity
                and sim_team_gk > sim_opp_gk
            )
        else:
            is_team_gk = (
                sim_team_gk >= self._gk_color_min_similarity
                and sim_team_gk > sim_opp_gk + _COLOR_MARGIN_GK
                and sim_team_gk > sim_team_of + _COLOR_MARGIN_OF
                and sim_team_gk > sim_opp_of + _COLOR_MARGIN_OF
            )

        # GKs don't operate in midfield — reject to prevent FPs
        if is_team_gk and 0.30 < player_pos[0] < 0.70:
            log.debug(
                "ball_touch.midfield_gk_rejection",
                frame=touch_frame, px=round(player_pos[0], 3),
            )
            is_team_gk = False

        # Opponent GK: skip — we don't produce reels for them.
        # Require same margin as team_gk to avoid ambiguous classification.
        is_opp_gk = (
            sim_opp_gk >= self._gk_color_min_similarity
            and sim_opp_gk > sim_team_gk + _COLOR_MARGIN_GK
            and sim_opp_gk > sim_team_of
            and sim_opp_gk > sim_opp_of
        )

        # Goal-area fallback: when both GK checks fail (ambiguous colors)
        # but the player is near a goal line and sim_team_gk is highest,
        # classify as our GK.  Spatial filter handles any residual FPs.
        if not is_team_gk and not is_opp_gk:
            in_goal_area = player_pos[0] < 0.15 or player_pos[0] > 0.85
            if (
                in_goal_area
                and sim_team_gk >= self._gk_color_min_similarity
                and sim_team_gk >= sim_opp_gk
            ):
                is_team_gk = True
                log.debug(
                    "ball_touch.goal_area_gk_fallback",
                    frame=touch_frame,
                    px=round(player_pos[0], 3),
                    sim_team_gk=round(sim_team_gk, 3),
                    sim_opp_gk=round(sim_opp_gk, 3),
                )

        if is_opp_gk:
            log.debug(
                "ball_touch.opponent_gk_touch",
                frame=touch_frame, reason=reason,
                sim_opp_gk=round(sim_opp_gk, 3),
                sim_team_gk=round(sim_team_gk, 3),
                px=round(player_pos[0], 3),
            )
            return None

        if not is_team_gk:
            # Outfield player touch — not a keeper event
            log.debug(
                "ball_touch.outfield_touch",
                frame=touch_frame, reason=reason,
                track_id=player_track_id,
                sim_team_gk=round(sim_team_gk, 3),
                sim_opp_gk=round(sim_opp_gk, 3),
                sim_team_of=round(sim_team_of, 3),
                sim_opp_of=round(sim_opp_of, 3),
                goal_area_override=goal_area_override,
            )
            return None

        # Team GK touch → keeper event
        event_type = self._reason_to_event_type(reason)

        # Upgrade standing save to diving save based on GK vertical velocity
        _DIVE_VELOCITY_THRESHOLD = 2.0  # body-heights per second
        if (
            event_type == EventType.SHOT_STOP_STANDING
            and gk_vertical_velocity > _DIVE_VELOCITY_THRESHOLD
        ):
            event_type = EventType.SHOT_STOP_DIVING
            log.debug(
                "ball_touch.upgraded_to_diving",
                frame=touch_frame,
                gk_vertical_velocity=round(gk_vertical_velocity, 2),
            )

        confidence = min(0.90, 0.60 + sim_team_gk * 0.3)

        event = Event(
            job_id=self.job_id,
            source_file=self.source_file,
            event_type=event_type,
            timestamp_start=max(0, timestamp - 0.5),
            timestamp_end=timestamp + 0.5,
            confidence=confidence,
            reel_targets=[],
            player_track_id=player_track_id,
            is_goalkeeper_event=True,
            frame_start=max(0, int((timestamp - 0.5) * fps)),
            frame_end=int((timestamp + 0.5) * fps),
            bounding_box=BoundingBox(
                x=ball_pos[0] - 0.02, y=ball_pos[1] - 0.02,
                width=0.04, height=0.04,
            ),
            metadata={
                "detection_method": "ball_touch",
                "touch_reason": reason,
                "sim_team_gk": round(sim_team_gk, 3),
                "sim_opp_gk": round(sim_opp_gk, 3),
                "sim_team_of": round(sim_team_of, 3),
                "sim_opp_of": round(sim_opp_of, 3),
                "player_track_id": player_track_id,
                "ball_x": round(ball_pos[0], 4),
                "ball_y": round(ball_pos[1], 4),
                "gk_vertical_velocity": round(gk_vertical_velocity, 3),
            },
        )

        log.info(
            "ball_touch.keeper_event",
            frame=touch_frame,
            event_type=event_type.value,
            reason=reason,
            sim_team_gk=round(sim_team_gk, 3),
            confidence=round(confidence, 3),
        )
        return event

    @staticmethod
    def _reason_to_event_type(reason: str) -> EventType:
        """Map touch reason to EventType."""
        mapping = {
            "ball_caught": EventType.CATCH,
            "speed_drop": EventType.SHOT_STOP_STANDING,
            "direction_change": EventType.SHOT_STOP_STANDING,
            "speed_spike": EventType.DISTRIBUTION_LONG,
            "ball_released": EventType.DISTRIBUTION_SHORT,
        }
        return mapping.get(reason, EventType.SHOT_STOP_STANDING)

    # -------------------------------------------------------------------
    # WS1: Ball-in-net detection
    # -------------------------------------------------------------------
    def _detect_ball_in_net(
        self,
        trajectory: BallTrajectory,
        tracks: list[Track],
        fps: float,
    ) -> list[Event]:
        """Detect goals by finding ball entering the net area.

        Heuristic: ball at x < 0.03 or x > 0.97, moving fast (speed > 0.30)
        toward that side, then disappears (gap >= 3 frames).
        """
        events: list[Event] = []
        frames = trajectory.frames
        if len(frames) < 4:
            return events

        for i, frame in enumerate(frames):
            pos = trajectory.position_at(frame)
            if pos is None:
                continue
            bx, by = pos

            # Ball must be at the extreme edge
            at_left_net = bx < 0.03
            at_right_net = bx > 0.97
            if not at_left_net and not at_right_net:
                continue

            # Must be moving fast toward that side
            vel = trajectory.velocity_at(frame, window=5)
            if vel is None:
                continue
            vx, vy = vel
            speed = (vx ** 2 + vy ** 2) ** 0.5
            if speed < 0.30:
                continue

            # Check direction: moving left into left net, or right into right net
            if at_left_net and vx > -0.05:
                continue
            if at_right_net and vx < 0.05:
                continue

            # Check for disappearance: gap of >= 3 frames after this
            has_gap = True
            for j in range(1, 4):
                check_frame = frame + j
                if trajectory.position_at(check_frame) is not None:
                    has_gap = False
                    break
            # Also count if this is the last few frames
            if i >= len(frames) - 3:
                has_gap = True

            if not has_gap:
                continue

            ts = trajectory.timestamp_at(frame)
            if ts is None:
                continue

            events.append(Event(
                job_id=self.job_id,
                source_file=self.source_file,
                event_type=EventType.GOAL,
                timestamp_start=max(0, ts - 2.0),
                timestamp_end=ts + 3.0,
                confidence=0.85,
                reel_targets=[],
                is_goalkeeper_event=False,
                frame_start=max(0, int((ts - 2.0) * fps)),
                frame_end=int((ts + 3.0) * fps),
                bounding_box=BoundingBox(
                    x=bx - 0.02, y=by - 0.02,
                    width=0.04, height=0.04,
                ),
                metadata={
                    "detection_method": "ball_in_net",
                    "ball_x": round(bx, 4),
                    "ball_speed": round(speed, 4),
                    "net_side": "left" if at_left_net else "right",
                },
            ))

        log.debug("ball_touch.ball_in_net", goals_found=len(events))
        return events

    # -------------------------------------------------------------------
    # WS2: Corner kick detection
    # -------------------------------------------------------------------
    def _detect_corner_kicks(
        self,
        trajectory: BallTrajectory,
        tracks: list[Track],
        fps: float,
        save_events: list[Event] | None = None,
    ) -> list[Event]:
        """Detect corner kicks using two methods:

        Method 1 (spatial): Ball near frame edge, stationary >= 0.5s, then
        kicked, with high player density.  Thresholds are wide to handle
        auto-panning cameras where ball at corner flag can appear at various
        frame positions.

        Method 2 (post-save restart): After a save event where ball goes out
        of play (trajectory gap), look for a restart pattern (ball reappears,
        stationary, kicked) with high player density.  This reliably detects
        corner kicks following GK deflections regardless of camera type.
        """
        events: list[Event] = []
        frames = trajectory.frames
        if len(frames) < 10:
            return events

        # Build frame-indexed player lookup for cluster check
        player_by_frame: dict[int, list[tuple[float, float]]] = {}
        for track in tracks:
            for det in track.detections:
                if det.class_name in ("player", "goalkeeper"):
                    player_by_frame.setdefault(det.frame_number, []).append(
                        (det.bbox.center_x, det.bbox.center_y)
                    )

        min_stationary_frames = max(3, int(0.5 * fps / 3))

        # --- Method 1: Spatial (disabled for auto-panning cameras) ---
        # Spatial detection produces too many false positives with auto-pan
        # cameras because any ball near a frame edge + players triggers it.
        # Kept as code for fixed-camera deployments but not called by default.
        used_timestamps: set[float] = set()

        # --- Method 2: Post-save restart inference ---
        if save_events:
            restart = self._detect_corners_post_save(
                save_events, trajectory, frames, player_by_frame,
                fps, min_stationary_frames, used_timestamps,
            )
            events.extend(restart)

        log.debug("ball_touch.corner_kicks", corners_found=len(events))
        return events

    def _detect_corners_spatial(
        self,
        trajectory: BallTrajectory,
        frames: list[int],
        player_by_frame: dict[int, list[tuple[float, float]]],
        fps: float,
        min_stationary_frames: int,
    ) -> list[Event]:
        """Spatial corner detection with relaxed thresholds for auto-pan cameras.

        Ball near frame edge (not necessarily in a corner) + stationary + kicked
        + high player density on one side of the frame.
        """
        events: list[Event] = []

        i = 0
        while i < len(frames) - min_stationary_frames:
            frame = frames[i]
            pos = trajectory.position_at(frame)
            if pos is None:
                i += 1
                continue
            bx, by = pos

            # Wide corner zone: ball near frame edge in at least one dimension.
            # With auto-panning cameras, the ball at a corner flag may appear
            # anywhere along the frame edges rather than at strict corners.
            near_x_edge = bx < 0.12 or bx > 0.88
            near_y_edge = by < 0.15 or by > 0.85
            # Require ball near an edge (either dimension) but NOT in center
            in_corner_zone = near_x_edge and near_y_edge
            # Also accept: ball near one edge if very close to that edge
            near_strong_edge = bx < 0.08 or bx > 0.92 or by < 0.10 or by > 0.90

            if not (in_corner_zone or near_strong_edge):
                i += 1
                continue

            result = self._check_stationary_and_kick(
                trajectory, frames, i, fps, min_stationary_frames,
            )
            if result is None:
                i += 1
                continue
            kick_frame, kick_speed, stationary_count = result

            # Player density check — need many players clustered on one side
            min_players = 3 if in_corner_zone else 5
            players_in_half = self._count_players_in_half(
                player_by_frame, kick_frame, bx,
            )
            if players_in_half < min_players:
                i += stationary_count + 1
                continue

            kick_ts = trajectory.timestamp_at(kick_frame)
            if kick_ts is None:
                i += stationary_count + 1
                continue

            events.append(self._make_corner_event(
                kick_ts, fps, bx, by, kick_speed, players_in_half,
                detection_method="corner_kick_spatial",
            ))
            i += stationary_count + int(fps)

        return events

    def _detect_corners_post_save(
        self,
        save_events: list[Event],
        trajectory: BallTrajectory,
        frames: list[int],
        player_by_frame: dict[int, list[tuple[float, float]]],
        fps: float,
        min_stationary_frames: int,
        used_timestamps: set[float],
    ) -> list[Event]:
        """Detect corner kicks following save events.

        Pattern: save → ball goes out (trajectory gap) → ball reappears →
        stationary (being placed for corner) → kicked with high player density.

        This captures corner kicks after GK deflections regardless of camera
        position, since we know a save happened and the ball went out.
        """
        events: list[Event] = []

        for save_ev in save_events:
            if save_ev.event_type not in (
                EventType.SHOT_STOP_STANDING,
                EventType.SHOT_STOP_DIVING,
                EventType.CATCH,
            ):
                continue

            save_end_ts = save_ev.timestamp_end
            save_frame = save_ev.frame_end

            # Look for a trajectory gap starting within 3s after the save
            # (ball going out of play after GK deflection)
            gap_found = False
            gap_end_frame = None
            for last_seen, first_reappear in trajectory.find_gaps(min_gap_frames=5):
                last_ts = trajectory.timestamp_at(last_seen)
                reappear_ts = trajectory.timestamp_at(first_reappear)
                if last_ts is None or reappear_ts is None:
                    continue
                # Gap must start near the save (within 1s before or 3s after)
                if -1.0 <= last_ts - save_end_ts <= 3.0:
                    # Gap must be at least 1s (real dead ball, not brief occlusion)
                    gap_duration = reappear_ts - last_ts
                    if gap_duration >= 1.0:
                        gap_found = True
                        gap_end_frame = first_reappear
                        break

            if not gap_found or gap_end_frame is None:
                continue

            # After the gap, look for stationary → kicked pattern
            gap_idx = bisect.bisect_left(frames, gap_end_frame)
            if gap_idx >= len(frames):
                continue

            # Scan forward from the gap end for stationary ball
            found_restart = False
            scan_limit = min(gap_idx + int(10 * fps), len(frames) - min_stationary_frames)
            for si in range(gap_idx, scan_limit):
                result = self._check_stationary_and_kick(
                    trajectory, frames, si, fps, min_stationary_frames,
                )
                if result is None:
                    continue
                kick_frame, kick_speed, stationary_count = result

                # Player density check — corner kicks have many players in frame
                # Use the ball's side to determine which half to check
                ball_pos = trajectory.position_at(kick_frame)
                if ball_pos is None:
                    continue
                bx, by = ball_pos

                players_in_half = self._count_players_in_half(
                    player_by_frame, kick_frame, bx,
                )
                if players_in_half < 4:
                    continue

                kick_ts = trajectory.timestamp_at(kick_frame)
                if kick_ts is None:
                    continue

                # Avoid duplicates with spatial method
                if any(abs(kick_ts - ut) < 3.0 for ut in used_timestamps):
                    continue

                events.append(self._make_corner_event(
                    kick_ts, fps, bx, by, kick_speed, players_in_half,
                    detection_method="corner_kick_post_save",
                    save_event_ts=round(save_ev.timestamp_start, 1),
                ))
                found_restart = True
                break

            if found_restart:
                log.info(
                    "ball_touch.corner_after_save",
                    save_ts=round(save_ev.timestamp_start, 1),
                    save_type=save_ev.event_type.value,
                )

        return events

    def _check_stationary_and_kick(
        self,
        trajectory: BallTrajectory,
        frames: list[int],
        start_idx: int,
        fps: float,
        min_stationary_frames: int,
    ) -> tuple[int, float, int] | None:
        """Check for stationary → kicked pattern starting at index.

        Returns (kick_frame, kick_speed, stationary_count) or None.
        """
        stationary_count = 0
        for j in range(start_idx, min(start_idx + int(3 * fps), len(frames))):
            speed = trajectory.speed_at(frames[j], window=3)
            if speed is not None and speed < 0.08:
                stationary_count += 1
            else:
                break

        if stationary_count < min_stationary_frames:
            return None

        kick_idx = start_idx + stationary_count
        if kick_idx >= len(frames):
            return None

        kick_frame = frames[kick_idx]
        kick_speed = trajectory.speed_at(kick_frame, window=3)
        if kick_speed is None or kick_speed < 0.12:
            return None

        return (kick_frame, kick_speed, stationary_count)

    def _count_players_in_half(
        self,
        player_by_frame: dict[int, list[tuple[float, float]]],
        frame: int,
        ball_x: float,
    ) -> int:
        """Count players in the same half of the frame as the ball."""
        if ball_x < 0.5:
            return sum(
                1 for px, py in player_by_frame.get(frame, [])
                if px < 0.55
            )
        else:
            return sum(
                1 for px, py in player_by_frame.get(frame, [])
                if px > 0.45
            )

    def _make_corner_event(
        self,
        kick_ts: float,
        fps: float,
        bx: float,
        by: float,
        kick_speed: float,
        players_in_half: int,
        detection_method: str = "corner_kick",
        save_event_ts: float | None = None,
    ) -> Event:
        """Create a CORNER_KICK event."""
        metadata: dict = {
            "detection_method": detection_method,
            "corner_x": round(bx, 4),
            "corner_y": round(by, 4),
            "kick_speed": round(kick_speed, 4),
            "players_in_half": players_in_half,
        }
        if save_event_ts is not None:
            metadata["save_event_ts"] = save_event_ts

        return Event(
            job_id=self.job_id,
            source_file=self.source_file,
            event_type=EventType.CORNER_KICK,
            timestamp_start=max(0, kick_ts - 1.0),
            timestamp_end=kick_ts + 4.0,
            confidence=0.70,
            reel_targets=[],
            is_goalkeeper_event=True,
            frame_start=max(0, int((kick_ts - 1.0) * fps)),
            frame_end=int((kick_ts + 4.0) * fps),
            bounding_box=BoundingBox(
                x=bx - 0.02, y=by - 0.02,
                width=0.04, height=0.04,
            ),
            metadata=metadata,
        )

    # -------------------------------------------------------------------
    # WS3: Smart clip endpoints
    # -------------------------------------------------------------------

    # Per-type max extension for smart endpoints (seconds)
    _SMART_ENDPOINT_TYPES: dict[EventType, float] = {
        EventType.DISTRIBUTION_SHORT: 10.0,
        EventType.DISTRIBUTION_LONG: 10.0,
        EventType.GOAL_KICK: 12.0,
        EventType.SHOT_STOP_STANDING: 5.0,
        EventType.SHOT_STOP_DIVING: 5.0,
        EventType.CATCH: 5.0,
        EventType.PUNCH: 5.0,
        EventType.CORNER_KICK: 8.0,
        EventType.ONE_ON_ONE: 8.0,
    }

    def _apply_smart_endpoints(
        self,
        events: list[Event],
        trajectory: BallTrajectory,
        tracks: list[Track],
        fps: float,
    ) -> list[Event]:
        """Extend event endpoints to next touch, out-of-bounds, or disappearance."""
        result = []
        for event in events:
            max_ext = self._SMART_ENDPOINT_TYPES.get(event.event_type)
            if max_ext is None:
                result.append(event)
                continue
            extended = self._extend_event_endpoint(
                event, trajectory, tracks, fps, max_ext,
            )
            result.append(extended)
        return result

    def _extend_event_endpoint(
        self,
        event: Event,
        trajectory: BallTrajectory,
        tracks: list[Track],
        fps: float,
        max_extension_sec: float,
    ) -> Event:
        """Extend event's timestamp_end to next touch / out-of-bounds / disappearance.

        Scans ball trajectory forward from event's timestamp_end for:
        - Out of bounds: x<0.01 or x>0.99 or y<0.01 or y>0.99
        - Next touch: direction change > 30deg or speed drop > 40%
        - Ball disappears: gap >= 5 frames
        - Max extension reached
        """
        frames = trajectory.frames
        if not frames:
            return event

        end_ts = event.timestamp_end
        max_ts = end_ts + max_extension_sec

        # Find starting frame index
        start_idx = None
        for i, f in enumerate(frames):
            ts = trajectory.timestamp_at(f)
            if ts is not None and ts >= end_ts:
                start_idx = i
                break
        if start_idx is None:
            return event

        new_end_ts = end_ts
        reason = None

        cos_20 = math.cos(math.radians(20.0))

        for i in range(start_idx, len(frames)):
            frame = frames[i]
            ts = trajectory.timestamp_at(frame)
            if ts is None or ts > max_ts:
                new_end_ts = min(ts or max_ts, max_ts)
                reason = "max_extension"
                break

            pos = trajectory.position_at(frame)
            if pos is None:
                continue

            bx, by = pos

            # Out of bounds
            if bx < 0.01 or bx > 0.99 or by < 0.01 or by > 0.99:
                new_end_ts = ts
                reason = "out_of_bounds"
                break

            # Gap detection: check if next frame has a gap >= 5
            if i + 1 < len(frames) and frames[i + 1] - frame >= 5:
                new_end_ts = ts + 0.5  # small buffer after last seen
                reason = "ball_disappeared"
                break

            # Direction/speed change (next touch by another player)
            if i >= start_idx + 2:  # need a few frames for velocity
                pre_vel = self._velocity_between(trajectory, frames[i - 2], frames[i - 1])
                post_vel = self._velocity_between(trajectory, frames[i - 1], frame)
                if pre_vel is not None and post_vel is not None:
                    pre_s = (pre_vel[0] ** 2 + pre_vel[1] ** 2) ** 0.5
                    post_s = (post_vel[0] ** 2 + post_vel[1] ** 2) ** 0.5
                    # Speed drop > 30%
                    if pre_s > 0.1 and post_s < pre_s * 0.7:
                        new_end_ts = ts
                        reason = "next_touch_speed"
                        break
                    # Direction change > 20deg
                    if pre_s > 0.03 and post_s > 0.03:
                        dot = pre_vel[0] * post_vel[0] + pre_vel[1] * post_vel[1]
                        denom = pre_s * post_s
                        if denom > 0:
                            cos_a = max(-1.0, min(1.0, dot / denom))
                            if cos_a < cos_20:
                                new_end_ts = ts
                                reason = "next_touch_direction"
                                break
        else:
            # Reached end of trajectory
            if frames:
                last_ts = trajectory.timestamp_at(frames[-1])
                if last_ts is not None:
                    new_end_ts = min(last_ts, max_ts)
                    reason = "trajectory_end"

        if new_end_ts > end_ts:
            data = event.model_dump()
            data["timestamp_end"] = new_end_ts
            data["frame_end"] = int(new_end_ts * fps)
            data["metadata"]["endpoint_extended"] = True
            data["metadata"]["endpoint_reason"] = reason
            data["metadata"]["extension_sec"] = round(new_end_ts - end_ts, 2)
            return Event(**data)

        log.debug(
            "ball_touch.endpoint_not_extended",
            event_type=event.event_type.value,
            timestamp_end=round(end_ts, 2),
        )
        return event

    @staticmethod
    def _merge_nearby_events(events: list[Event], min_gap_sec: float) -> list[Event]:
        """Merge events that are too close together (likely the same action)."""
        if len(events) <= 1:
            return events

        events = sorted(events, key=lambda e: e.timestamp_start)
        merged = [events[0]]
        for event in events[1:]:
            prev = merged[-1]
            if event.timestamp_start - prev.timestamp_end < min_gap_sec:
                data = prev.model_dump()
                data["timestamp_end"] = max(prev.timestamp_end, event.timestamp_end)
                data["frame_end"] = max(prev.frame_end, event.frame_end)
                data["confidence"] = max(prev.confidence, event.confidence)
                merged[-1] = Event(**data)
            else:
                merged.append(event)
        return merged
