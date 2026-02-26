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
    BoundingBox, Detection, Event, EventType, Track, EVENT_REEL_MAP,
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
        gk_color_min_similarity: float = 0.55,
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

        # Pre-compute direction change threshold as cosine
        self._cos_threshold = math.cos(math.radians(direction_change_deg))

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

        # Step 5: Merge nearby
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

        Uses tight windows immediately adjacent to the frame so the
        detection localizes precisely at the change point.
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

        # Ignore if ball is too slow before contact
        if pre_speed < self._min_touch_speed:
            return None, 0.0

        # Speed drop: post speed < (1 - ratio) * pre speed
        if pre_speed > 0 and post_speed < pre_speed * (1.0 - self._speed_change_ratio):
            return "speed_drop", pre_speed

        # Speed spike: post >> pre (kick)
        if post_speed > pre_speed * (1.0 + self._speed_change_ratio) and post_speed > self._min_touch_speed:
            if self._detect_distributions:
                return "speed_spike", post_speed
            return None, 0.0

        # Direction change
        if pre_speed > self._min_touch_speed * 0.5 and post_speed > self._min_touch_speed * 0.5:
            dot = pre_vx * post_vx + pre_vy * post_vy
            denom = pre_speed * post_speed
            if denom > 0:
                cos_angle = max(-1.0, min(1.0, dot / denom))
                if cos_angle < self._cos_threshold:
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

            # Find nearest player within ±2 frames tolerance
            best_player = None
            best_dist = float("inf")
            frame_tolerance = 2

            for f_offset in range(-frame_tolerance, frame_tolerance + 1):
                f = touch_frame + f_offset
                for tid, px, py, jersey_hsv in player_by_frame.get(f, []):
                    dist = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
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

            # Classify toucher by jersey color
            event = self._classify_touch(
                touch_frame=touch_frame,
                timestamp=ball_ts,
                reason=reason,
                player_track_id=tid,
                player_jersey_hsv=jersey_hsv,
                ball_pos=(bx, by),
                fps=fps,
                team_gk_hsv=team_gk_hsv,
                opp_gk_hsv=opp_gk_hsv,
                team_outfield_hsv=team_outfield_hsv,
                opp_outfield_hsv=opp_outfield_hsv,
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
                log.info(
                    "ball_touch.dead_ball_reclassified",
                    original_type=event.event_type.value,
                    original_frame=touch_frame,
                    kick_frame=kick_frame,
                    kick_ts=round(kick_ts, 1),
                )
                # Replace save/catch with GOAL_KICK at kick moment
                result.append(Event(
                    job_id=event.job_id,
                    source_file=event.source_file,
                    event_type=EventType.GOAL_KICK,
                    timestamp_start=max(0, kick_ts - 0.5),
                    timestamp_end=kick_ts + 2.0,
                    confidence=event.confidence,
                    reel_targets=["keeper"],
                    player_track_id=event.player_track_id,
                    is_goalkeeper_event=True,
                    frame_start=max(0, int((kick_ts - 0.5) * fps)),
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

    def _classify_touch(
        self,
        touch_frame: int,
        timestamp: float,
        reason: str,
        player_track_id: int,
        player_jersey_hsv: Optional[tuple[float, float, float]],
        ball_pos: tuple[float, float],
        fps: float,
        team_gk_hsv: Optional[tuple[float, float, float]],
        opp_gk_hsv: Optional[tuple[float, float, float]],
        team_outfield_hsv: Optional[tuple[float, float, float]],
        opp_outfield_hsv: Optional[tuple[float, float, float]],
    ) -> Optional[Event]:
        """
        Classify a touch based on the toucher's jersey color.

        Team GK → keeper event
        Opponent GK → skip
        Outfield → skip (highlights shots handled by classify_highlights_events)
        No color data → skip
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

        # Team GK: highest similarity must be to team GK, and above threshold
        is_team_gk = (
            sim_team_gk >= self._gk_color_min_similarity
            and sim_team_gk > sim_opp_gk
            and sim_team_gk > sim_team_of
            and sim_team_gk > sim_opp_of
        )

        # Opponent GK: skip — we don't produce reels for them
        is_opp_gk = (
            sim_opp_gk >= self._gk_color_min_similarity
            and sim_opp_gk > sim_team_gk
            and sim_opp_gk > sim_team_of
            and sim_opp_gk > sim_opp_of
        )

        if is_opp_gk:
            log.debug(
                "ball_touch.opponent_gk_touch",
                frame=touch_frame, reason=reason,
                sim_opp_gk=round(sim_opp_gk, 3),
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
            )
            return None

        # Team GK touch → keeper event
        event_type = self._reason_to_event_type(reason)
        confidence = min(0.90, 0.60 + sim_team_gk * 0.3)

        event = Event(
            job_id=self.job_id,
            source_file=self.source_file,
            event_type=event_type,
            timestamp_start=max(0, timestamp - 0.5),
            timestamp_end=timestamp + 0.5,
            confidence=confidence,
            reel_targets=["keeper"],
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
                "player_track_id": player_track_id,
                "ball_x": round(ball_pos[0], 4),
                "ball_y": round(ball_pos[1], 4),
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
