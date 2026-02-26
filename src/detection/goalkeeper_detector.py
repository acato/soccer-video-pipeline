"""
Goalkeeper identification and GK-specific event detection.

GK identification uses a jersey-first approach:
  1. Jersey color uniqueness — PRIMARY signal: per-half outlier color = GK
  2. Glove color heuristic — supplementary confidence boost
  3. Edge heuristic — supplementary cross-check (logged, not decisive)

Two keepers are tracked: keeper_a (left half) and keeper_b (right half).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import structlog

from src.detection.models import (
    BoundingBox, Detection, Event, EventType, FieldPosition,
    Track, EVENT_REEL_MAP,
)
from src.ingestion.models import MatchConfig

log = structlog.get_logger(__name__)

# Goal area bounds in field coordinates (meters)
# Standard pitch: 105m x 68m
GK_ZONE_DEPTH_METERS = 20.0   # Distance from goal line to consider GK territory
GOAL_LINE_LEFT_X = 0.0
GOAL_LINE_RIGHT_X = 105.0
PITCH_WIDTH = 68.0

class GoalkeeperDetector:
    """
    Identifies goalkeeper tracks and classifies GK-specific events.

    Tracks two keepers (keeper_a = left half, keeper_b = right half).
    State is per-match (not per-chunk).
    """

    def __init__(self, job_id: str, source_file: str, match_config: Optional[MatchConfig] = None):
        self.job_id = job_id
        self.source_file = source_file
        self._match_config = match_config
        self._gk_track_ids: dict[str, Optional[int]] = {
            "keeper_a": None,
            "keeper_b": None,
        }
        # Maps spatial role → reel target label.
        # None means the opponent's GK — no reel is produced for them.
        # Set to "keeper" for the team's GK once identification has run.
        self._gk_reel_labels: dict[str, Optional[str]] = {
            "keeper_a": None,
            "keeper_b": None,
        }
        self._homography: Optional[np.ndarray] = None
        self._prev_gk_positions: dict[str, Optional[tuple[float, float]]] = {
            "keeper_a": None,
            "keeper_b": None,
        }

    def set_homography(self, H: np.ndarray) -> None:
        """Set the 3x3 field homography matrix (pixel -> meters)."""
        self._homography = H

    def identify_goalkeepers(
        self,
        tracks: list[Track],
        frame_shape: tuple[int, int],
        track_colors: dict[int, tuple[float, float, float]],
        frames_data: Optional[dict[int, np.ndarray]] = None,
    ) -> dict[str, Optional[int]]:
        """
        Identify goalkeepers using edge heuristic + color confirmation.

        1. Edge heuristic finds GK by position (near goal line, isolated, stable)
        2. Jersey color check: team GK (→ "keeper" label) vs opponent (→ None)

        Returns {"keeper_a": track_id or None, "keeper_b": track_id or None}
        """
        from src.detection.jersey_classifier import (
            resolve_jersey_color,
            compute_jersey_similarity,
        )

        # Build track position map (mean_x per track)
        track_positions = self._compute_track_positions(tracks)

        team_gk_hsv = resolve_jersey_color(self._match_config.team.gk_color)
        opp_gk_hsv = resolve_jersey_color(self._match_config.opponent.gk_color)

        result: dict[str, Optional[int]] = {"keeper_a": None, "keeper_b": None}

        # STEP 1: Edge heuristic finds GK candidates per half
        edge_per_half = self._identify_edge_per_half(tracks)
        if not edge_per_half:
            return result

        # STEP 2: Among edge candidates, find best team-GK color match per half.
        # The edge filter removes midfield players; color matching among the
        # remaining edge-region players identifies the team's GK.
        team_candidates: list[tuple[str, int, float]] = []  # (role, tid, sim_team)
        for role, candidates_list in edge_per_half.items():
            best_team_tid = None
            best_team_sim = -1.0
            best_opp_tid = None
            best_opp_sim = -1.0

            for tid, edge_score in candidates_list:
                if tid not in track_colors:
                    continue
                sim_team = compute_jersey_similarity(track_colors[tid], team_gk_hsv)
                sim_opp = compute_jersey_similarity(track_colors[tid], opp_gk_hsv)
                if sim_team > sim_opp and sim_team > best_team_sim:
                    best_team_tid = tid
                    best_team_sim = sim_team
                if sim_opp >= sim_team and sim_opp > best_opp_sim:
                    best_opp_tid = tid
                    best_opp_sim = sim_opp

            # Prefer the team-color match for this half
            if best_team_tid is not None:
                result[role] = best_team_tid
                team_candidates.append((role, best_team_tid, best_team_sim))
                log.info(
                    "gk_detector.team_gk_confirmed",
                    role=role, track_id=best_team_tid,
                    sim_team=round(best_team_sim, 3),
                    edge_candidates=len(candidates_list),
                )
            elif best_opp_tid is not None:
                result[role] = best_opp_tid
                self._gk_reel_labels[role] = None
                log.info(
                    "gk_detector.opponent_gk",
                    role=role, track_id=best_opp_tid,
                    sim_opp=round(best_opp_sim, 3),
                    edge_candidates=len(candidates_list),
                )
            else:
                # No color data — carry forward previous label
                if candidates_list:
                    result[role] = candidates_list[0][0]
                log.debug(
                    "gk_detector.no_color_data",
                    role=role,
                    carried_label=self._gk_reel_labels.get(role),
                )

        # Update internal state for cross-chunk continuity
        for role in ("keeper_a", "keeper_b"):
            tid = result.get(role)
            if tid is not None:
                self._gk_track_ids[role] = tid
                if tid in track_positions:
                    self._prev_gk_positions[role] = (track_positions[tid], 0.5)

        # Assign "keeper" label to at most one role (best team match)
        if team_candidates:
            team_candidates.sort(key=lambda c: c[2], reverse=True)
            best_role, best_tid, _ = team_candidates[0]
            self._gk_reel_labels[best_role] = "keeper"
            for role, _, _ in team_candidates[1:]:
                self._gk_reel_labels[role] = None

        return result

    def identify_goalkeeper(
        self,
        tracks: list[Track],
        frame_shape: tuple[int, int],
    ) -> Optional[int]:
        """
        Deprecated: use identify_goalkeepers() instead.
        Returns the first non-None keeper track_id (backward compat).
        """
        # If already identified and still active, reuse
        for role in ("keeper_a", "keeper_b"):
            tid = self._gk_track_ids.get(role)
            if tid is not None:
                active_ids = {t.track_id for t in tracks}
                if tid in active_ids:
                    return tid

        # Fallback to edge heuristic (old behavior)
        if self._homography is not None:
            gk_id = self._identify_by_position(tracks, frame_shape)
            if gk_id is not None:
                return gk_id

        gk_id = self._identify_by_edge_heuristic(tracks)
        if gk_id is not None:
            return gk_id

        return None

    def reel_label_for(self, role: str) -> Optional[str]:
        """
        Return the reel target label for a spatial keeper role.

        Returns 'keeper' if this role is the team's GK, or None if it is the
        opponent's GK (meaning no events should be emitted for them).
        """
        return self._gk_reel_labels.get(role)

    def classify_gk_events(
        self,
        gk_track: Track,
        all_tracks: list[Track],
        source_fps: float,
        keeper_role: str = "keeper_a",
    ) -> list[Event]:
        """
        Classify GK-specific events from the GK track.
        Returns list of Events with reel_targets=[keeper_role].

        Detects ball touches using proximity + trajectory change:
        the ball must be near the GK AND change direction, slow down,
        or disappear (caught).  Event windows are ±0.5s so that with
        keeper reel padding (±1.5s) the final clip is ±2s.
        """
        events = []

        if not gk_track.detections:
            return events

        n = len(gk_track.detections)
        mean_w = sum(d.bbox.width for d in gk_track.detections) / n
        mean_h = sum(d.bbox.height for d in gk_track.detections) / n

        log.info(
            "gk_detector.bbox_dims",
            keeper_role=keeper_role,
            mean_bbox_width=round(mean_w, 4),
            mean_bbox_height=round(mean_h, 4),
            num_detections=n,
        )

        events.extend(self._detect_ball_contacts(gk_track, all_tracks, source_fps, keeper_role))

        return events

    # -- Private detection methods --

    def _detect_ball_contacts(
        self, gk_track: Track, all_tracks: list[Track], fps: float,
        keeper_role: str = "keeper_a",
    ) -> list[Event]:
        """
        Detect when GK touches the ball: proximity + trajectory change.

        Two-gate filter:
        1. **Proximity**: ball within ARM_REACH (0.06) of GK bbox edge.
        2. **Trajectory change**: ball changes direction (>45°), speed
           drops (>60%), ball disappears (catch), or ball appears (throw).

        A ball merely passing near the GK keeps its trajectory and is
        rejected.  Only actual touches alter the ball's behaviour.
        """
        events: list[Event] = []
        dets = gk_track.detections
        if not dets:
            return events

        # Collect all ball detections indexed by frame number.
        ball_by_frame: dict[int, tuple[float, float, float]] = {}  # frame → (ts, x, y)
        for track in all_tracks:
            for det in track.detections:
                if det.class_name == "ball":
                    ball_by_frame[det.frame_number] = (
                        det.timestamp, det.bbox.center_x, det.bbox.center_y,
                    )

        if not ball_by_frame:
            log.debug(
                "gk_detector.ball_contacts_no_ball_data",
                keeper_role=keeper_role,
            )
            return events

        # Sorted ball frame numbers for trajectory lookups.
        sorted_ball_frames = sorted(ball_by_frame.keys())

        # Proximity reach beyond GK bbox edge.  Generous (~goal area)
        # because the trajectory-change gate handles precision.
        ARM_REACH = 0.12
        # Trajectory analysis window (frames before/after contact).
        TRAJ_WINDOW = 30  # ~1s at 30fps

        # Use mean GK position for the entire chunk.  The GK stays in
        # the goal area, so the mean is a stable reference — avoids the
        # problem of sparse GK detections (3–12 per 30s chunk) making
        # frame-by-frame matching miss most ball frames.
        n = len(dets)
        mean_cx = sum(d.bbox.center_x for d in dets) / n
        mean_cy = sum(d.bbox.center_y for d in dets) / n
        mean_w  = sum((d.bbox.width or 0.05) for d in dets) / n
        mean_h  = sum((d.bbox.height or 0.15) for d in dets) / n

        gk_left   = mean_cx - mean_w / 2
        gk_right  = mean_cx + mean_w / 2
        gk_top    = mean_cy - mean_h / 2
        gk_bottom = mean_cy + mean_h / 2

        log.debug(
            "gk_detector.ball_contact_area",
            keeper_role=keeper_role,
            gk_cx=round(mean_cx, 3), gk_cy=round(mean_cy, 3),
            gk_w=round(mean_w, 3), gk_h=round(mean_h, 3),
            zone_x=(round(gk_left - ARM_REACH, 3), round(gk_right + ARM_REACH, 3)),
            zone_y=(round(gk_top - ARM_REACH, 3), round(gk_bottom + ARM_REACH, 3)),
        )

        # Iterate over ALL ball detections (dense, hundreds per chunk)
        # and check proximity to the GK's area.
        contact_moments: list[tuple[float, float, float, str]] = []  # (ts, bx, by, reason)
        candidates = 0
        min_dist = float("inf")
        seen_ball_frames: set[int] = set()

        for ball_frame in sorted_ball_frames:
            if ball_frame in seen_ball_frames:
                continue
            ball_ts, bx, by = ball_by_frame[ball_frame]

            # Gate 1: proximity to mean GK bbox.
            dx = max(gk_left - bx, 0, bx - gk_right)
            dy = max(gk_top - by, 0, by - gk_bottom)
            dist = (dx**2 + dy**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
            if dist > ARM_REACH:
                continue

            candidates += 1

            # Gate 2: trajectory change.
            reason = self._check_trajectory_change(
                ball_frame, ball_by_frame, sorted_ball_frames, TRAJ_WINDOW,
            )
            if reason:
                contact_moments.append((ball_ts, bx, by, reason))
                # Skip nearby ball frames to avoid duplicate contacts.
                for skip_off in range(-5, 6):
                    seen_ball_frames.add(ball_frame + skip_off)

        if not contact_moments:
            log.debug(
                "gk_detector.ball_contacts_none",
                keeper_role=keeper_role,
                gk_dets=len(dets),
                ball_frames=len(ball_by_frame),
                proximity_candidates=candidates,
                min_ball_dist=round(min_dist, 4) if min_dist < float("inf") else None,
            )
            return events

        # Group contacts within 2s into single events.
        contact_moments.sort()
        groups: list[list[tuple[float, float, float, str]]] = [[contact_moments[0]]]
        for moment in contact_moments[1:]:
            if moment[0] - groups[-1][-1][0] < 2.0:
                groups[-1].append(moment)
            else:
                groups.append([moment])

        for group in groups:
            ts_center = (group[0][0] + group[-1][0]) / 2.0
            events.append(Event(
                job_id=self.job_id,
                source_file=self.source_file,
                event_type=EventType.SHOT_STOP_STANDING,
                timestamp_start=max(0, ts_center - 0.5),
                timestamp_end=ts_center + 0.5,
                confidence=0.85,
                reel_targets=[keeper_role],
                player_track_id=gk_track.track_id,
                is_goalkeeper_event=True,
                frame_start=max(0, int((ts_center - 0.5) * fps)),
                frame_end=int((ts_center + 0.5) * fps),
                bounding_box=gk_track.detections[0].bbox,
                metadata={
                    "detection_method": "ball_contact",
                    "keeper_role": keeper_role,
                    "contact_count": len(group),
                    "trajectory_reason": group[0][3],
                },
            ))

        log.info(
            "gk_detector.ball_contacts",
            keeper_role=keeper_role,
            contacts=len(groups),
            raw_contact_frames=len(contact_moments),
            proximity_candidates=candidates,
        )
        return self._merge_nearby_events(events, min_gap_sec=2.0)

    @staticmethod
    def _check_trajectory_change(
        contact_frame: int,
        ball_by_frame: dict[int, tuple[float, float, float]],
        sorted_frames: list[int],
        window: int,
    ) -> Optional[str]:
        """
        Check if ball trajectory changes around *contact_frame*.

        Returns a reason string if trajectory changed, or None if the ball
        maintained consistent motion (i.e. just passed by without being touched).

        Checks:
        - **direction_change**: pre/post velocity angle > 45°
        - **speed_drop**: post speed < 40% of pre speed (catch/control)
        - **ball_caught**: ball detected before but disappears after
        - **ball_released**: ball absent before but appears after (throw/kick)
        - **speed_spike**: post speed >> pre speed (GK kick)
        """
        import bisect

        # Find ball positions BEFORE contact (window frames back, skip ±3 near contact).
        lo = bisect.bisect_left(sorted_frames, contact_frame - window)
        hi = bisect.bisect_left(sorted_frames, contact_frame - 2)
        pre_frames = sorted_frames[lo:hi]

        # Find ball positions AFTER contact (window frames forward, skip ±3 near contact).
        lo2 = bisect.bisect_right(sorted_frames, contact_frame + 2)
        hi2 = bisect.bisect_right(sorted_frames, contact_frame + window)
        post_frames = sorted_frames[lo2:hi2]

        has_pre = len(pre_frames) >= 2
        has_post = len(post_frames) >= 2

        # Ball disappears after contact → catch.
        if has_pre and not has_post:
            return "ball_caught"

        # Ball appears after contact → GK throw/kick from hands.
        if not has_pre and has_post:
            return "ball_released"

        if not has_pre or not has_post:
            return None  # not enough data

        # Compute pre-contact velocity vector.
        x0_pre, y0_pre = ball_by_frame[pre_frames[0]][1], ball_by_frame[pre_frames[0]][2]
        x1_pre, y1_pre = ball_by_frame[pre_frames[-1]][1], ball_by_frame[pre_frames[-1]][2]
        dt_pre = ball_by_frame[pre_frames[-1]][0] - ball_by_frame[pre_frames[0]][0]
        if dt_pre <= 0:
            return None
        pre_vx = (x1_pre - x0_pre) / dt_pre
        pre_vy = (y1_pre - y0_pre) / dt_pre
        pre_speed = (pre_vx**2 + pre_vy**2) ** 0.5

        # Compute post-contact velocity vector.
        x0_post, y0_post = ball_by_frame[post_frames[0]][1], ball_by_frame[post_frames[0]][2]
        x1_post, y1_post = ball_by_frame[post_frames[-1]][1], ball_by_frame[post_frames[-1]][2]
        dt_post = ball_by_frame[post_frames[-1]][0] - ball_by_frame[post_frames[0]][0]
        if dt_post <= 0:
            return None
        post_vx = (x1_post - x0_post) / dt_post
        post_vy = (y1_post - y0_post) / dt_post
        post_speed = (post_vx**2 + post_vy**2) ** 0.5

        # Speed drop → catch/control.
        if pre_speed > 0.15 and post_speed < pre_speed * 0.4:
            return "speed_drop"

        # Direction change > 45°.
        if pre_speed > 0.15 and post_speed > 0.15:
            dot = pre_vx * post_vx + pre_vy * post_vy
            cos_angle = dot / (pre_speed * post_speed)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            if cos_angle < 0.707:  # cos(45°) ≈ 0.707
                return "direction_change"

        # Speed spike → GK kick/throw (ball was slow/stationary, now fast).
        if post_speed > 0.3 and (pre_speed < 0.1 or post_speed > pre_speed * 3):
            return "speed_spike"

        return None

    def _detect_distribution(
        self, gk_track: Track, fps: float, keeper_role: str = "keeper_a",
    ) -> list[Event]:
        """
        Detect goal kicks and distribution from GK track.
        Uses velocity analysis: sudden bbox position change after stationary period.
        Velocities are normalized by bbox width (body-widths per second).
        """
        events = []
        dets = gk_track.detections
        if len(dets) < 5:
            return events

        velocities = []
        for i in range(1, len(dets)):
            dx = dets[i].bbox.center_x - dets[i-1].bbox.center_x
            dt = dets[i].timestamp - dets[i-1].timestamp
            bbox_w = dets[i].bbox.width or 0.01
            velocities.append(abs(dx / (dt * bbox_w)) if dt > 0 else 0)

        threshold = 1.5  # body-widths per second
        pre_vel_limit = 1.0  # "stationary" in normalized units (jitter ≈ 0.3-0.6)
        max_post_vel = 0.0
        best_candidate = None

        # Adaptive window: sparse GK tracks (avg 8 dets/chunk) need small windows
        window = min(10, max(3, len(velocities) // 3))
        for i in range(window, len(velocities)):
            pre_vel = np.mean(velocities[max(0, i-window):i])
            post_vel = np.mean(velocities[i:min(len(velocities), i+5)])

            if pre_vel < pre_vel_limit and post_vel > max_post_vel:
                max_post_vel = post_vel
                best_candidate = (pre_vel, post_vel)

            if pre_vel < pre_vel_limit and post_vel > threshold:
                det = dets[i]
                event_type = self._classify_distribution_type(det)
                vel_ratio = post_vel / max(pre_vel, 0.001)
                confidence = min(0.90, 0.55 + vel_ratio * 0.01)
                events.append(Event(
                    job_id=self.job_id,
                    source_file=self.source_file,
                    event_type=event_type,
                    timestamp_start=max(0, det.timestamp - 0.5),
                    timestamp_end=det.timestamp + 2.0,
                    confidence=confidence,
                    reel_targets=[keeper_role],
                    player_track_id=gk_track.track_id,
                    is_goalkeeper_event=True,
                    frame_start=max(0, det.frame_number - int(0.5 * fps)),
                    frame_end=det.frame_number + int(2 * fps),
                    bounding_box=det.bbox,
                    metadata={
                        "detection_method": "velocity_transition",
                        "pre_vel": round(float(pre_vel), 4),
                        "post_vel": round(float(post_vel), 4),
                        "keeper_role": keeper_role,
                    },
                ))

        if not events:
            log.debug(
                "gk_detector.distribution_miss",
                num_dets=len(dets),
                max_post_vel=round(max_post_vel, 4),
                threshold=threshold,
                best_candidate=best_candidate,
                keeper_role=keeper_role,
            )

        return events

    def _detect_saves(
        self, gk_track: Track, all_tracks: list[Track], fps: float,
        keeper_role: str = "keeper_a",
    ) -> list[Event]:
        """
        Detect save events: GK makes sudden vertical or lateral movement
        coinciding with ball proximity.
        Velocities are normalized by bbox height (body-heights per second).
        """
        events = []
        dets = gk_track.detections
        if len(dets) < 4:
            return events

        ball_positions: dict[int, tuple[float, float]] = {}
        for track in all_tracks:
            for det in track.detections:
                if det.class_name == "ball":
                    ball_positions[det.frame_number] = (det.bbox.center_x, det.bbox.center_y)
        has_ball_data = len(ball_positions) > 0

        vel_threshold = 1.0 if has_ball_data else 1.5  # body-heights per second
        dive_threshold = 2.5  # body-heights per second
        max_vertical_velocity = 0.0

        for i in range(1, len(dets) - 1):
            prev, curr, nxt = dets[i-1], dets[i], dets[i+1]

            dy_prev = abs(curr.bbox.center_y - prev.bbox.center_y)
            dy_next = abs(nxt.bbox.center_y - curr.bbox.center_y)
            dt_prev = curr.timestamp - prev.timestamp
            dt_next = nxt.timestamp - curr.timestamp

            if dt_prev <= 0 or dt_next <= 0:
                continue

            bbox_h = curr.bbox.height or 0.01
            v_prev = dy_prev / (dt_prev * bbox_h)
            v_next = dy_next / (dt_next * bbox_h)
            vertical_velocity = (v_prev + v_next) / 2.0

            if vertical_velocity > max_vertical_velocity:
                max_vertical_velocity = vertical_velocity

            if vertical_velocity <= vel_threshold:
                continue

            ball_distance = None
            gk_x, gk_y = curr.bbox.center_x, curr.bbox.center_y
            frame = curr.frame_number
            # ±5 frames covers adjacent detection-step frames even at step=3
            for f_offset in range(-5, 6):
                if (frame + f_offset) in ball_positions:
                    bx, by = ball_positions[frame + f_offset]
                    dist = ((gk_x - bx) ** 2 + (gk_y - by) ** 2) ** 0.5
                    if ball_distance is None or dist < ball_distance:
                        ball_distance = dist

            # For sub-dive velocities, require ball proximity when available.
            # For dive-threshold velocities (unmistakable athletic motion),
            # only reject if the ball is confirmed far away — allow missing
            # ball data since the dive itself is strong evidence.
            if vertical_velocity <= dive_threshold:
                if has_ball_data and (ball_distance is None or ball_distance > 0.15):
                    continue
            elif has_ball_data and ball_distance is not None and ball_distance > 0.30:
                continue

            event_type = (
                EventType.SHOT_STOP_DIVING
                if vertical_velocity > dive_threshold
                else EventType.SHOT_STOP_STANDING
            )

            base_conf = min(0.80, 0.50 + vertical_velocity)
            ball_bonus = 0.10 if (ball_distance is not None and ball_distance < 0.10) else 0.0
            confidence_cap = 0.90 if has_ball_data else 0.75
            confidence = min(confidence_cap, base_conf + ball_bonus)

            metadata = {
                "vertical_velocity": round(vertical_velocity, 4),
                "keeper_role": keeper_role,
            }
            if ball_distance is not None:
                metadata["ball_distance"] = round(ball_distance, 4)

            events.append(Event(
                job_id=self.job_id,
                source_file=self.source_file,
                event_type=event_type,
                timestamp_start=max(0, prev.timestamp - 0.5),
                timestamp_end=nxt.timestamp + 2.0,
                confidence=confidence,
                reel_targets=[keeper_role],
                player_track_id=gk_track.track_id,
                is_goalkeeper_event=True,
                frame_start=prev.frame_number,
                frame_end=nxt.frame_number + int(2 * fps),
                bounding_box=curr.bbox,
                metadata=metadata,
            ))

        if not events:
            log.debug(
                "gk_detector.saves_miss",
                num_dets=len(dets),
                max_vertical_velocity=round(max_vertical_velocity, 4),
                vel_threshold=round(vel_threshold, 4),
                has_ball_data=has_ball_data,
                keeper_role=keeper_role,
            )

        return self._merge_nearby_events(events, min_gap_sec=2.0)

    def _detect_one_on_ones(
        self, gk_track: Track, all_tracks: list[Track], fps: float,
        keeper_role: str = "keeper_a",
    ) -> list[Event]:
        """
        Detect one-on-one: GK moves significantly away from goal line.
        Requires 3 consecutive frames beyond threshold to filter noise.
        Deviation is normalized by bbox height (body-heights from baseline).
        """
        events = []
        dets = gk_track.detections
        if len(dets) < 4:
            return events

        deviation_threshold = 0.3  # body-heights from baseline
        consecutive_required = 3
        max_deviation = 0.0

        center_y_values = [d.bbox.center_y for d in dets]
        consecutive_count = 0

        for i, det in enumerate(dets):
            baseline_y = np.mean(center_y_values[:max(1, i)])
            deviation = abs(det.bbox.center_y - baseline_y) / (det.bbox.height or 0.01)

            if deviation > max_deviation:
                max_deviation = deviation

            if deviation > deviation_threshold:
                consecutive_count += 1
            else:
                consecutive_count = 0

            if consecutive_count >= consecutive_required:
                excess = deviation - deviation_threshold
                confidence = min(0.88, 0.65 + excess * 2.0)

                # ONE_ON_ONE goes to both the keeper reel and highlights
                reel_targets = [keeper_role, "highlights"]

                events.append(Event(
                    job_id=self.job_id,
                    source_file=self.source_file,
                    event_type=EventType.ONE_ON_ONE,
                    timestamp_start=max(0, det.timestamp - 0.5),
                    timestamp_end=det.timestamp + 3.0,
                    confidence=confidence,
                    reel_targets=reel_targets,
                    player_track_id=gk_track.track_id,
                    is_goalkeeper_event=True,
                    frame_start=max(0, det.frame_number - int(0.5 * fps)),
                    frame_end=det.frame_number + int(3 * fps),
                    bounding_box=det.bbox,
                    metadata={
                        "deviation": round(deviation, 4),
                        "consecutive_frames": consecutive_count,
                        "keeper_role": keeper_role,
                    },
                ))
                consecutive_count = 0

        if not events:
            log.debug(
                "gk_detector.one_on_one_miss",
                num_dets=len(dets),
                max_deviation=round(max_deviation, 4),
                deviation_threshold=round(deviation_threshold, 4),
                keeper_role=keeper_role,
            )

        return self._merge_nearby_events(events, min_gap_sec=5.0)

    def _classify_distribution_type(self, det: Detection) -> EventType:
        """Heuristic: if GK bbox is very low in frame, likely a goal kick."""
        if det.bbox.center_y > 0.75:
            return EventType.GOAL_KICK
        elif det.bbox.center_y > 0.55:
            return EventType.DISTRIBUTION_SHORT
        return EventType.DISTRIBUTION_LONG

    def _compute_track_positions(
        self, tracks: list[Track], min_detections: int = 3,
    ) -> dict[int, float]:
        """Compute mean_x position for each track."""
        positions: dict[int, float] = {}
        for track in tracks:
            dets = [
                d for d in track.detections
                if d.class_name in ("player", "goalkeeper")
            ]
            if len(dets) < min_detections:
                continue
            xs = [d.bbox.center_x for d in dets]
            positions[track.track_id] = float(np.mean(xs))
        return positions

    def _identify_by_edge_heuristic(
        self, tracks: list[Track], min_detections: int = 15,
    ) -> Optional[int]:
        """Return the single best GK candidate (backward compat)."""
        per_half = self._identify_edge_per_half(tracks, min_detections)
        if not per_half:
            return None
        # Return the one with the higher score across all halves
        all_candidates = [c for candidates in per_half.values() for c in candidates]
        if not all_candidates:
            return None
        best = max(all_candidates, key=lambda x: x[1])
        return best[0]

    def _identify_edge_per_half(
        self, tracks: list[Track], min_detections: int = 15,
    ) -> dict[str, list[tuple[int, float]]]:
        """
        Find edge-region GK candidates per half, sorted by edge score.

        Returns {"keeper_a": [(track_id, score), ...], "keeper_b": [(track_id, score), ...]}
        Each list is sorted by descending score.  Caller picks the best
        team-color match from these candidates.

        Multi-signal scoring:
          - Edge proximity (30%): mean_x near frame edge (< 0.12 from edge)
          - Stability (15%): inverse of positional variance
          - Isolation (30%): average distance from candidate to all other
            same-half tracks (GK is typically the most isolated)
          - Behind-defensive-line (25%): candidate is outermost among
            same-half players (deeper than defenders)

        Cross-chunk continuity: candidates near previous GK position get a bonus.
        """
        track_means: dict[int, tuple[float, float]] = {}
        for track in tracks:
            player_dets = [
                d for d in track.detections
                if d.class_name in ("player", "goalkeeper")
            ]
            if len(player_dets) < min_detections:
                continue
            xs = np.array([d.bbox.center_x for d in player_dets])
            ys = np.array([d.bbox.center_y for d in player_dets])
            track_means[track.track_id] = (float(np.mean(xs)), float(np.mean(ys)))

        if not track_means:
            return {}

        candidates: list[tuple[int, float, dict, bool]] = []  # (id, score, scores, is_left)

        for track in tracks:
            if track.track_id not in track_means:
                continue

            mean_x, mean_y = track_means[track.track_id]
            player_dets = [
                d for d in track.detections
                if d.class_name in ("player", "goalkeeper")
            ]
            xs = np.array([d.bbox.center_x for d in player_dets])
            ys = np.array([d.bbox.center_y for d in player_dets])
            var_x = float(np.var(xs))
            var_y = float(np.var(ys))

            edge_dist = min(mean_x, 1.0 - mean_x)
            edge_score = max(0.0, 1.0 - edge_dist / 0.12)

            if edge_score <= 0.0:
                continue

            positional_var = var_x + var_y
            stability = 1.0 / (1.0 + positional_var * 100.0)

            candidate_half_left = mean_x < 0.5
            same_half_means = [
                m for tid, m in track_means.items()
                if tid != track.track_id and (m[0] < 0.5) == candidate_half_left
            ]
            if same_half_means:
                distances = [
                    ((mean_x - mx) ** 2 + (mean_y - my) ** 2) ** 0.5
                    for mx, my in same_half_means
                ]
                avg_dist = float(np.mean(distances))
                isolation = min(1.0, avg_dist / 0.3)
            else:
                isolation = 0.5

            if same_half_means:
                if candidate_half_left:
                    deeper_count = sum(1 for mx, _ in same_half_means if mx < mean_x)
                    behind_line = 1.0 - (deeper_count / len(same_half_means))
                else:
                    deeper_count = sum(1 for mx, _ in same_half_means if mx > mean_x)
                    behind_line = 1.0 - (deeper_count / len(same_half_means))
            else:
                behind_line = 0.5

            score = (
                edge_score * 0.30
                + stability * 0.15
                + isolation * 0.30
                + behind_line * 0.25
            )

            # Cross-chunk continuity bonus
            for role, prev_pos in self._prev_gk_positions.items():
                if prev_pos is not None:
                    prev_x, prev_y = prev_pos
                    dist_to_prev = ((mean_x - prev_x) ** 2 + (mean_y - prev_y) ** 2) ** 0.5
                    if dist_to_prev < 0.10:
                        score += 0.15
                        break

            candidates.append((track.track_id, score, {
                "edge": round(edge_score, 3),
                "stability": round(stability, 3),
                "isolation": round(isolation, 3),
                "behind_line": round(behind_line, 3),
            }, candidate_half_left))

        if not candidates:
            return {}

        # All qualifying candidates per half, sorted by score
        result: dict[str, list[tuple[int, float]]] = {}
        min_score = 0.45
        left = [(tid, s, sc) for tid, s, sc, is_left in candidates if is_left]
        right = [(tid, s, sc) for tid, s, sc, is_left in candidates if not is_left]

        for role, group in (("keeper_a", left), ("keeper_b", right)):
            if not group:
                continue
            group.sort(key=lambda c: c[1], reverse=True)
            qualified = [(tid, s) for tid, s, sc in group if s >= min_score]
            if not qualified:
                continue
            best_id, best_score = qualified[0]
            best_scores = next(sc for tid, s, sc in group if tid == best_id)
            log.info(
                "gk_detector.edge_heuristic_result",
                role=role,
                track_id=best_id,
                score=round(best_score, 3),
                scores=best_scores,
                candidates=len(qualified),
            )
            result[role] = qualified

        return result

    def _identify_by_position(
        self, tracks: list[Track], frame_shape: tuple[int, int]
    ) -> Optional[int]:
        """Use homography to find player consistently in GK zone."""
        if self._homography is None:
            return None

        gk_zone_candidates: dict[int, int] = Counter()

        for track in tracks:
            for det in track.detections[-30:]:
                pixel_center = np.array([
                    [det.bbox.center_x * frame_shape[1],
                     det.bbox.center_y * frame_shape[0]]
                ], dtype=np.float32).reshape(-1, 1, 2)

                field_pt = cv2_perspective_transform(pixel_center, self._homography)
                if field_pt is not None:
                    fx, fy = field_pt[0][0]
                    if self._in_gk_zone(fx):
                        gk_zone_candidates[track.track_id] += 1

        if not gk_zone_candidates:
            return None

        best_id = max(gk_zone_candidates, key=gk_zone_candidates.get)
        if gk_zone_candidates[best_id] >= 5:
            return best_id
        return None

    def _in_gk_zone(self, field_x: float) -> bool:
        """Check if field X coordinate is within GK zone."""
        return (field_x < GK_ZONE_DEPTH_METERS or
                field_x > (GOAL_LINE_RIGHT_X - GK_ZONE_DEPTH_METERS))

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


def cv2_perspective_transform(
    points: np.ndarray, H: np.ndarray
) -> Optional[np.ndarray]:
    """Apply perspective transform H to points. Returns None on error."""
    try:
        import cv2
        return cv2.perspectiveTransform(points, H)
    except Exception:
        return None
