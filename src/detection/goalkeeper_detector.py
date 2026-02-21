"""
Goalkeeper identification and GK-specific event detection.

GK identification uses a multi-signal approach:
  1. Field position — GK stays near their goal line (< 20m from goal)
  2. Jersey color — GK wears a distinct color from outfield players
  3. Track continuity — once identified, ByteTrack maintains identity

Field homography is required for position-based identification.
Falls back to jersey-color-only if homography is unavailable.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import structlog

from src.detection.models import (
    BoundingBox, Detection, Event, EventType, FieldPosition,
    Track, EVENT_REEL_MAP
)

log = structlog.get_logger(__name__)

# Goal area bounds in field coordinates (meters)
# Standard pitch: 105m x 68m
GK_ZONE_DEPTH_METERS = 20.0   # Distance from goal line to consider GK territory
GOAL_LINE_LEFT_X = 0.0
GOAL_LINE_RIGHT_X = 105.0
PITCH_WIDTH = 68.0


class GoalkeeperDetector:
    """
    Identifies the goalkeeper track and classifies GK-specific events.

    State is per-match (not per-chunk): GK identity persists across the full video.
    """

    def __init__(self, job_id: str, source_file: str, team_goal_x: float = 0.0):
        """
        Args:
            team_goal_x: X coordinate of the GK's goal (0.0 = left goal, 105.0 = right goal)
        """
        self.job_id = job_id
        self.source_file = source_file
        self.team_goal_x = team_goal_x
        self._gk_track_id: Optional[int] = None
        self._gk_jersey_hsv: Optional[tuple[float, float, float]] = None
        self._homography: Optional[np.ndarray] = None  # 3x3 perspective matrix
        self._prev_gk_position: Optional[tuple[float, float]] = None  # Cross-chunk continuity

    def set_homography(self, H: np.ndarray) -> None:
        """Set the 3x3 field homography matrix (pixel → meters)."""
        self._homography = H

    def identify_goalkeeper(
        self,
        tracks: list[Track],
        frame_shape: tuple[int, int],
    ) -> Optional[int]:
        """
        Identify the goalkeeper track ID from a set of tracks.
        Updates internal GK identity state.

        Returns track_id of identified GK, or None if not identified.
        """
        # If already identified and still active in these tracks, reuse
        if self._gk_track_id is not None:
            active_ids = {t.track_id for t in tracks}
            if self._gk_track_id in active_ids:
                return self._gk_track_id
            # Track IDs reset between chunks — save position before clearing
            old_gk_tracks = [t for t in tracks if t.track_id == self._gk_track_id]
            if old_gk_tracks and old_gk_tracks[0].detections:
                dets = old_gk_tracks[0].detections
                xs = [d.bbox.center_x for d in dets]
                ys = [d.bbox.center_y for d in dets]
                self._prev_gk_position = (float(np.mean(xs)), float(np.mean(ys)))
            self._gk_track_id = None

        if self._homography is not None:
            gk_id = self._identify_by_position(tracks, frame_shape)
            if gk_id is not None:
                log.info("gk_detector.identified_by_position", track_id=gk_id)
                self._gk_track_id = gk_id
                return gk_id

        # Fallback: position heuristic using normalized bbox coords (no homography)
        gk_id = self._identify_by_edge_heuristic(tracks)
        if gk_id is not None:
            log.info("gk_detector.identified_by_edge_heuristic", track_id=gk_id)
            self._gk_track_id = gk_id
            return gk_id

        log.debug("gk_detector.identification_deferred", reason="insufficient_tracks")
        return None

    def classify_gk_events(
        self,
        gk_track: Track,
        all_tracks: list[Track],
        source_fps: float,
    ) -> list[Event]:
        """
        Classify GK-specific events from the GK track.
        Returns list of Events with reel_targets=["goalkeeper"].
        """
        events = []

        if not gk_track.detections:
            return events

        events.extend(self._detect_distribution(gk_track, source_fps))
        events.extend(self._detect_saves(gk_track, all_tracks, source_fps))
        events.extend(self._detect_one_on_ones(gk_track, all_tracks, source_fps))

        return events

    # ── Private detection methods ──────────────────────────────────────────

    def _detect_distribution(self, gk_track: Track, fps: float) -> list[Event]:
        """
        Detect goal kicks and distribution from GK track.
        Uses velocity analysis: sudden bbox position change after stationary period.
        """
        events = []
        dets = gk_track.detections
        if len(dets) < 5:
            return events

        # Compute center-x velocity over sliding window
        velocities = []
        for i in range(1, len(dets)):
            dx = dets[i].bbox.center_x - dets[i-1].bbox.center_x
            dt = dets[i].timestamp - dets[i-1].timestamp
            velocities.append(abs(dx / dt) if dt > 0 else 0)

        # Find windows where GK was stationary then moved (distribution action)
        window = 10
        for i in range(window, len(velocities)):
            pre_vel = np.mean(velocities[max(0, i-window):i])
            post_vel = np.mean(velocities[i:min(len(velocities), i+5)])

            if pre_vel < 0.01 and post_vel > 0.12:  # Stationary → moving
                det = dets[i]
                event_type = self._classify_distribution_type(det)
                # Scale confidence by velocity transition strength
                vel_ratio = post_vel / max(pre_vel, 0.001)
                confidence = min(0.90, 0.55 + vel_ratio * 0.01)
                events.append(Event(
                    job_id=self.job_id,
                    source_file=self.source_file,
                    event_type=event_type,
                    timestamp_start=max(0, det.timestamp - 1.0),
                    timestamp_end=det.timestamp + 3.0,
                    confidence=confidence,
                    reel_targets=EVENT_REEL_MAP[event_type],
                    player_track_id=gk_track.track_id,
                    is_goalkeeper_event=True,
                    frame_start=max(0, det.frame_number - int(fps)),
                    frame_end=det.frame_number + int(3 * fps),
                    bounding_box=det.bbox,
                    metadata={
                        "detection_method": "velocity_transition",
                        "pre_vel": round(float(pre_vel), 4),
                        "post_vel": round(float(post_vel), 4),
                    },
                ))

        return events

    def _detect_saves(
        self, gk_track: Track, all_tracks: list[Track], fps: float
    ) -> list[Event]:
        """
        Detect save events: GK makes sudden vertical or lateral movement
        coinciding with ball proximity.
        """
        events = []
        dets = gk_track.detections
        if len(dets) < 4:
            return events

        # Build frame-indexed ball position lookup from all tracks
        ball_positions: dict[int, tuple[float, float]] = {}
        for track in all_tracks:
            for det in track.detections:
                if det.class_name == "ball":
                    ball_positions[det.frame_number] = (det.bbox.center_x, det.bbox.center_y)
        has_ball_data = len(ball_positions) > 0

        # Velocity threshold depends on whether we have ball data for confirmation
        vel_threshold = 0.25 if has_ball_data else 0.35
        dive_threshold = 0.40

        for i in range(1, len(dets) - 1):
            # 3-detection smoothed velocity: use detections i-1, i, i+1
            prev, curr, nxt = dets[i-1], dets[i], dets[i+1]

            dy_prev = abs(curr.bbox.center_y - prev.bbox.center_y)
            dy_next = abs(nxt.bbox.center_y - curr.bbox.center_y)
            dt_prev = curr.timestamp - prev.timestamp
            dt_next = nxt.timestamp - curr.timestamp

            if dt_prev <= 0 or dt_next <= 0:
                continue

            # Smoothed vertical velocity across 3 detections
            v_prev = dy_prev / dt_prev
            v_next = dy_next / dt_next
            vertical_velocity = (v_prev + v_next) / 2.0

            if vertical_velocity <= vel_threshold:
                continue

            # Ball proximity check: require ball within 0.15 normalized distance (±2 frames)
            ball_distance = None
            gk_x, gk_y = curr.bbox.center_x, curr.bbox.center_y
            frame = curr.frame_number
            for f_offset in range(-2, 3):
                if (frame + f_offset) in ball_positions:
                    bx, by = ball_positions[frame + f_offset]
                    dist = ((gk_x - bx) ** 2 + (gk_y - by) ** 2) ** 0.5
                    if ball_distance is None or dist < ball_distance:
                        ball_distance = dist

            # If we have ball data, require proximity
            if has_ball_data and (ball_distance is None or ball_distance > 0.15):
                continue

            event_type = (
                EventType.SHOT_STOP_DIVING
                if vertical_velocity > dive_threshold
                else EventType.SHOT_STOP_STANDING
            )

            # Confidence: base from velocity + ball proximity bonus
            base_conf = min(0.80, 0.50 + vertical_velocity)
            ball_bonus = 0.10 if (ball_distance is not None and ball_distance < 0.10) else 0.0
            confidence_cap = 0.90 if has_ball_data else 0.75
            confidence = min(confidence_cap, base_conf + ball_bonus)

            metadata = {"vertical_velocity": round(vertical_velocity, 4)}
            if ball_distance is not None:
                metadata["ball_distance"] = round(ball_distance, 4)

            events.append(Event(
                job_id=self.job_id,
                source_file=self.source_file,
                event_type=event_type,
                timestamp_start=max(0, prev.timestamp - 0.5),
                timestamp_end=nxt.timestamp + 2.0,
                confidence=confidence,
                reel_targets=EVENT_REEL_MAP[event_type],
                player_track_id=gk_track.track_id,
                is_goalkeeper_event=True,
                frame_start=prev.frame_number,
                frame_end=nxt.frame_number + int(2 * fps),
                bounding_box=curr.bbox,
                metadata=metadata,
            ))

        return self._merge_nearby_events(events, min_gap_sec=2.0)

    def _detect_one_on_ones(
        self, gk_track: Track, all_tracks: list[Track], fps: float
    ) -> list[Event]:
        """
        Detect one-on-one: GK moves significantly away from goal line.
        Requires 3 consecutive frames beyond threshold to filter noise.
        """
        events = []
        dets = gk_track.detections
        if len(dets) < 4:
            return events

        deviation_threshold = 0.20
        consecutive_required = 3

        center_y_values = [d.bbox.center_y for d in dets]
        consecutive_count = 0

        for i, det in enumerate(dets):
            baseline_y = np.mean(center_y_values[:max(1, i)])
            deviation = abs(det.bbox.center_y - baseline_y)

            if deviation > deviation_threshold:
                consecutive_count += 1
            else:
                consecutive_count = 0

            if consecutive_count >= consecutive_required:
                # Scale confidence by magnitude of deviation beyond threshold
                excess = deviation - deviation_threshold
                confidence = min(0.88, 0.65 + excess * 2.0)

                events.append(Event(
                    job_id=self.job_id,
                    source_file=self.source_file,
                    event_type=EventType.ONE_ON_ONE,
                    timestamp_start=max(0, det.timestamp - 1.0),
                    timestamp_end=det.timestamp + 4.0,
                    confidence=confidence,
                    reel_targets=EVENT_REEL_MAP[EventType.ONE_ON_ONE],
                    player_track_id=gk_track.track_id,
                    is_goalkeeper_event=True,
                    frame_start=max(0, det.frame_number - int(fps)),
                    frame_end=det.frame_number + int(4 * fps),
                    bounding_box=det.bbox,
                    metadata={
                        "deviation": round(deviation, 4),
                        "consecutive_frames": consecutive_count,
                    },
                ))
                # Reset to avoid duplicate detections for the same rush
                consecutive_count = 0

        return self._merge_nearby_events(events, min_gap_sec=5.0)

    def _classify_distribution_type(self, det: Detection) -> EventType:
        """Heuristic: if GK bbox is very low in frame, likely a goal kick."""
        if det.bbox.center_y > 0.75:
            return EventType.GOAL_KICK
        elif det.bbox.center_y > 0.55:
            return EventType.DISTRIBUTION_SHORT
        return EventType.DISTRIBUTION_LONG

    def _identify_by_edge_heuristic(
        self, tracks: list[Track], min_detections: int = 15,
    ) -> Optional[int]:
        """
        Identify GK without homography using normalized bbox coordinates.

        Multi-signal scoring:
          - Edge proximity (30%): mean_x near frame edge (< 0.12 from edge)
          - Stability (15%): inverse of positional variance
          - Isolation (30%): average distance from candidate to all other
            same-half tracks (GK is typically the most isolated)
          - Behind-defensive-line (25%): candidate is outermost among
            same-half players (deeper than defenders)

        Cross-chunk continuity: candidates near previous GK position get a bonus.
        """
        # Pre-compute per-track mean positions for isolation/behind-line scoring
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
            return None

        candidates: list[tuple[int, float, dict]] = []  # (track_id, score, scores_dict)

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

            # ── Edge proximity (30%) — tighter cutoff: < 0.12 from edge ──
            edge_dist = min(mean_x, 1.0 - mean_x)
            # Score is 1.0 at the edge, 0.0 at 0.12 or beyond
            edge_score = max(0.0, 1.0 - edge_dist / 0.12)

            if edge_score <= 0.0:
                continue  # Must be near an edge to be a GK candidate

            # ── Stability (15%) ──────────────────────────────────────────
            positional_var = var_x + var_y
            stability = 1.0 / (1.0 + positional_var * 100.0)

            # ── Isolation score (30%) ────────────────────────────────────
            # GK is typically the most spatially isolated player on their half
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
                # Normalize: 0.3+ distance is very isolated → score 1.0
                isolation = min(1.0, avg_dist / 0.3)
            else:
                isolation = 0.5  # Only player on this half — neutral

            # ── Behind-defensive-line score (25%) ────────────────────────
            # GK should be the outermost (closest to edge) among same-half players
            if same_half_means:
                if candidate_half_left:
                    # Left half: lower x = closer to left edge = deeper
                    deeper_count = sum(1 for mx, _ in same_half_means if mx < mean_x)
                    behind_line = 1.0 - (deeper_count / len(same_half_means))
                else:
                    # Right half: higher x = closer to right edge = deeper
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
            if self._prev_gk_position is not None:
                prev_x, prev_y = self._prev_gk_position
                dist_to_prev = ((mean_x - prev_x) ** 2 + (mean_y - prev_y) ** 2) ** 0.5
                if dist_to_prev < 0.10:
                    score += 0.15

            candidates.append((track.track_id, score, {
                "edge": round(edge_score, 3),
                "stability": round(stability, 3),
                "isolation": round(isolation, 3),
                "behind_line": round(behind_line, 3),
            }))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[1], reverse=True)
        best_id, best_score, best_scores = candidates[0]

        if best_score < 0.45:
            log.debug(
                "gk_detector.edge_heuristic_rejected",
                best_score=round(best_score, 3),
                scores=best_scores,
            )
            return None

        log.info(
            "gk_detector.edge_heuristic_result",
            track_id=best_id,
            score=round(best_score, 3),
            scores=best_scores,
            candidates=len(candidates),
        )
        return best_id

    def _identify_by_position(
        self, tracks: list[Track], frame_shape: tuple[int, int]
    ) -> Optional[int]:
        """Use homography to find player consistently in GK zone."""
        if self._homography is None:
            return None

        gk_zone_candidates: dict[int, int] = Counter()

        for track in tracks:
            for det in track.detections[-30:]:  # Last 30 detections
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

        # Player most consistently in GK zone is the GK
        best_id = max(gk_zone_candidates, key=gk_zone_candidates.get)
        if gk_zone_candidates[best_id] >= 5:  # Min observations
            return best_id
        return None

    def _in_gk_zone(self, field_x: float) -> bool:
        """Check if field X coordinate is within GK zone."""
        if self.team_goal_x <= 10:  # Left goal
            return field_x < GK_ZONE_DEPTH_METERS
        else:  # Right goal
            return field_x > (GOAL_LINE_RIGHT_X - GK_ZONE_DEPTH_METERS)

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
                # Extend previous event, keep higher confidence
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
