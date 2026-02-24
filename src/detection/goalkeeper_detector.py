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

# Reference GK bbox height (normalized) for threshold calibration.
# Thresholds were tuned for a close-camera view where the GK bbox is ~25% of
# frame height.  For wide-angle footage the GK appears much smaller, so we
# scale thresholds proportionally.
REFERENCE_BBOX_HEIGHT = 0.25


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
        Identify both goalkeepers using known GK jersey colors from match_config.

        PRIMARY: supervised color matching against team/opponent GK colors
        SUPPLEMENTARY: glove color confidence boost, edge heuristic cross-check

        Returns {"keeper_a": track_id or None, "keeper_b": track_id or None}
        """
        from src.detection.jersey_classifier import (
            identify_gk_by_known_colors,
            resolve_jersey_color,
            compute_jersey_similarity,
            detect_glove_color,
        )

        # Build track position map (mean_x per track)
        track_positions = self._compute_track_positions(tracks)

        team_gk_hsv = resolve_jersey_color(self._match_config.team.gk_color)
        opp_gk_hsv = resolve_jersey_color(self._match_config.opponent.gk_color)
        result = identify_gk_by_known_colors(
            track_colors,
            track_positions,
            home_gk_hsv=team_gk_hsv,
            away_gk_hsv=opp_gk_hsv,
        )
        # Label the team's GK role as "keeper"; leave the opponent's as None.
        for role in ("keeper_a", "keeper_b"):
            tid = result.get(role)
            if tid is None or tid not in track_colors:
                self._gk_reel_labels[role] = None
                continue
            sim_team = compute_jersey_similarity(track_colors[tid], team_gk_hsv)
            sim_opp = compute_jersey_similarity(track_colors[tid], opp_gk_hsv)
            self._gk_reel_labels[role] = "keeper" if sim_team >= sim_opp else None

        # SUPPLEMENTARY: glove color confidence boost (log only)
        if frames_data:
            for role in ("keeper_a", "keeper_b"):
                tid = result.get(role)
                if tid is None:
                    continue
                track = next((t for t in tracks if t.track_id == tid), None)
                if track and track.detections:
                    det = track.detections[0]
                    if det.frame_number in frames_data:
                        glove_score = detect_glove_color(
                            frames_data[det.frame_number], det.bbox, frame_shape
                        )
                        if glove_score is not None:
                            log.debug(
                                "gk_detector.glove_score",
                                role=role,
                                track_id=tid,
                                glove_score=round(glove_score, 3),
                            )

        # SUPPLEMENTARY: edge heuristic cross-check (log agreement/disagreement)
        edge_gk = self._identify_by_edge_heuristic(tracks)
        if edge_gk is not None:
            for role in ("keeper_a", "keeper_b"):
                if result.get(role) == edge_gk:
                    log.debug("gk_detector.edge_confirms_jersey", role=role, track_id=edge_gk)
                    break
            else:
                if any(result.get(r) is not None for r in ("keeper_a", "keeper_b")):
                    log.debug(
                        "gk_detector.edge_disagrees",
                        edge_id=edge_gk,
                        jersey_ids=result,
                    )

        # Update internal state
        for role in ("keeper_a", "keeper_b"):
            tid = result.get(role)
            if tid is not None:
                self._gk_track_ids[role] = tid
            # Save position for cross-chunk continuity
            if tid is not None and tid in track_positions:
                self._prev_gk_positions[role] = (track_positions[tid], 0.5)

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

        Velocity/deviation thresholds are scaled by the GK's apparent size
        (mean bbox height) so that wide-angle footage with a small GK bbox
        uses proportionally lower thresholds.
        """
        events = []

        if not gk_track.detections:
            return events

        mean_h = sum(d.bbox.height for d in gk_track.detections) / len(gk_track.detections)
        bbox_scale = max(0.3, min(1.0, mean_h / REFERENCE_BBOX_HEIGHT))

        log.info(
            "gk_detector.bbox_scale",
            keeper_role=keeper_role,
            mean_bbox_height=round(mean_h, 4),
            bbox_scale=round(bbox_scale, 3),
            num_detections=len(gk_track.detections),
        )

        events.extend(self._detect_distribution(gk_track, source_fps, keeper_role, bbox_scale))
        events.extend(self._detect_saves(gk_track, all_tracks, source_fps, keeper_role, bbox_scale))
        events.extend(self._detect_one_on_ones(gk_track, all_tracks, source_fps, keeper_role, bbox_scale))

        return events

    # -- Private detection methods --

    def _detect_distribution(
        self, gk_track: Track, fps: float, keeper_role: str = "keeper_a",
        bbox_scale: float = 1.0,
    ) -> list[Event]:
        """
        Detect goal kicks and distribution from GK track.
        Uses velocity analysis: sudden bbox position change after stationary period.
        """
        events = []
        dets = gk_track.detections
        if len(dets) < 5:
            return events

        velocities = []
        for i in range(1, len(dets)):
            dx = dets[i].bbox.center_x - dets[i-1].bbox.center_x
            dt = dets[i].timestamp - dets[i-1].timestamp
            velocities.append(abs(dx / dt) if dt > 0 else 0)

        window = 10
        for i in range(window, len(velocities)):
            pre_vel = np.mean(velocities[max(0, i-window):i])
            post_vel = np.mean(velocities[i:min(len(velocities), i+5)])

            if pre_vel < 0.01 and post_vel > 0.12 * bbox_scale:
                det = dets[i]
                event_type = self._classify_distribution_type(det)
                vel_ratio = post_vel / max(pre_vel, 0.001)
                confidence = min(0.90, 0.55 + vel_ratio * 0.01)
                events.append(Event(
                    job_id=self.job_id,
                    source_file=self.source_file,
                    event_type=event_type,
                    timestamp_start=max(0, det.timestamp - 1.0),
                    timestamp_end=det.timestamp + 3.0,
                    confidence=confidence,
                    reel_targets=[keeper_role],
                    player_track_id=gk_track.track_id,
                    is_goalkeeper_event=True,
                    frame_start=max(0, det.frame_number - int(fps)),
                    frame_end=det.frame_number + int(3 * fps),
                    bounding_box=det.bbox,
                    metadata={
                        "detection_method": "velocity_transition",
                        "pre_vel": round(float(pre_vel), 4),
                        "post_vel": round(float(post_vel), 4),
                        "keeper_role": keeper_role,
                    },
                ))

        return events

    def _detect_saves(
        self, gk_track: Track, all_tracks: list[Track], fps: float,
        keeper_role: str = "keeper_a", bbox_scale: float = 1.0,
    ) -> list[Event]:
        """
        Detect save events: GK makes sudden vertical or lateral movement
        coinciding with ball proximity.
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

        vel_threshold = (0.25 if has_ball_data else 0.35) * bbox_scale
        dive_threshold = 0.40 * bbox_scale

        for i in range(1, len(dets) - 1):
            prev, curr, nxt = dets[i-1], dets[i], dets[i+1]

            dy_prev = abs(curr.bbox.center_y - prev.bbox.center_y)
            dy_next = abs(nxt.bbox.center_y - curr.bbox.center_y)
            dt_prev = curr.timestamp - prev.timestamp
            dt_next = nxt.timestamp - curr.timestamp

            if dt_prev <= 0 or dt_next <= 0:
                continue

            v_prev = dy_prev / dt_prev
            v_next = dy_next / dt_next
            vertical_velocity = (v_prev + v_next) / 2.0

            if vertical_velocity <= vel_threshold:
                continue

            ball_distance = None
            gk_x, gk_y = curr.bbox.center_x, curr.bbox.center_y
            frame = curr.frame_number
            for f_offset in range(-2, 3):
                if (frame + f_offset) in ball_positions:
                    bx, by = ball_positions[frame + f_offset]
                    dist = ((gk_x - bx) ** 2 + (gk_y - by) ** 2) ** 0.5
                    if ball_distance is None or dist < ball_distance:
                        ball_distance = dist

            if has_ball_data and (ball_distance is None or ball_distance > 0.15):
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

        return self._merge_nearby_events(events, min_gap_sec=2.0)

    def _detect_one_on_ones(
        self, gk_track: Track, all_tracks: list[Track], fps: float,
        keeper_role: str = "keeper_a", bbox_scale: float = 1.0,
    ) -> list[Event]:
        """
        Detect one-on-one: GK moves significantly away from goal line.
        Requires 3 consecutive frames beyond threshold to filter noise.
        """
        events = []
        dets = gk_track.detections
        if len(dets) < 4:
            return events

        deviation_threshold = 0.20 * bbox_scale
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
                excess = deviation - deviation_threshold
                confidence = min(0.88, 0.65 + excess * 2.0)

                # ONE_ON_ONE goes to both the keeper reel and highlights
                reel_targets = [keeper_role, "highlights"]

                events.append(Event(
                    job_id=self.job_id,
                    source_file=self.source_file,
                    event_type=EventType.ONE_ON_ONE,
                    timestamp_start=max(0, det.timestamp - 1.0),
                    timestamp_end=det.timestamp + 4.0,
                    confidence=confidence,
                    reel_targets=reel_targets,
                    player_track_id=gk_track.track_id,
                    is_goalkeeper_event=True,
                    frame_start=max(0, det.frame_number - int(fps)),
                    frame_end=det.frame_number + int(4 * fps),
                    bounding_box=det.bbox,
                    metadata={
                        "deviation": round(deviation, 4),
                        "consecutive_frames": consecutive_count,
                        "keeper_role": keeper_role,
                    },
                ))
                consecutive_count = 0

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

        candidates: list[tuple[int, float, dict]] = []

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
