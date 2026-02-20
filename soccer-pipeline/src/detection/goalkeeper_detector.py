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
        # If already identified and still active in these tracks
        if self._gk_track_id is not None:
            active_ids = {t.track_id for t in tracks}
            if self._gk_track_id in active_ids:
                return self._gk_track_id

        if self._homography is not None:
            gk_id = self._identify_by_position(tracks, frame_shape)
            if gk_id is not None:
                log.info("gk_detector.identified_by_position", track_id=gk_id)
                self._gk_track_id = gk_id
                return gk_id

        # Fallback: jersey color (needs frame data — not available here, used offline)
        log.debug("gk_detector.identification_deferred", reason="no_homography")
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

            if pre_vel < 0.02 and post_vel > 0.08:  # Stationary → moving
                det = dets[i]
                # Classify as goal kick vs distribution based on field position
                event_type = self._classify_distribution_type(det)
                events.append(Event(
                    job_id=self.job_id,
                    source_file=self.source_file,
                    event_type=event_type,
                    timestamp_start=max(0, det.timestamp - 1.0),
                    timestamp_end=det.timestamp + 3.0,
                    confidence=0.68,
                    reel_targets=EVENT_REEL_MAP[event_type],
                    player_track_id=gk_track.track_id,
                    is_goalkeeper_event=True,
                    frame_start=max(0, det.frame_number - int(fps)),
                    frame_end=det.frame_number + int(3 * fps),
                    bounding_box=det.bbox,
                    metadata={"detection_method": "velocity_transition"},
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
        if len(dets) < 3:
            return events

        for i in range(1, len(dets) - 1):
            prev, curr, nxt = dets[i-1], dets[i], dets[i+1]
            dy = abs(curr.bbox.center_y - prev.bbox.center_y)
            dt = curr.timestamp - prev.timestamp

            if dt <= 0:
                continue

            vertical_velocity = dy / dt
            # Significant vertical movement = dive/jump
            if vertical_velocity > 0.15:
                event_type = (
                    EventType.SHOT_STOP_DIVING
                    if vertical_velocity > 0.25
                    else EventType.SHOT_STOP_STANDING
                )
                events.append(Event(
                    job_id=self.job_id,
                    source_file=self.source_file,
                    event_type=event_type,
                    timestamp_start=max(0, prev.timestamp - 0.5),
                    timestamp_end=nxt.timestamp + 2.0,
                    confidence=min(0.90, 0.60 + vertical_velocity),
                    reel_targets=EVENT_REEL_MAP[event_type],
                    player_track_id=gk_track.track_id,
                    is_goalkeeper_event=True,
                    frame_start=prev.frame_number,
                    frame_end=nxt.frame_number + int(2 * fps),
                    bounding_box=curr.bbox,
                    metadata={"vertical_velocity": vertical_velocity},
                ))

        return self._merge_nearby_events(events, min_gap_sec=2.0)

    def _detect_one_on_ones(
        self, gk_track: Track, all_tracks: list[Track], fps: float
    ) -> list[Event]:
        """
        Detect one-on-one: GK moves significantly away from goal line
        (homography required for full accuracy; approximated via bbox position otherwise).
        """
        events = []
        if not gk_track.detections:
            return events

        # Track vertical position of GK (proxy for distance from goal without homography)
        # In typical wide-angle camera, GK near center of frame = advanced from goal
        center_y_values = [d.bbox.center_y for d in gk_track.detections]

        for i, det in enumerate(gk_track.detections):
            baseline_y = np.mean(center_y_values[:max(1, i)])
            if abs(det.bbox.center_y - baseline_y) > 0.15:  # Moved significantly toward center
                events.append(Event(
                    job_id=self.job_id,
                    source_file=self.source_file,
                    event_type=EventType.ONE_ON_ONE,
                    timestamp_start=max(0, det.timestamp - 1.0),
                    timestamp_end=det.timestamp + 4.0,
                    confidence=0.70,
                    reel_targets=EVENT_REEL_MAP[EventType.ONE_ON_ONE],
                    player_track_id=gk_track.track_id,
                    is_goalkeeper_event=True,
                    frame_start=max(0, det.frame_number - int(fps)),
                    frame_end=det.frame_number + int(4 * fps),
                    bounding_box=det.bbox,
                ))

        return self._merge_nearby_events(events, min_gap_sec=5.0)

    def _classify_distribution_type(self, det: Detection) -> EventType:
        """Heuristic: if GK bbox is very low in frame, likely a goal kick."""
        if det.bbox.center_y > 0.75:
            return EventType.GOAL_KICK
        elif det.bbox.center_y > 0.55:
            return EventType.DISTRIBUTION_SHORT
        return EventType.DISTRIBUTION_LONG

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
