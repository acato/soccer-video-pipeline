"""
ByteTrack-based player tracker.

Assigns persistent track IDs across frames within a chunk.
Tracking state resets between chunks (overlap handles boundary events).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import structlog

from src.detection.models import BoundingBox, Detection, Track

log = structlog.get_logger(__name__)


class PlayerTracker:
    """
    Wraps ByteTrack (via boxmot) to assign persistent IDs to player detections.

    Usage:
        tracker = PlayerTracker()
        for frame_detections in chunk_detections_by_frame:
            tracks = tracker.update(frame_detections, frame_shape)
        all_tracks = tracker.get_all_tracks()
    """

    def __init__(self, track_thresh: float = 0.5, match_thresh: float = 0.8):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self._tracker = None
        self._track_map: dict[int, Track] = {}  # track_id -> Track

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],  # (height, width)
    ) -> list[Detection]:
        """
        Feed one frame's detections to the tracker.
        Returns the same detections with track_id assigned.
        """
        tracker = self._get_tracker()
        h, w = frame_shape

        if not detections:
            # Feed empty array to allow tracker to age out lost tracks
            empty = np.empty((0, 6), dtype=np.float32)
            tracker.update(empty, np.zeros((h, w, 3), dtype=np.uint8))
            return []

        # Build [x1, y1, x2, y2, conf, cls] array for boxmot
        dets_array = np.array([
            [
                d.bbox.x * w,
                d.bbox.y * h,
                (d.bbox.x + d.bbox.width) * w,
                (d.bbox.y + d.bbox.height) * h,
                d.confidence,
                0,  # class idx (single class tracking)
            ]
            for d in detections
        ], dtype=np.float32)

        # Dummy frame (tracker needs shape but we're not using appearance features here)
        dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
        tracked = tracker.update(dets_array, dummy_frame)

        # Map tracked results back to Detection objects
        tracked_detections = list(detections)  # Copy
        if tracked is not None and len(tracked) > 0:
            for i, det in enumerate(tracked_detections):
                # Match by proximity to tracked box
                matched_id = self._match_detection_to_track(det, tracked, w, h)
                if matched_id is not None:
                    tracked_detections[i] = Detection(
                        **det.model_dump(exclude={"track_id"}),
                        track_id=matched_id,
                    )
                    # Update track record
                    if matched_id not in self._track_map:
                        self._track_map[matched_id] = Track(track_id=matched_id)
                    self._track_map[matched_id].detections.append(tracked_detections[i])

        return tracked_detections

    def get_all_tracks(self) -> list[Track]:
        """Return all tracks accumulated this chunk."""
        return list(self._track_map.values())

    def reset(self) -> None:
        """Reset tracker state for a new chunk."""
        self._tracker = None
        self._track_map = {}

    def _match_detection_to_track(
        self,
        det: Detection,
        tracked: np.ndarray,
        w: int,
        h: int,
    ) -> Optional[int]:
        """Match a detection to the nearest tracked box by IoU. Returns track_id or None."""
        det_box = np.array([
            det.bbox.x * w,
            det.bbox.y * h,
            (det.bbox.x + det.bbox.width) * w,
            (det.bbox.y + det.bbox.height) * h,
        ])
        best_iou = 0.3  # Minimum IoU to accept a match
        best_id = None
        for row in tracked:
            if len(row) < 5:
                continue
            t_box = row[:4]
            iou = _compute_iou(det_box, t_box)
            if iou > best_iou:
                best_iou = iou
                best_id = int(row[4])  # track_id is column 4 in ByteTrack output
        return best_id

    def _get_tracker(self):
        if self._tracker is None:
            try:
                from boxmot import ByteTrack
                self._tracker = ByteTrack(
                    track_thresh=self.track_thresh,
                    match_thresh=self.match_thresh,
                )
                log.info("tracker.initialized", type="ByteTrack")
            except ImportError:
                log.warning("tracker.boxmot_not_found", fallback="SimpleIoUTracker")
                self._tracker = _SimpleIoUTracker()
        return self._tracker


def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union of two [x1,y1,x2,y2] boxes."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


class _SimpleIoUTracker:
    """
    Minimal IoU-based tracker fallback when boxmot is not installed.
    Sufficient for basic testing; production should use ByteTrack.
    """

    def __init__(self):
        self._next_id = 1
        self._tracks: list[tuple[int, np.ndarray]] = []  # (id, last_box)

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if len(dets) == 0:
            self._tracks = []
            return np.empty((0, 6))

        results = []
        new_tracks = []
        used_track_ids = set()

        for det in dets:
            det_box = det[:4]
            best_iou, best_id = 0.4, None
            for tid, tbox in self._tracks:
                if tid in used_track_ids:
                    continue
                iou = _compute_iou(det_box, tbox)
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1

            used_track_ids.add(best_id)
            new_tracks.append((best_id, det_box))
            results.append([*det_box, best_id, det[4]])

        self._tracks = new_tracks
        return np.array(results, dtype=np.float32)
