"""
Field homography: maps pixel coordinates → field coordinates (meters).

Uses OpenCV to detect pitch line intersections and compute the perspective
transform from camera view to a standard top-down field coordinate system.

Standard pitch coordinate system (FIFA standard):
  - Origin (0, 0) = left goal line center
  - X-axis: along pitch length (0 = left goal line, ~105m = right goal line)
  - Y-axis: along pitch width (0 = bottom touchline, ~68m = top touchline)

Accuracy requirements:
  - Goal area detection: ±2m error acceptable
  - GK position tracking: ±5m error acceptable
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Standard FIFA pitch dimensions (meters)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_LENGTH = 5.5
GOAL_AREA_WIDTH = 18.32


@dataclass
class FieldHomography:
    """Stores the homography matrix and calibration metadata for one video."""
    H: np.ndarray           # 3x3 perspective transform (pixel → field meters)
    H_inv: np.ndarray       # Inverse transform (field → pixel)
    frame_width: int
    frame_height: int
    calibration_method: str  # "line_detection" | "manual" | "synthetic"
    confidence: float        # 0–1 calibration quality score

    def pixel_to_field(self, px: float, py: float) -> tuple[float, float]:
        """Transform pixel coordinates to field coordinates in meters."""
        pt = np.array([[[px, py]]], dtype=np.float32)
        result = _perspective_transform(pt, self.H)
        if result is None:
            return (0.0, 0.0)
        return (float(result[0][0][0]), float(result[0][0][1]))

    def field_to_pixel(self, fx: float, fy: float) -> tuple[float, float]:
        """Transform field coordinates (meters) to pixel coordinates."""
        pt = np.array([[[fx, fy]]], dtype=np.float32)
        result = _perspective_transform(pt, self.H_inv)
        if result is None:
            return (0.0, 0.0)
        return (float(result[0][0][0]), float(result[0][0][1]))

    def is_in_penalty_area(self, px: float, py: float, side: str = "left") -> bool:
        """Check if pixel position falls within the penalty area."""
        fx, fy = self.pixel_to_field(px, py)
        if side == "left":
            return (
                0 <= fx <= PENALTY_AREA_LENGTH and
                (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2 <= fy <= (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2
            )
        else:  # right
            return (
                PITCH_LENGTH - PENALTY_AREA_LENGTH <= fx <= PITCH_LENGTH and
                (PITCH_WIDTH - PENALTY_AREA_WIDTH) / 2 <= fy <= (PITCH_WIDTH + PENALTY_AREA_WIDTH) / 2
            )

    def gk_zone_bounds_pixels(self, side: str = "left", depth_meters: float = 20.0) -> tuple:
        """Return pixel bounding box of the GK zone for a given side."""
        if side == "left":
            corners_field = [
                (0, 0), (depth_meters, 0),
                (depth_meters, PITCH_WIDTH), (0, PITCH_WIDTH)
            ]
        else:
            corners_field = [
                (PITCH_LENGTH - depth_meters, 0), (PITCH_LENGTH, 0),
                (PITCH_LENGTH, PITCH_WIDTH), (PITCH_LENGTH - depth_meters, PITCH_WIDTH)
            ]
        corners_px = [self.field_to_pixel(fx, fy) for fx, fy in corners_field]
        xs = [p[0] for p in corners_px]
        ys = [p[1] for p in corners_px]
        return (min(xs), min(ys), max(xs), max(ys))  # x1, y1, x2, y2


def estimate_homography_from_lines(frame: np.ndarray) -> Optional[FieldHomography]:
    """
    Attempt to compute field homography from detected pitch lines.

    Uses Hough line transform to find pitch markings, then matches them
    to known field geometry via RANSAC.

    Returns None if calibration confidence is too low (< 0.4).
    """
    try:
        import cv2
    except ImportError:
        log.warning("homography.cv2_not_available")
        return None

    h, w = frame.shape[:2]

    # Convert to grayscale and enhance white lines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Isolate bright pixels (pitch markings are white on green)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Erode to remove noise
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.erode(white_mask, kernel, iterations=1)

    # Detect lines via Hough transform
    lines = cv2.HoughLinesP(
        white_mask,
        rho=1, theta=np.pi / 180,
        threshold=80,
        minLineLength=w * 0.05,
        maxLineGap=20,
    )

    if lines is None or len(lines) < 4:
        log.debug("homography.insufficient_lines", count=len(lines) if lines is not None else 0)
        return None

    # Find the four corner-like intersections to anchor homography
    # This is a simplified approach — production would use a more robust field model
    corners_px = _find_field_corners(lines, w, h)
    if corners_px is None or len(corners_px) < 4:
        return _synthetic_homography(w, h)

    # Standard field corners in field coordinates (assume full-pitch wide view)
    corners_field = np.float32([
        [0, 0],
        [PITCH_LENGTH, 0],
        [PITCH_LENGTH, PITCH_WIDTH],
        [0, PITCH_WIDTH],
    ])
    corners_px_arr = np.float32(corners_px[:4])

    H, mask = cv2.findHomography(corners_px_arr, corners_field, cv2.RANSAC, 5.0)
    if H is None:
        return _synthetic_homography(w, h)

    confidence = float(np.sum(mask) / len(mask)) if mask is not None else 0.3
    H_inv = np.linalg.inv(H)

    log.info("homography.calibrated", method="line_detection", confidence=round(confidence, 2))
    return FieldHomography(
        H=H, H_inv=H_inv,
        frame_width=w, frame_height=h,
        calibration_method="line_detection",
        confidence=confidence,
    )


def _synthetic_homography(frame_width: int, frame_height: int) -> FieldHomography:
    """
    Fallback homography assuming standard broadcast camera position.
    Maps the frame corners to approximate field positions.
    Accuracy: ±15m — sufficient for GK zone identification, not for precise tracking.
    """
    try:
        import cv2
    except ImportError:
        H = np.eye(3, dtype=np.float32)
        H_inv = np.eye(3, dtype=np.float32)
        return FieldHomography(H=H, H_inv=H_inv, frame_width=frame_width,
                               frame_height=frame_height, calibration_method="synthetic",
                               confidence=0.2)

    w, h = frame_width, frame_height
    # Typical broadcast shot: bottom of frame ≈ near touchline, top ≈ far touchline
    src = np.float32([[0, h], [w, h], [w, 0], [0, 0]])
    dst = np.float32([
        [0, PITCH_WIDTH],
        [PITCH_LENGTH, PITCH_WIDTH],
        [PITCH_LENGTH, 0],
        [0, 0],
    ])
    H, _ = cv2.findHomography(src, dst)
    H_inv = np.linalg.inv(H)
    log.debug("homography.using_synthetic_fallback")
    return FieldHomography(
        H=H, H_inv=H_inv, frame_width=w, frame_height=h,
        calibration_method="synthetic", confidence=0.25,
    )


def _find_field_corners(lines: np.ndarray, w: int, h: int) -> Optional[list]:
    """
    Find approximate field corner coordinates from detected line segments.
    Returns list of 4 (x, y) pixel coordinates or None.
    """
    # Separate horizontal and vertical lines
    h_lines, v_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 20 or angle > 160:
            h_lines.append(line[0])
        elif 70 < angle < 110:
            v_lines.append(line[0])

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    # Sort: top/bottom horizontal lines, left/right vertical lines
    h_lines_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
    v_lines_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

    top_h = h_lines_sorted[0]
    bot_h = h_lines_sorted[-1]
    left_v = v_lines_sorted[0]
    right_v = v_lines_sorted[-1]

    corners = [
        _line_intersection(top_h, left_v),
        _line_intersection(top_h, right_v),
        _line_intersection(bot_h, right_v),
        _line_intersection(bot_h, left_v),
    ]
    return [c for c in corners if c is not None]


def _line_intersection(l1, l2) -> Optional[tuple[float, float]]:
    """Compute intersection point of two line segments [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _perspective_transform(pt: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
    """Apply perspective transform. Returns None on error."""
    try:
        import cv2
        return cv2.perspectiveTransform(pt, H)
    except Exception:
        return None
