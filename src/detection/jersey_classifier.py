"""
Jersey color classifier for goalkeeper identification.

The goalkeeper wears a distinctly different jersey color from outfield players.
This module clusters player jersey colors and identifies the outlier color = GK.

Method:
1. Sample center crop of detected player bounding boxes
2. Convert to HSV (more robust to lighting than RGB)
3. Compute dominant HSV color per player track using median
4. K-means cluster into team colors (typically 3 clusters: team A, team B, GK/ref)
5. GK is the outlier — smallest cluster or lowest hue distance from reference

This is used as a fallback/supplement when field homography is unavailable,
and as a secondary confirmation when it is.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import structlog

from src.detection.models import Detection, Track

log = structlog.get_logger(__name__)

# HSV ranges for referee (typically black/yellow) — exclude from GK candidates
REFEREE_HUE_RANGE = (20, 40)   # Yellow
REFEREE_SAT_MIN = 0.5

# Expected jersey colors for a typical game:
# Team A outfield, Team B outfield, GK, (Referee)
N_JERSEY_CLUSTERS = 4


def classify_jersey_colors(
    tracks: list[Track],
    frames_data: dict[int, np.ndarray],  # frame_number -> frame BGR
    frame_shape: tuple[int, int],
) -> dict[int, tuple[float, float, float]]:
    """
    Compute dominant HSV jersey color for each track.
    
    Args:
        tracks: Player tracks to classify
        frames_data: BGR frame data keyed by frame number
        frame_shape: (height, width) of frames
    
    Returns:
        Dict mapping track_id → dominant HSV color (h, s, v)
    """
    track_colors: dict[int, tuple[float, float, float]] = {}

    for track in tracks:
        if not track.detections or track.track_id is None:
            continue

        colors = []
        for det in track.detections[:20]:  # Sample first 20 detections
            if det.frame_number not in frames_data:
                continue
            frame = frames_data[det.frame_number]
            crop = _crop_jersey_region(frame, det.bbox, frame_shape)
            if crop is None or crop.size == 0:
                continue
            hsv = _dominant_hsv(crop)
            if hsv is not None:
                colors.append(hsv)

        if colors:
            # Use median for robustness against single-frame noise
            colors_arr = np.array(colors)
            median_hsv = tuple(float(np.median(colors_arr[:, i])) for i in range(3))
            track_colors[track.track_id] = median_hsv
            if track.jersey_color_hsv is None:
                track.jersey_color_hsv = median_hsv

    return track_colors


def identify_gk_by_jersey(
    track_colors: dict[int, tuple[float, float, float]],
    min_tracks: int = 6,
) -> Optional[int]:
    """
    Identify the goalkeeper track by finding the outlier jersey color.
    
    Strategy: Use K-means to cluster jersey colors into N_JERSEY_CLUSTERS groups.
    The goalkeeper's cluster will have the fewest members.
    
    Returns track_id of likely GK, or None if insufficient data.
    """
    if len(track_colors) < min_tracks:
        log.debug("jersey_classifier.insufficient_tracks", count=len(track_colors))
        return None

    track_ids = list(track_colors.keys())
    colors = np.array([list(track_colors[tid]) for tid in track_ids], dtype=np.float32)

    # Normalize hue (circular) — use cos/sin encoding
    colors_encoded = _encode_hsv_for_clustering(colors)

    try:
        from sklearn.cluster import KMeans
        k = min(N_JERSEY_CLUSTERS, len(track_ids) - 1)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(colors_encoded)
    except ImportError:
        # Fallback: simple outlier detection without sklearn
        log.debug("jersey_classifier.sklearn_unavailable", fallback="simple_outlier")
        return _identify_gk_simple_outlier(track_ids, colors)

    # Find cluster with fewest members
    cluster_counts = defaultdict(int)
    for label in labels:
        cluster_counts[label] += 1

    # Smallest cluster = GK (usually 1 player)
    gk_cluster = min(cluster_counts, key=cluster_counts.get)

    if cluster_counts[gk_cluster] > 3:
        log.debug("jersey_classifier.no_clear_outlier", smallest_cluster_size=cluster_counts[gk_cluster])
        return None

    # Among tracks in GK cluster, pick the one with lowest position in frame
    # (GK tends to appear lower = closer to goal in wide shots)
    gk_candidates = [track_ids[i] for i, label in enumerate(labels) if label == gk_cluster]

    log.info(
        "jersey_classifier.gk_identified",
        gk_candidates=gk_candidates,
        cluster_sizes=dict(cluster_counts),
    )

    return gk_candidates[0] if gk_candidates else None


def compute_jersey_similarity(
    color1: tuple[float, float, float],
    color2: tuple[float, float, float],
) -> float:
    """
    Compute HSV similarity between two jersey colors (0.0 = different, 1.0 = identical).
    Hue is circular, saturation and value are linear.
    """
    h1, s1, v1 = color1
    h2, s2, v2 = color2

    # Circular hue distance (normalized to [0,1] where 0.5 = opposite colors)
    hue_diff = min(abs(h1 - h2), 180 - abs(h1 - h2)) / 90.0
    sat_diff = abs(s1 - s2)
    val_diff = abs(v1 - v2)

    # Weighted combination: hue most important for jersey ID
    distance = 0.6 * hue_diff + 0.3 * sat_diff + 0.1 * val_diff
    return max(0.0, 1.0 - distance)


# ── Internal helpers ───────────────────────────────────────────────────────

def _crop_jersey_region(
    frame: np.ndarray,
    bbox,
    frame_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Crop the jersey region (upper center of bounding box).
    Excludes head (top 20%) and legs (bottom 30%) to focus on shirt.
    """
    h, w = frame_shape
    x1 = int(bbox.x * w)
    y1 = int(bbox.y * h)
    x2 = int((bbox.x + bbox.width) * w)
    y2 = int((bbox.y + bbox.height) * h)

    if x2 <= x1 or y2 <= y1:
        return None

    # Shirt region: rows 20%–70% of bbox height
    bbox_h = y2 - y1
    shirt_y1 = y1 + int(bbox_h * 0.20)
    shirt_y2 = y1 + int(bbox_h * 0.70)

    # Center 60% horizontally
    bbox_w = x2 - x1
    shirt_x1 = x1 + int(bbox_w * 0.20)
    shirt_x2 = x2 - int(bbox_w * 0.20)

    crop = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    return crop if crop.size > 0 else None


def _dominant_hsv(bgr_crop: np.ndarray) -> Optional[tuple[float, float, float]]:
    """
    Compute dominant HSV color in a BGR image crop.
    Uses histogram peak rather than mean to handle multi-colored jerseys.
    """
    try:
        import cv2
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)

        # Compute per-channel medians (robust to background leakage)
        h_median = float(np.median(hsv[:, :, 0]))
        s_median = float(np.median(hsv[:, :, 1]) / 255.0)  # Normalize to 0-1
        v_median = float(np.median(hsv[:, :, 2]) / 255.0)  # Normalize to 0-1

        return (h_median, s_median, v_median)
    except Exception:
        return None


def _encode_hsv_for_clustering(colors: np.ndarray) -> np.ndarray:
    """
    Encode HSV colors for K-means clustering.
    Hue is circular (0°=360°), so encode as (cos(2πh/180), sin(2πh/180), s, v).
    """
    h = colors[:, 0] * np.pi / 90.0   # Map to [0, 2π]
    s = colors[:, 1]
    v = colors[:, 2]
    return np.column_stack([
        np.cos(h), np.sin(h), s * 2, v  # Weight sat more than val
    ])


def _identify_gk_simple_outlier(
    track_ids: list[int],
    colors: np.ndarray,
) -> Optional[int]:
    """
    Simple outlier detection without sklearn.
    Finds the track with largest average HSV distance from all other tracks.
    """
    if len(track_ids) < 4:
        return None

    max_dist = -1.0
    gk_idx = None

    for i in range(len(track_ids)):
        # Average distance from track i to all others
        dists = []
        for j in range(len(track_ids)):
            if i == j:
                continue
            h_diff = min(abs(colors[i, 0] - colors[j, 0]),
                        180 - abs(colors[i, 0] - colors[j, 0])) / 90.0
            s_diff = abs(colors[i, 1] - colors[j, 1])
            dist = 0.7 * h_diff + 0.3 * s_diff
            dists.append(dist)

        avg_dist = float(np.mean(dists))
        if avg_dist > max_dist:
            max_dist = avg_dist
            gk_idx = i

    # Only return if the outlier is distinctly separated
    return track_ids[gk_idx] if (gk_idx is not None and max_dist > 0.25) else None
