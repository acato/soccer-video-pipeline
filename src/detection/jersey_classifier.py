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

# Human-readable jersey color names → HSV (H: 0-180, S: 0-1, V: 0-1).
# Used when match_config is provided so classification is supervised rather
# than relying on unsupervised outlier detection.
JERSEY_COLOR_PALETTE: dict[str, tuple[float, float, float]] = {
    # Achromatic
    "white":        (0.0,   0.05, 0.95),
    "silver":       (0.0,   0.05, 0.65),
    "gray":         (0.0,   0.06, 0.45),
    "black":        (0.0,   0.08, 0.10),
    # Reds
    "red":          (0.0,   0.85, 0.70),
    "dark_red":     (0.0,   0.85, 0.40),
    "maroon":       (0.0,   0.80, 0.28),
    "burgundy":     (170.0, 0.72, 0.30),
    # Oranges / yellows
    "orange":       (12.0,  0.90, 0.85),
    "neon_orange":  (10.0,  0.95, 0.95),
    "yellow":       (28.0,  0.85, 0.90),
    "neon_yellow":  (35.0,  0.95, 0.95),
    # Greens
    "green":        (60.0,  0.80, 0.55),
    "dark_green":   (60.0,  0.85, 0.28),
    "neon_green":   (55.0,  0.95, 0.95),
    "teal":         (88.0,  0.80, 0.55),
    # Blues
    "sky_blue":     (103.0, 0.48, 0.85),
    "light_blue":   (107.0, 0.58, 0.82),
    "blue":         (112.0, 0.82, 0.65),
    "dark_blue":    (115.0, 0.90, 0.32),
    "navy":         (116.0, 0.92, 0.20),
    # Other
    "purple":       (135.0, 0.65, 0.50),
    "pink":         (157.0, 0.45, 0.80),
    "hot_pink":     (153.0, 0.80, 0.82),
    "neon_pink":    (153.0, 0.90, 0.95),
}


def resolve_jersey_color(name: str) -> tuple[float, float, float]:
    """
    Return the HSV tuple for a named jersey color.
    Raises ValueError for unknown names.
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in JERSEY_COLOR_PALETTE:
        valid = ", ".join(sorted(JERSEY_COLOR_PALETTE))
        raise ValueError(f"Unknown jersey color {name!r}. Valid options: {valid}")
    return JERSEY_COLOR_PALETTE[key]

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


def identify_gk_by_jersey_per_half(
    track_colors: dict[int, tuple[float, float, float]],
    track_positions: dict[int, float],
    min_tracks: int = 6,
    uniqueness_threshold: float = 0.30,
) -> dict[str, Optional[int]]:
    """
    Identify one goalkeeper per half by jersey color uniqueness.

    Partitions tracks into left/right halves by mean_x position, then per half
    finds the track with the highest average HSV distance from all other
    same-half tracks. If that distance exceeds the threshold → keeper.

    Args:
        track_colors: {track_id: (h, s, v)} dominant jersey colors
        track_positions: {track_id: mean_x} normalized 0-1 horizontal position
        min_tracks: minimum total tracks needed before attempting identification
        uniqueness_threshold: minimum avg HSV distance to qualify as keeper

    Returns:
        {"keeper_a": track_id or None, "keeper_b": track_id or None}
        keeper_a = left half (mean_x < 0.5), keeper_b = right half
    """
    result: dict[str, Optional[int]] = {"keeper_a": None, "keeper_b": None}

    # Need enough tracks to meaningfully partition
    common_ids = set(track_colors.keys()) & set(track_positions.keys())
    if len(common_ids) < min_tracks:
        log.debug("jersey_per_half.insufficient_tracks", count=len(common_ids))
        return result

    # Filter out referee-hue-range tracks
    filtered_ids = [
        tid for tid in common_ids
        if not _is_referee_hue(track_colors[tid])
    ]
    if len(filtered_ids) < min_tracks:
        log.debug("jersey_per_half.too_few_after_ref_filter", count=len(filtered_ids))
        return result

    # Partition into left/right halves
    left_ids = [tid for tid in filtered_ids if track_positions[tid] < 0.5]
    right_ids = [tid for tid in filtered_ids if track_positions[tid] >= 0.5]

    for half_name, half_ids in [("keeper_a", left_ids), ("keeper_b", right_ids)]:
        if len(half_ids) < 2:
            continue
        gk_id = _find_outlier_in_half(half_ids, track_colors, uniqueness_threshold)
        if gk_id is not None:
            result[half_name] = gk_id
            log.info(
                "jersey_per_half.keeper_identified",
                half=half_name,
                track_id=gk_id,
            )

    return result


def identify_gk_by_known_colors(
    track_colors: dict[int, tuple[float, float, float]],
    track_positions: dict[int, float],
    home_gk_hsv: tuple[float, float, float],
    away_gk_hsv: tuple[float, float, float],
    outfield_colors: list[tuple[float, float, float]] | None = None,
    min_similarity: float = 0.40,
    min_separation: float = 0.03,
) -> dict[str, Optional[int]]:
    """
    Identify goalkeepers using the known GK jersey colors from match_config.

    For each known GK color, finds the track with the closest jersey color.
    Assigns keeper_a (left half, mean_x < 0.5) and keeper_b (right half) by
    position after color matching — not before — so it works even when a GK
    drifts into the wrong half early in processing.

    Args:
        track_colors: {track_id: (h, s, v)} dominant jersey colors
        track_positions: {track_id: mean_x} normalized horizontal position
        home_gk_hsv: HSV of the home team GK jersey
        away_gk_hsv: HSV of the away team GK jersey
        outfield_colors: list of outfield HSV colors; tracks matching an
            outfield color better than both GK colors are rejected
        min_similarity: minimum similarity score to accept a match (0–1)
        min_separation: minimum margin between best and second-best match
            for a given GK color; below this the match is too noisy

    Returns:
        {"keeper_a": track_id or None, "keeper_b": track_id or None}
    """
    result: dict[str, Optional[int]] = {"keeper_a": None, "keeper_b": None}

    common_ids = set(track_colors.keys()) & set(track_positions.keys())
    if not common_ids:
        log.debug(
            "jersey_classifier.gk_by_known_colors_no_data",
            track_colors_count=len(track_colors),
            track_positions_count=len(track_positions),
        )
        return result

    # Find the best-matching track for each known GK color
    best_home: tuple[Optional[int], float] = (None, -1.0)
    second_home: float = -1.0
    best_away: tuple[Optional[int], float] = (None, -1.0)
    second_away: float = -1.0

    for tid in common_ids:
        color = track_colors[tid]
        sim_home = compute_jersey_similarity(color, home_gk_hsv)
        sim_away = compute_jersey_similarity(color, away_gk_hsv)
        best_gk_sim = max(sim_home, sim_away)

        # Outfield rejection: skip tracks that look more like an outfield
        # player than either GK color (prevents field players from being
        # misidentified as GK when their jersey drifts toward a GK hue).
        if outfield_colors:
            best_outfield_sim = max(
                compute_jersey_similarity(color, of) for of in outfield_colors
            )
            if best_outfield_sim > best_gk_sim:
                continue

        if sim_home > best_home[1]:
            second_home = best_home[1]
            best_home = (tid, sim_home)
        elif sim_home > second_home:
            second_home = sim_home

        if sim_away > best_away[1]:
            second_away = best_away[1]
            best_away = (tid, sim_away)
        elif sim_away > second_away:
            second_away = sim_away

    home_id, home_sim = best_home
    away_id, away_sim = best_away

    # Reject weak matches
    if home_sim < min_similarity:
        home_id = None
    if away_sim < min_similarity:
        away_id = None

    # Reject noisy matches: best must beat runner-up by min_separation
    if home_id is not None and (home_sim - second_home) < min_separation:
        log.debug(
            "jersey_classifier.home_gk_too_close",
            best=round(home_sim, 3),
            second=round(second_home, 3),
            margin=round(home_sim - second_home, 3),
        )
        home_id = None
    if away_id is not None and (away_sim - second_away) < min_separation:
        log.debug(
            "jersey_classifier.away_gk_too_close",
            best=round(away_sim, 3),
            second=round(second_away, 3),
            margin=round(away_sim - second_away, 3),
        )
        away_id = None

    # If both colors matched the same track, keep the stronger match
    if home_id is not None and home_id == away_id:
        if home_sim >= away_sim:
            away_id = None
        else:
            home_id = None

    # Assign keeper_a / keeper_b by mean_x position
    for tid in (t for t in (home_id, away_id) if t is not None):
        role = "keeper_a" if track_positions[tid] < 0.5 else "keeper_b"
        if result[role] is not None:
            # Both keepers on the same half — put the second in the other slot
            other = "keeper_b" if role == "keeper_a" else "keeper_a"
            result[other] = tid
        else:
            result[role] = tid

    log.info(
        "jersey_classifier.gk_by_known_colors",
        keeper_a=result["keeper_a"],
        keeper_b=result["keeper_b"],
        home_sim=round(home_sim, 3) if home_id is not None else None,
        away_sim=round(away_sim, 3) if away_id is not None else None,
    )
    return result


def detect_glove_color(
    frame: np.ndarray,
    bbox,
    frame_shape: tuple[int, int],
) -> Optional[float]:
    """
    Detect goalkeeper gloves by analyzing the hand/wrist region of a bbox.

    Crops the bottom 25% of the bounding box (hand area) and measures the
    fraction of high-saturation, high-value, non-skin-tone pixels (typical
    of brightly colored GK gloves).

    Returns:
        0.0–1.0 confidence score, or None if crop is invalid.
    """
    crop = _crop_hand_region(frame, bbox, frame_shape)
    if crop is None or crop.size == 0:
        return None

    try:
        import cv2
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        h = hsv[:, :, 0].astype(float)
        s = hsv[:, :, 1].astype(float) / 255.0
        v = hsv[:, :, 2].astype(float) / 255.0

        total_pixels = h.size
        if total_pixels == 0:
            return None

        # Glove pixels: high saturation, high value, NOT skin tone (hue 5-25)
        is_saturated = s > 0.35
        is_bright = v > 0.30
        is_skin = (h >= 5) & (h <= 25) & (s > 0.2)
        glove_mask = is_saturated & is_bright & ~is_skin

        fraction = float(np.sum(glove_mask)) / total_pixels
        return min(1.0, fraction)
    except Exception:
        return None


def extract_jersey_color(
    frame: np.ndarray,
    bbox,
    frame_shape: tuple[int, int],
) -> Optional[tuple[float, float, float]]:
    """
    Extract the dominant HSV jersey color from a single detection.

    Convenience wrapper used by PlayerDetector to compute jersey colors
    per-detection while frames are still in memory.
    """
    crop = _crop_jersey_region(frame, bbox, frame_shape)
    if crop is None or crop.size == 0:
        return None
    return _dominant_hsv(crop)


def _crop_hand_region(
    frame: np.ndarray,
    bbox,
    frame_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Crop the bottom 25% of a bounding box (hand/wrist area)."""
    h, w = frame_shape
    x1 = int(bbox.x * w)
    y1 = int(bbox.y * h)
    x2 = int((bbox.x + bbox.width) * w)
    y2 = int((bbox.y + bbox.height) * h)

    if x2 <= x1 or y2 <= y1:
        return None

    bbox_h = y2 - y1
    hand_y1 = y1 + int(bbox_h * 0.75)
    hand_y2 = y2

    crop = frame[hand_y1:hand_y2, x1:x2]
    return crop if crop.size > 0 else None


def _is_referee_hue(hsv: tuple[float, float, float]) -> bool:
    """Check if an HSV color falls in the referee hue range."""
    h, s, _ = hsv
    return REFEREE_HUE_RANGE[0] <= h <= REFEREE_HUE_RANGE[1] and s >= REFEREE_SAT_MIN


def _find_outlier_in_half(
    half_ids: list[int],
    track_colors: dict[int, tuple[float, float, float]],
    threshold: float,
) -> Optional[int]:
    """Find the color outlier among tracks on one half of the pitch."""
    colors = np.array([list(track_colors[tid]) for tid in half_ids], dtype=np.float32)

    max_dist = -1.0
    best_idx = None

    for i in range(len(half_ids)):
        dists = []
        for j in range(len(half_ids)):
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
            best_idx = i

    if best_idx is not None and max_dist > threshold:
        return half_ids[best_idx]
    return None


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
