"""Unit tests for jersey color classifier."""
import numpy as np
import pytest
from src.detection.jersey_classifier import (
    compute_jersey_similarity,
    identify_gk_by_jersey_per_half,
    detect_glove_color,
    _identify_gk_simple_outlier,
    _encode_hsv_for_clustering,
    _is_referee_hue,
    _crop_hand_region,
)
from src.detection.models import BoundingBox


@pytest.mark.unit
class TestJerseySimilarity:

    def test_identical_colors_score_1(self):
        color = (60.0, 0.8, 0.9)
        assert abs(compute_jersey_similarity(color, color) - 1.0) < 0.001

    def test_opposite_hues_score_low(self):
        red = (0.0, 0.9, 0.9)
        green = (90.0, 0.9, 0.9)  # Opposite on HSV wheel
        score = compute_jersey_similarity(red, green)
        assert score < 0.5

    def test_similar_colors_score_high(self):
        blue1 = (120.0, 0.8, 0.9)
        blue2 = (125.0, 0.75, 0.85)
        score = compute_jersey_similarity(blue1, blue2)
        assert score > 0.7

    def test_symmetry(self):
        c1 = (30.0, 0.7, 0.8)
        c2 = (80.0, 0.5, 0.6)
        assert abs(compute_jersey_similarity(c1, c2) - compute_jersey_similarity(c2, c1)) < 0.001


@pytest.mark.unit
class TestGKOutlierDetection:

    def test_detects_outlier_color(self):
        """6 blue players + 1 yellow player -> yellow is outlier (GK)."""
        track_ids = [1, 2, 3, 4, 5, 6, 7]
        colors = np.float32([
            [120, 0.8, 0.9],  # track 1, team A
            [122, 0.75, 0.88], # track 2, team A
            [118, 0.82, 0.91], # track 3, team A
            [0, 0.9, 0.9],    # track 4, team B
            [5, 0.85, 0.87],  # track 5, team B
            [2, 0.88, 0.92],  # track 6, team B
            [30, 0.95, 0.99], # track 7, GK (yellow outlier)
        ])
        result = _identify_gk_simple_outlier(track_ids, colors)
        assert result == 7  # The yellow outlier

    def test_returns_none_when_no_clear_outlier(self):
        """All same color -> no outlier."""
        track_ids = [1, 2, 3, 4]
        colors = np.float32([
            [120, 0.8, 0.9],
            [121, 0.81, 0.91],
            [119, 0.79, 0.89],
            [120, 0.80, 0.90],
        ])
        result = _identify_gk_simple_outlier(track_ids, colors)
        assert result is None

    def test_returns_none_for_small_groups(self):
        """Need at least 4 tracks for reliable outlier detection."""
        track_ids = [1, 2, 3]
        colors = np.float32([[120, 0.8, 0.9]] * 3)
        result = _identify_gk_simple_outlier(track_ids, colors)
        assert result is None


@pytest.mark.unit
class TestHSVEncoding:

    def test_encoding_shape(self):
        colors = np.float32([[120, 0.8, 0.9], [30, 0.5, 0.7]])
        encoded = _encode_hsv_for_clustering(colors)
        assert encoded.shape == (2, 4)  # cos_h, sin_h, s*2, v

    def test_hue_0_equals_180_when_encoded(self):
        """Hue 0 and 180 are the same color (red) -- encoding should be similar."""
        c0 = np.float32([[0, 0.8, 0.9]])
        c180 = np.float32([[180, 0.8, 0.9]])
        enc0 = _encode_hsv_for_clustering(c0)
        enc180 = _encode_hsv_for_clustering(c180)
        assert abs(enc0[0, 0] - enc180[0, 0]) < 0.01


@pytest.mark.unit
class TestIdentifyGKByJerseyPerHalf:

    def test_identifies_keeper_on_each_half(self):
        """Two distinct GKs among blue/red teams, one per half."""
        track_colors = {
            # Left half — team A (blue) + GK (neon green, hue=60)
            1: (120.0, 0.80, 0.90),
            2: (122.0, 0.78, 0.88),
            3: (118.0, 0.82, 0.91),
            4: (60.0, 0.95, 0.99),   # GK on left (neon green)
            # Right half — team B (red) + GK (green)
            5: (0.0, 0.90, 0.90),
            6: (5.0, 0.85, 0.87),
            7: (2.0, 0.88, 0.92),
            8: (80.0, 0.90, 0.85),   # GK on right
        }
        track_positions = {
            1: 0.10, 2: 0.20, 3: 0.30, 4: 0.05,
            5: 0.70, 6: 0.80, 7: 0.90, 8: 0.95,
        }
        result = identify_gk_by_jersey_per_half(track_colors, track_positions)
        assert result["keeper_a"] == 4
        assert result["keeper_b"] == 8

    def test_returns_none_when_insufficient_tracks(self):
        track_colors = {1: (120.0, 0.8, 0.9), 2: (0.0, 0.9, 0.9)}
        track_positions = {1: 0.1, 2: 0.9}
        result = identify_gk_by_jersey_per_half(track_colors, track_positions, min_tracks=6)
        assert result["keeper_a"] is None
        assert result["keeper_b"] is None

    def test_returns_none_when_no_outlier(self):
        """All same color -> no keeper identified."""
        track_colors = {i: (120.0, 0.80, 0.90) for i in range(1, 9)}
        track_positions = {i: 0.1 * i for i in range(1, 9)}
        result = identify_gk_by_jersey_per_half(track_colors, track_positions)
        assert result["keeper_a"] is None
        assert result["keeper_b"] is None

    def test_filters_referee_tracks(self):
        """Referee-hue tracks should not be considered GK candidates."""
        track_colors = {
            1: (120.0, 0.80, 0.90),
            2: (122.0, 0.78, 0.88),
            3: (30.0, 0.70, 0.80),   # Referee (yellow, high sat)
            4: (60.0, 0.90, 0.95),   # GK (green outlier)
            5: (0.0, 0.90, 0.90),
            6: (5.0, 0.85, 0.87),
            7: (2.0, 0.88, 0.92),
            8: (80.0, 0.90, 0.85),
        }
        track_positions = {
            1: 0.10, 2: 0.20, 3: 0.30, 4: 0.05,
            5: 0.70, 6: 0.80, 7: 0.90, 8: 0.95,
        }
        result = identify_gk_by_jersey_per_half(track_colors, track_positions)
        # Referee (track 3) should be filtered, track 4 should be keeper_a
        assert result["keeper_a"] == 4


@pytest.mark.unit
class TestDetectGloveColor:

    def test_returns_none_for_invalid_bbox(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0)
        result = detect_glove_color(frame, bbox, (100, 100))
        assert result is None

    def test_returns_score_for_bright_gloves(self):
        """A frame with bright saturated pixels in hand region should score > 0."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Fill bottom 25% of bbox region with bright green (glove-like)
        frame[150:200, 50:150] = [0, 255, 0]  # Green BGR
        bbox = BoundingBox(x=0.25, y=0.0, width=0.5, height=1.0)
        result = detect_glove_color(frame, bbox, (200, 200))
        assert result is not None
        assert result > 0.0

    def test_returns_low_score_for_dark_region(self):
        """Dark pixels should give low glove score."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)  # All black
        bbox = BoundingBox(x=0.25, y=0.0, width=0.5, height=1.0)
        result = detect_glove_color(frame, bbox, (200, 200))
        assert result is not None
        assert result < 0.1


@pytest.mark.unit
class TestIsRefereeHue:
    def test_referee_yellow_detected(self):
        assert _is_referee_hue((30.0, 0.6, 0.8))

    def test_non_referee_not_detected(self):
        assert not _is_referee_hue((120.0, 0.8, 0.9))

    def test_low_saturation_not_referee(self):
        assert not _is_referee_hue((30.0, 0.3, 0.8))
