"""Unit tests for jersey color classifier."""
import numpy as np
import pytest
from src.detection.jersey_classifier import (
    compute_jersey_similarity,
    _identify_gk_simple_outlier,
    _encode_hsv_for_clustering,
)


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
        """6 blue players + 1 yellow player → yellow is outlier (GK)."""
        track_ids = [1, 2, 3, 4, 5, 6, 7]
        # Team A: blue (hue ~120)
        # Team B: red (hue ~0/180)
        # GK: yellow (hue ~30) — outlier
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
        """All same color → no outlier."""
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
        """Hue 0 and 180 are the same color (red) — encoding should be similar."""
        c0 = np.float32([[0, 0.8, 0.9]])
        c180 = np.float32([[180, 0.8, 0.9]])
        enc0 = _encode_hsv_for_clustering(c0)
        enc180 = _encode_hsv_for_clustering(c180)
        # cos(0) ≈ cos(2π) = 1, cos(π*2) = 1
        assert abs(enc0[0, 0] - enc180[0, 0]) < 0.01
