"""Unit tests for jersey color classifier."""
import numpy as np
import pytest
from src.detection.jersey_classifier import (
    JERSEY_COLOR_PALETTE,
    compute_jersey_similarity,
    identify_gk_by_jersey_per_half,
    identify_gk_by_known_colors,
    resolve_jersey_color,
    detect_glove_color,
    _identify_gk_simple_outlier,
    _encode_hsv_for_clustering,
    _is_referee_hue,
    _crop_hand_region,
)
from src.detection.models import BoundingBox


@pytest.mark.unit
class TestResolveJerseyColor:

    def test_known_color_returns_hsv_tuple(self):
        hsv = resolve_jersey_color("dark_blue")
        assert isinstance(hsv, tuple)
        assert len(hsv) == 3

    def test_case_insensitive(self):
        assert resolve_jersey_color("Dark_Blue") == resolve_jersey_color("dark_blue")

    def test_spaces_normalised_to_underscores(self):
        assert resolve_jersey_color("dark blue") == resolve_jersey_color("dark_blue")

    def test_hyphens_normalised(self):
        assert resolve_jersey_color("dark-blue") == resolve_jersey_color("dark_blue")

    def test_unknown_color_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown jersey color"):
            resolve_jersey_color("chartreuse_with_polka_dots")

    def test_all_palette_entries_resolve(self):
        for name in JERSEY_COLOR_PALETTE:
            hsv = resolve_jersey_color(name)
            h, s, v = hsv
            assert 0.0 <= h <= 180.0, f"{name}: H out of range"
            assert 0.0 <= s <= 1.0,   f"{name}: S out of range"
            assert 0.0 <= v <= 1.0,   f"{name}: V out of range"


@pytest.mark.unit
class TestJerseySimilarity:

    def test_identical_colors_score_1(self):
        color = (60.0, 0.8, 0.9)
        assert abs(compute_jersey_similarity(color, color) - 1.0) < 0.001

    def test_opposite_hues_score_low(self):
        red = (0.0, 0.9, 0.9)
        green = (90.0, 0.9, 0.9)  # Opposite on HSV wheel
        assert compute_jersey_similarity(red, green) < 0.5

    def test_similar_colors_score_high(self):
        blue1 = (120.0, 0.8, 0.9)
        blue2 = (125.0, 0.75, 0.85)
        assert compute_jersey_similarity(blue1, blue2) > 0.7

    def test_symmetry(self):
        c1 = (30.0, 0.7, 0.8)
        c2 = (80.0, 0.5, 0.6)
        assert abs(compute_jersey_similarity(c1, c2) - compute_jersey_similarity(c2, c1)) < 0.001


@pytest.mark.unit
class TestIdentifyGKByKnownColors:
    """Tests for supervised GK identification using known jersey colors."""

    def _make_scenario(self):
        """
        8 tracks: 3 blue (team outfield), 3 red (opponent outfield),
        1 neon_yellow (team GK, left half), 1 neon_green (opponent GK, right half).
        """
        team_gk_hsv   = resolve_jersey_color("neon_yellow")   # (35, 0.95, 0.95)
        opp_gk_hsv    = resolve_jersey_color("neon_green")     # (55, 0.95, 0.95)
        blue           = resolve_jersey_color("blue")
        red            = resolve_jersey_color("red")
        track_colors = {
            1: blue, 2: blue, 3: blue,       # team outfield (left half)
            4: team_gk_hsv,                  # team GK (left half)
            5: red,  6: red,  7: red,        # opponent outfield (right half)
            8: opp_gk_hsv,                   # opponent GK (right half)
        }
        track_positions = {
            1: 0.1, 2: 0.2, 3: 0.3, 4: 0.05,
            5: 0.7, 6: 0.8, 7: 0.9, 8: 0.95,
        }
        return track_colors, track_positions, team_gk_hsv, opp_gk_hsv

    def test_identifies_both_keepers(self):
        colors, positions, team_hsv, opp_hsv = self._make_scenario()
        result = identify_gk_by_known_colors(colors, positions, team_hsv, opp_hsv)
        assert result["keeper_a"] == 4   # team GK on left
        assert result["keeper_b"] == 8   # opponent GK on right

    def test_returns_none_when_no_close_match(self):
        """If no track is close enough to the known GK colors, return None."""
        grey = (0.0, 0.05, 0.5)
        track_colors   = {i: grey for i in range(1, 9)}
        track_positions = {i: i * 0.1 for i in range(1, 9)}
        team_gk_hsv  = resolve_jersey_color("neon_yellow")
        opp_gk_hsv   = resolve_jersey_color("neon_pink")
        result = identify_gk_by_known_colors(
            track_colors, track_positions, team_gk_hsv, opp_gk_hsv,
            min_similarity=0.80,  # Very strict — grey won't pass
        )
        assert result["keeper_a"] is None
        assert result["keeper_b"] is None

    def test_same_track_not_assigned_twice(self):
        """If both GK colors match the same track, only the better match wins."""
        neon_yellow = resolve_jersey_color("neon_yellow")
        neon_green  = resolve_jersey_color("neon_green")
        # Only one track that looks like neon_yellow; no neon_green track
        track_colors    = {1: neon_yellow, 2: resolve_jersey_color("blue")}
        track_positions = {1: 0.1, 2: 0.9}
        result = identify_gk_by_known_colors(
            track_colors, track_positions, neon_yellow, neon_green
        )
        # Track 1 should match neon_yellow (team GK); neon_green has no match
        assert result["keeper_a"] == 1
        assert result["keeper_b"] is None

    def test_empty_tracks_returns_none(self):
        result = identify_gk_by_known_colors(
            {}, {},
            resolve_jersey_color("neon_yellow"),
            resolve_jersey_color("neon_green"),
        )
        assert result == {"keeper_a": None, "keeper_b": None}


@pytest.mark.unit
class TestGKOutlierDetection:

    def test_detects_outlier_color(self):
        """6 blue/red players + 1 yellow player → yellow is outlier (GK)."""
        track_ids = [1, 2, 3, 4, 5, 6, 7]
        colors = np.float32([
            [120, 0.8, 0.9],
            [122, 0.75, 0.88],
            [118, 0.82, 0.91],
            [0,   0.9,  0.9],
            [5,   0.85, 0.87],
            [2,   0.88, 0.92],
            [30,  0.95, 0.99],  # Yellow outlier (GK)
        ])
        assert _identify_gk_simple_outlier(track_ids, colors) == 7

    def test_returns_none_when_no_clear_outlier(self):
        track_ids = [1, 2, 3, 4]
        colors = np.float32([[120, 0.8, 0.9]] * 4)
        assert _identify_gk_simple_outlier(track_ids, colors) is None

    def test_returns_none_for_small_groups(self):
        track_ids = [1, 2, 3]
        colors = np.float32([[120, 0.8, 0.9]] * 3)
        assert _identify_gk_simple_outlier(track_ids, colors) is None


@pytest.mark.unit
class TestHSVEncoding:

    def test_encoding_shape(self):
        colors = np.float32([[120, 0.8, 0.9], [30, 0.5, 0.7]])
        encoded = _encode_hsv_for_clustering(colors)
        assert encoded.shape == (2, 4)  # cos_h, sin_h, s*2, v

    def test_hue_0_equals_180_when_encoded(self):
        """Hue 0 and 180 are the same color (red) — encoding should be similar."""
        c0   = np.float32([[0,   0.8, 0.9]])
        c180 = np.float32([[180, 0.8, 0.9]])
        enc0   = _encode_hsv_for_clustering(c0)
        enc180 = _encode_hsv_for_clustering(c180)
        assert abs(enc0[0, 0] - enc180[0, 0]) < 0.01


@pytest.mark.unit
class TestIdentifyGKByJerseyPerHalf:

    def test_identifies_keeper_on_each_half(self):
        track_colors = {
            1: (120.0, 0.80, 0.90), 2: (122.0, 0.78, 0.88), 3: (118.0, 0.82, 0.91),
            4: (60.0, 0.95, 0.99),   # GK left (neon green)
            5: (0.0, 0.90, 0.90),   6: (5.0, 0.85, 0.87),   7: (2.0, 0.88, 0.92),
            8: (80.0, 0.90, 0.85),   # GK right
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
        track_colors = {i: (120.0, 0.80, 0.90) for i in range(1, 9)}
        track_positions = {i: 0.1 * i for i in range(1, 9)}
        result = identify_gk_by_jersey_per_half(track_colors, track_positions)
        assert result["keeper_a"] is None
        assert result["keeper_b"] is None

    def test_filters_referee_tracks(self):
        track_colors = {
            1: (120.0, 0.80, 0.90), 2: (122.0, 0.78, 0.88),
            3: (30.0, 0.70, 0.80),  # Referee (yellow, high sat)
            4: (60.0, 0.90, 0.95),  # GK (green outlier)
            5: (0.0, 0.90, 0.90),   6: (5.0, 0.85, 0.87),
            7: (2.0, 0.88, 0.92),   8: (80.0, 0.90, 0.85),
        }
        track_positions = {
            1: 0.10, 2: 0.20, 3: 0.30, 4: 0.05,
            5: 0.70, 6: 0.80, 7: 0.90, 8: 0.95,
        }
        result = identify_gk_by_jersey_per_half(track_colors, track_positions)
        assert result["keeper_a"] == 4  # Referee (track 3) filtered out


@pytest.mark.unit
class TestDetectGloveColor:

    def test_returns_none_for_invalid_bbox(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0)
        assert detect_glove_color(frame, bbox, (100, 100)) is None

    def test_returns_score_for_bright_gloves(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[150:200, 50:150] = [0, 255, 0]  # Green BGR — glove-like
        bbox = BoundingBox(x=0.25, y=0.0, width=0.5, height=1.0)
        result = detect_glove_color(frame, bbox, (200, 200))
        assert result is not None
        assert result > 0.0

    def test_returns_low_score_for_dark_region(self):
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
