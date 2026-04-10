"""Tests for the canary system in dual_pass_detector."""
import pytest

from src.detection.dual_pass_detector import (
    CanaryFailure,
    DualPassConfig,
    DualPassDetector,
)
from src.detection.models import Event, EventType
from src.detection.triage_scanner import TriageScanResult


def _make_detector(**overrides) -> DualPassDetector:
    """Create a DualPassDetector with mock state (no real file/sampler)."""
    cfg = DualPassConfig(**overrides)
    det = DualPassDetector.__new__(DualPassDetector)
    det._cfg = cfg
    return det


def _make_event(event_type: str = "throw_in", start: float = 100.0) -> Event:
    """Create a minimal Event for testing."""
    return Event(
        event_id="test",
        job_id="test",
        source_file="test.mp4",
        event_type=EventType(event_type),
        timestamp_start=start,
        timestamp_end=start + 5.0,
        confidence=0.8,
        reel_targets=[],
        is_goalkeeper_event=False,
        frame_start=int(start * 30),
        frame_end=int((start + 5) * 30),
        reviewed=False,
        review_override=None,
        metadata={},
    )


# ── Canary 1: Classify empty-response ─────────────────────────────────


class TestClassifyCanary:
    """Canary 1: detect all-empty 32B responses."""

    @pytest.mark.unit
    def test_fires_on_all_empty(self):
        det = _make_detector(canary_enabled=True, canary_action="fail")
        det._canary_sub_total = 20
        det._canary_sub_nonempty = 0
        det._canary_checked = False

        with pytest.raises(CanaryFailure, match="0/20"):
            det._check_classify_canary()

    @pytest.mark.unit
    def test_passes_on_healthy_data(self):
        det = _make_detector(canary_enabled=True, canary_action="fail")
        det._canary_sub_total = 20
        det._canary_sub_nonempty = 5  # 25%
        det._canary_checked = False

        det._check_classify_canary()
        assert det._canary_checked is True

    @pytest.mark.unit
    def test_passes_at_threshold(self):
        det = _make_detector(canary_enabled=True, canary_min_nonempty_fraction=0.05)
        det._canary_sub_total = 20
        det._canary_sub_nonempty = 1  # exactly 5%
        det._canary_checked = False

        det._check_classify_canary()  # Should not raise

    @pytest.mark.unit
    def test_warn_mode_no_exception(self):
        det = _make_detector(canary_enabled=True, canary_action="warn")
        det._canary_sub_total = 20
        det._canary_sub_nonempty = 0
        det._canary_checked = False

        det._check_classify_canary()  # Should NOT raise
        assert det._canary_checked is True

    @pytest.mark.unit
    def test_disabled(self):
        cfg = DualPassConfig(canary_enabled=False)
        assert cfg.canary_enabled is False


# ── Canary 2: Triage distribution ─────────────────────────────────────


class TestTriageCanary:
    """Canary 2: detect degenerate triage label distributions."""

    @pytest.mark.unit
    def test_fires_on_single_label_dominance(self):
        """If one label is >92% of all windows, canary should fire."""
        det = _make_detector(
            canary_enabled=True, canary_action="fail",
            triage_canary_max_single_label_pct=0.92,
        )
        result = TriageScanResult(
            flags=[],  # not checked here
            label_counts={"PLAY": 950, "SET_PIECE": 30, "ATTACK": 20},
            total_windows=1000,
        )
        with pytest.raises(CanaryFailure, match="PLAY.*95.0%"):
            det._check_triage_canary(result)

    @pytest.mark.unit
    def test_fires_on_too_few_active(self):
        """If <3% of windows are active, triage is too conservative."""
        det = _make_detector(
            canary_enabled=True, canary_action="fail",
            triage_canary_min_active_pct=0.03,
        )
        result = TriageScanResult(
            flags=[],
            label_counts={"PLAY": 900, "DEAD": 80, "SET_PIECE": 10, "ATTACK": 10},
            total_windows=1000,
        )
        with pytest.raises(CanaryFailure, match="2.0%.*active"):
            det._check_triage_canary(result)

    @pytest.mark.unit
    def test_passes_on_healthy_distribution(self):
        """Realistic distribution should pass."""
        det = _make_detector(canary_enabled=True, canary_action="fail")
        result = TriageScanResult(
            flags=[],
            label_counts={
                "PLAY": 400, "DEAD": 100,
                "SET_PIECE": 200, "ATTACK": 250, "SHOT_SAVE": 30, "GOAL": 20,
            },
            total_windows=1000,
        )
        det._check_triage_canary(result)  # Should not raise

    @pytest.mark.unit
    def test_warn_mode(self):
        """In warn mode, degenerate distribution logs but doesn't raise."""
        det = _make_detector(canary_enabled=True, canary_action="warn")
        result = TriageScanResult(
            flags=[],
            label_counts={"PLAY": 999, "SET_PIECE": 1},
            total_windows=1000,
        )
        det._check_triage_canary(result)  # Should NOT raise

    @pytest.mark.unit
    def test_zero_windows_skips(self):
        """No windows scanned = skip check."""
        det = _make_detector(canary_enabled=True, canary_action="fail")
        result = TriageScanResult(flags=[], label_counts={}, total_windows=0)
        det._check_triage_canary(result)  # Should not raise


# ── Canary 4: Event type diversity ────────────────────────────────────


class TestDiversityCanary:
    """Canary 4: detect single-type bias in classify output."""

    @pytest.mark.unit
    def test_passes_on_diverse_events(self):
        det = _make_detector(
            canary_enabled=True, diversity_canary_min_types=2,
            diversity_canary_check_after=5,
        )
        det._diversity_checked = False
        events = [
            _make_event("throw_in", 100),
            _make_event("corner_kick", 200),
            _make_event("goal_kick", 300),
            _make_event("throw_in", 400),
            _make_event("shot_on_target", 500),
        ]
        det._check_diversity_canary(events)
        assert det._diversity_checked is True

    @pytest.mark.unit
    def test_warns_on_single_type(self):
        """Single type doesn't fail (just warns), since it could be legit."""
        det = _make_detector(
            canary_enabled=True, diversity_canary_min_types=2,
        )
        det._diversity_checked = False
        events = [_make_event("throw_in", i * 100) for i in range(50)]
        # Should not raise — diversity canary only warns
        det._check_diversity_canary(events)
        assert det._diversity_checked is True


# ── Canary 5: vLLM latency ───────────────────────────────────────────


class TestLatencyCanary:
    """Canary 5: detect vLLM latency spikes."""

    @pytest.mark.unit
    def test_healthy_latencies(self):
        det = _make_detector(
            canary_enabled=True, latency_canary_max_p95_sec=180.0,
        )
        det._classify_latencies = [25.0, 30.0, 28.0, 35.0, 40.0] * 10
        det._check_latency_canary()  # Should not raise/warn

    @pytest.mark.unit
    def test_empty_latencies_skips(self):
        det = _make_detector(canary_enabled=True)
        det._classify_latencies = []
        det._check_latency_canary()  # Should not raise

    @pytest.mark.unit
    def test_high_latencies_logged(self):
        """High p95 should trigger a warning log (not an exception)."""
        det = _make_detector(
            canary_enabled=True, latency_canary_max_p95_sec=60.0,
        )
        # 95th percentile will be ~190s
        det._classify_latencies = [30.0] * 19 + [200.0]
        det._check_latency_canary()  # Should not raise — latency canary only warns
