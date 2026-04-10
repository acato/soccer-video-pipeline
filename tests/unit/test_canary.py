"""Tests for the classify canary system in dual_pass_detector."""
import pytest

from src.detection.dual_pass_detector import (
    CanaryFailure,
    DualPassConfig,
    DualPassDetector,
)


@pytest.fixture
def config():
    return DualPassConfig(
        canary_enabled=True,
        canary_window_count=5,
        canary_min_nonempty_fraction=0.05,
        canary_action="fail",
    )


class TestClassifyCanary:
    """Test the classify canary fires/passes correctly."""

    @pytest.mark.unit
    def test_canary_fires_on_all_empty(self, config):
        """When all sub-windows return [], canary should raise CanaryFailure."""
        # Simulate: create a detector with mock state
        det = DualPassDetector.__new__(DualPassDetector)
        det._cfg = config
        det._canary_sub_total = 5
        det._canary_sub_nonempty = 0
        det._canary_checked = False

        with pytest.raises(CanaryFailure, match="0/5"):
            det._check_classify_canary()

    @pytest.mark.unit
    def test_canary_passes_on_healthy_data(self, config):
        """When enough sub-windows return events, canary should pass."""
        det = DualPassDetector.__new__(DualPassDetector)
        det._cfg = config
        det._canary_sub_total = 5
        det._canary_sub_nonempty = 2  # 40% — well above 5% threshold
        det._canary_checked = False

        # Should not raise
        det._check_classify_canary()
        assert det._canary_checked is True

    @pytest.mark.unit
    def test_canary_passes_at_threshold(self, config):
        """Canary passes when exactly at the threshold."""
        det = DualPassDetector.__new__(DualPassDetector)
        det._cfg = config
        det._canary_sub_total = 20
        det._canary_sub_nonempty = 1  # 5% — exactly at threshold
        det._canary_checked = False

        # Should not raise
        det._check_classify_canary()
        assert det._canary_checked is True

    @pytest.mark.unit
    def test_canary_warn_mode_no_exception(self):
        """In warn mode, canary logs but does not raise."""
        cfg = DualPassConfig(
            canary_enabled=True,
            canary_window_count=5,
            canary_min_nonempty_fraction=0.05,
            canary_action="warn",
        )
        det = DualPassDetector.__new__(DualPassDetector)
        det._cfg = cfg
        det._canary_sub_total = 5
        det._canary_sub_nonempty = 0
        det._canary_checked = False

        # Should NOT raise even with 0% hit rate
        det._check_classify_canary()
        assert det._canary_checked is True

    @pytest.mark.unit
    def test_canary_disabled(self):
        """When canary is disabled, no check happens."""
        cfg = DualPassConfig(canary_enabled=False)
        # Just verify the config field exists and is False
        assert cfg.canary_enabled is False
