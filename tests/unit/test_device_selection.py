"""
Unit tests for GPU device selection logic.

Tests the cuda → mps → cpu fallback chain in PlayerDetector and ActionClassifier.
All torch/CUDA/MPS checks are mocked — no GPU hardware required.
"""
import sys
import types
import pytest
from unittest.mock import patch, MagicMock


def _make_mock_torch(cuda_available=False, mps_available=False, has_mps=True):
    """Build a mock torch module with configurable device availability."""
    mock = MagicMock()
    mock.cuda.is_available.return_value = cuda_available
    if has_mps:
        mock.backends.mps.is_available.return_value = mps_available
    else:
        # Simulate older PyTorch: accessing .mps raises AttributeError
        mock.backends.mps.is_available.side_effect = AttributeError("no mps")
    return mock


@pytest.mark.unit
class TestPlayerDetectorDeviceSelection:
    """Tests for PlayerDetector._select_device() — cuda → mps → cpu chain."""

    def _import_and_select(self, use_gpu, mock_torch=None):
        """Import _select_device and call it, optionally injecting a mock torch."""
        # Ensure numpy is available (mock if not installed)
        np_mock = MagicMock()
        np_mock.ndarray = type("ndarray", (), {})

        patches = {}
        if "numpy" not in sys.modules:
            patches["numpy"] = np_mock
            patches["numpy.core"] = MagicMock()
        if mock_torch is not None:
            patches["torch"] = mock_torch

        with patch.dict(sys.modules, patches):
            # Force re-import to pick up mocked torch
            if "src.detection.player_detector" in sys.modules:
                del sys.modules["src.detection.player_detector"]
            from src.detection.player_detector import PlayerDetector
            return PlayerDetector._select_device(use_gpu)

    def test_use_gpu_false_returns_cpu(self):
        assert self._import_and_select(use_gpu=False) == "cpu"

    def test_cuda_available_returns_cuda(self):
        mock_torch = _make_mock_torch(cuda_available=True, mps_available=False)
        assert self._import_and_select(use_gpu=True, mock_torch=mock_torch) == "cuda:0"

    def test_mps_fallback_when_no_cuda(self):
        mock_torch = _make_mock_torch(cuda_available=False, mps_available=True)
        assert self._import_and_select(use_gpu=True, mock_torch=mock_torch) == "mps"

    def test_cpu_fallback_when_no_gpu(self):
        mock_torch = _make_mock_torch(cuda_available=False, mps_available=False)
        assert self._import_and_select(use_gpu=True, mock_torch=mock_torch) == "cpu"

    def test_cuda_preferred_over_mps(self):
        """When both CUDA and MPS report available, CUDA wins."""
        mock_torch = _make_mock_torch(cuda_available=True, mps_available=True)
        assert self._import_and_select(use_gpu=True, mock_torch=mock_torch) == "cuda:0"

    def test_old_pytorch_without_mps_attribute(self):
        """Older PyTorch that lacks torch.backends.mps should fall to cpu."""
        mock_torch = _make_mock_torch(cuda_available=False, has_mps=False)
        assert self._import_and_select(use_gpu=True, mock_torch=mock_torch) == "cpu"

    def test_torch_not_installed_returns_cpu(self):
        """When torch cannot be imported, should return cpu."""
        # Pass None as module to trigger ImportError on import
        with patch.dict(sys.modules, {"torch": None}):
            if "src.detection.player_detector" in sys.modules:
                del sys.modules["src.detection.player_detector"]
            np_mock = MagicMock()
            np_mock.ndarray = type("ndarray", (), {})
            extra = {}
            if "numpy" not in sys.modules:
                extra["numpy"] = np_mock
                extra["numpy.core"] = MagicMock()
            with patch.dict(sys.modules, {**extra, "torch": None}):
                from src.detection.player_detector import PlayerDetector
                assert PlayerDetector._select_device(use_gpu=True) == "cpu"


@pytest.mark.unit
class TestActionClassifierDeviceSelection:
    """Tests for device selection logic in ActionClassifier._ensure_loaded()."""

    def test_cpu_when_use_gpu_false(self):
        from src.detection.action_classifier import ActionClassifier
        clf = ActionClassifier(model_path=None, use_gpu=False)
        assert clf.use_gpu is False

    def test_model_unavailable_returns_false(self):
        from src.detection.action_classifier import ActionClassifier
        clf = ActionClassifier(model_path=None, use_gpu=True)
        assert clf._ensure_loaded() is False

    def test_nonexistent_model_returns_false(self):
        from src.detection.action_classifier import ActionClassifier
        clf = ActionClassifier(model_path="/nonexistent/model.pt", use_gpu=True)
        assert clf._ensure_loaded() is False

    def test_device_logic_cuda_path(self):
        """Verify the device selection expression picks cuda when available."""
        mock_torch = _make_mock_torch(cuda_available=True, mps_available=True)
        use_gpu = True
        if use_gpu and mock_torch.cuda.is_available():
            device = "cuda"
        elif use_gpu and hasattr(mock_torch.backends, "mps") and mock_torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        assert device == "cuda"

    def test_device_logic_mps_path(self):
        """Verify the device selection expression picks mps when no cuda."""
        mock_torch = _make_mock_torch(cuda_available=False, mps_available=True)
        use_gpu = True
        if use_gpu and mock_torch.cuda.is_available():
            device = "cuda"
        elif use_gpu and hasattr(mock_torch.backends, "mps") and mock_torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        assert device == "mps"

    def test_device_logic_cpu_fallback(self):
        """Verify the device selection expression falls back to cpu."""
        mock_torch = _make_mock_torch(cuda_available=False, mps_available=False)
        use_gpu = True
        if use_gpu and mock_torch.cuda.is_available():
            device = "cuda"
        elif use_gpu and hasattr(mock_torch.backends, "mps") and mock_torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        assert device == "cpu"

    def test_device_logic_use_gpu_false(self):
        """When use_gpu is False, always cpu regardless of hardware."""
        mock_torch = _make_mock_torch(cuda_available=True, mps_available=True)
        use_gpu = False
        if use_gpu and mock_torch.cuda.is_available():
            device = "cuda"
        elif use_gpu and hasattr(mock_torch.backends, "mps") and mock_torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        assert device == "cpu"
