"""
Unit tests for src/utils/sleep_inhibitor.py
"""
import subprocess
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestSleepInhibitorOnMacOS:
    """Tests that run with sys.platform patched to 'darwin'."""

    @patch("src.utils.sleep_inhibitor.sys")
    @patch("src.utils.sleep_inhibitor.subprocess.Popen")
    def test_caffeinate_starts_and_stops(self, mock_popen, mock_sys):
        mock_sys.platform = "darwin"
        proc = MagicMock()
        proc.pid = 12345
        mock_popen.return_value = proc

        from src.utils.sleep_inhibitor import SleepInhibitor

        with SleepInhibitor(job_id="job-1", enabled=True):
            mock_popen.assert_called_once_with(
                ["caffeinate", "-i"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)

    @patch("src.utils.sleep_inhibitor.sys")
    @patch("src.utils.sleep_inhibitor.subprocess.Popen")
    def test_cleanup_on_exception(self, mock_popen, mock_sys):
        mock_sys.platform = "darwin"
        proc = MagicMock()
        proc.pid = 99
        mock_popen.return_value = proc

        from src.utils.sleep_inhibitor import SleepInhibitor

        with pytest.raises(RuntimeError):
            with SleepInhibitor(job_id="job-err", enabled=True):
                raise RuntimeError("boom")

        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)

    @patch("src.utils.sleep_inhibitor.sys")
    @patch("src.utils.sleep_inhibitor.subprocess.Popen")
    def test_handles_already_exited_process(self, mock_popen, mock_sys):
        mock_sys.platform = "darwin"
        proc = MagicMock()
        proc.pid = 42
        proc.terminate.side_effect = ProcessLookupError
        mock_popen.return_value = proc

        from src.utils.sleep_inhibitor import SleepInhibitor

        # Should not raise
        with SleepInhibitor(job_id="job-gone", enabled=True):
            pass

    @patch("src.utils.sleep_inhibitor.sys")
    @patch("src.utils.sleep_inhibitor.subprocess.Popen")
    def test_double_exit_is_safe(self, mock_popen, mock_sys):
        mock_sys.platform = "darwin"
        proc = MagicMock()
        proc.pid = 7
        mock_popen.return_value = proc

        from src.utils.sleep_inhibitor import SleepInhibitor

        inhibitor = SleepInhibitor(job_id="job-dbl", enabled=True)
        inhibitor.__enter__()
        inhibitor.__exit__(None, None, None)
        # Second exit should be a no-op (proc is None)
        inhibitor.__exit__(None, None, None)
        proc.terminate.assert_called_once()

    @patch("src.utils.sleep_inhibitor.sys")
    @patch("src.utils.sleep_inhibitor.subprocess.Popen")
    def test_caffeinate_not_found(self, mock_popen, mock_sys):
        mock_sys.platform = "darwin"
        mock_popen.side_effect = FileNotFoundError

        from src.utils.sleep_inhibitor import SleepInhibitor

        # Should not raise — logs a warning and continues
        with SleepInhibitor(job_id="job-nf", enabled=True):
            pass


@pytest.mark.unit
class TestSleepInhibitorNonMacOS:
    """Tests that run with sys.platform patched to a non-macOS value."""

    @patch("src.utils.sleep_inhibitor.sys")
    @patch("src.utils.sleep_inhibitor.subprocess.Popen")
    def test_noop_on_linux(self, mock_popen, mock_sys):
        mock_sys.platform = "linux"

        from src.utils.sleep_inhibitor import SleepInhibitor

        with SleepInhibitor(job_id="job-lin", enabled=True):
            pass

        mock_popen.assert_not_called()


@pytest.mark.unit
class TestSleepInhibitorDisabled:
    """Tests when the feature is disabled via config."""

    @patch("src.utils.sleep_inhibitor.sys")
    @patch("src.utils.sleep_inhibitor.subprocess.Popen")
    def test_noop_when_disabled(self, mock_popen, mock_sys):
        mock_sys.platform = "darwin"

        from src.utils.sleep_inhibitor import SleepInhibitor

        with SleepInhibitor(job_id="job-off", enabled=False):
            pass

        mock_popen.assert_not_called()
