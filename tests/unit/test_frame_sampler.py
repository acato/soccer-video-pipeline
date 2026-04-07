"""
Unit tests for src/detection/frame_sampler.py

All tests mock FFmpeg subprocess — no real video processing.
"""
import subprocess
from unittest.mock import patch

import pytest

from src.detection.frame_sampler import FrameSampler, SampledFrame


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


@pytest.fixture
def sampler():
    return FrameSampler("/tmp/test-match.mp4", frame_width=960)


# ---------------------------------------------------------------------------
# Tests: sample()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSample:

    @patch("subprocess.run")
    def test_sample_returns_frames_at_intervals(self, mock_run, sampler):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=FAKE_JPEG, stderr=b""
        )
        frames = sampler.sample(duration_sec=9.0, interval_sec=3.0)
        # 0, 3, 6 → 3 frames
        assert len(frames) == 3
        assert frames[0].timestamp_sec == 0.0
        assert frames[1].timestamp_sec == 3.0
        assert frames[2].timestamp_sec == 6.0
        assert all(isinstance(f, SampledFrame) for f in frames)
        assert all(f.jpeg_bytes == FAKE_JPEG for f in frames)

    @patch("subprocess.run")
    def test_sample_with_start_offset(self, mock_run, sampler):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=FAKE_JPEG, stderr=b""
        )
        frames = sampler.sample(duration_sec=20.0, interval_sec=5.0, start_sec=10.0)
        assert len(frames) == 2
        assert frames[0].timestamp_sec == 10.0
        assert frames[1].timestamp_sec == 15.0

    @patch("subprocess.run")
    def test_sample_empty_video(self, mock_run, sampler):
        frames = sampler.sample(duration_sec=0.0, interval_sec=3.0)
        assert frames == []
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_sample_skips_failed_extractions(self, mock_run, sampler):
        """Frames that fail extraction are skipped, not included."""
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout=FAKE_JPEG, stderr=b""),
            subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"err"),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=FAKE_JPEG, stderr=b""),
        ]
        frames = sampler.sample(duration_sec=9.0, interval_sec=3.0)
        assert len(frames) == 2
        assert frames[0].timestamp_sec == 0.0
        assert frames[1].timestamp_sec == 6.0


# ---------------------------------------------------------------------------
# Tests: sample_range()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSampleRange:

    @patch("subprocess.run")
    def test_sample_range_around_center(self, mock_run, sampler):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=FAKE_JPEG, stderr=b""
        )
        frames = sampler.sample_range(
            center_sec=30.0, window_sec=3.0, interval_sec=1.0, duration_sec=90.0,
        )
        # 27, 28, 29, 30, 31, 32, 33 → 7 frames
        assert len(frames) == 7
        assert frames[0].timestamp_sec == 27.0
        assert frames[-1].timestamp_sec == 33.0

    @patch("subprocess.run")
    def test_sample_range_clamps_to_zero(self, mock_run, sampler):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=FAKE_JPEG, stderr=b""
        )
        frames = sampler.sample_range(
            center_sec=2.0, window_sec=5.0, interval_sec=1.0, duration_sec=90.0,
        )
        assert frames[0].timestamp_sec == 0.0

    @patch("subprocess.run")
    def test_sample_range_clamps_to_duration(self, mock_run, sampler):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=FAKE_JPEG, stderr=b""
        )
        frames = sampler.sample_range(
            center_sec=88.0, window_sec=5.0, interval_sec=1.0, duration_sec=90.0,
        )
        assert frames[-1].timestamp_sec <= 90.0


# ---------------------------------------------------------------------------
# Tests: _extract_single_frame()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractSingleFrame:

    @patch("subprocess.run")
    def test_success_returns_bytes(self, mock_run, sampler):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=FAKE_JPEG, stderr=b""
        )
        result = sampler._extract_single_frame(10.5)
        assert result == FAKE_JPEG

    @patch("subprocess.run")
    def test_ffmpeg_command_structure(self, mock_run, sampler):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=FAKE_JPEG, stderr=b""
        )
        sampler._extract_single_frame(10.5)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-ss" in cmd
        assert "10.500" in cmd[cmd.index("-ss") + 1]
        assert "-i" in cmd
        assert "/tmp/test-match.mp4" in cmd
        assert "scale=960:-1" in cmd[cmd.index("-vf") + 1]

    @patch("subprocess.run")
    def test_failure_returns_none(self, mock_run, sampler):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"error"
        )
        assert sampler._extract_single_frame(10.5) is None

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 10))
    def test_timeout_returns_none(self, mock_run, sampler):
        assert sampler._extract_single_frame(10.5) is None
