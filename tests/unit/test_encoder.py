"""
Unit tests for src/assembly/encoder.py

Tests FFmpeg command construction and clip validation logic.
Actual FFmpeg calls are mocked — no video files needed.
"""
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


@pytest.mark.unit
class TestExtractClip:
    def test_invalid_duration_returns_false(self, tmp_path: Path):
        from src.assembly.encoder import extract_clip
        result = extract_clip("source.mp4", start_sec=50.0, end_sec=30.0, output_path=str(tmp_path / "out.mp4"))
        assert result is False

    @patch("src.assembly.encoder._run_ffmpeg_extract")
    @patch("src.assembly.encoder.os.replace")
    @patch("src.assembly.encoder.Path")
    def test_stream_copy_attempted_first(self, mock_path_cls, mock_replace, mock_ffmpeg, tmp_path):
        from src.assembly.encoder import extract_clip

        # Simulate stream copy success
        mock_ffmpeg.return_value = True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value = MagicMock(st_size=1024 * 1024)
        mock_path_instance.with_suffix.return_value = mock_path_instance
        mock_path_instance.parent.mkdir = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        extract_clip("source.mp4", 10.0, 20.0, str(tmp_path / "out.mp4"), codec="copy")

        first_call = mock_ffmpeg.call_args_list[0]
        assert first_call.kwargs.get("video_codec") == "copy" or first_call[1].get("video_codec") == "copy"

    @patch("subprocess.run")
    def test_ffmpeg_command_contains_seek_before_input(self, mock_run, tmp_path):
        from src.assembly.encoder import _run_ffmpeg_extract
        mock_run.return_value = MagicMock(returncode=0)
        output = str(tmp_path / "clip.mp4")
        _run_ffmpeg_extract("source.mp4", 15.0, 10.0, output)

        cmd = mock_run.call_args[0][0]
        # -ss should appear before -i for fast seeking
        ss_idx = cmd.index("-ss")
        i_idx = cmd.index("-i")
        assert ss_idx < i_idx

    @patch("subprocess.run")
    def test_concat_writes_list_file(self, mock_run, tmp_path):
        from src.assembly.encoder import concat_clips
        mock_run.return_value = MagicMock(returncode=0)

        clip1 = tmp_path / "clip_0001.mp4"
        clip2 = tmp_path / "clip_0002.mp4"
        clip1.write_bytes(b"fake mp4 data")
        clip2.write_bytes(b"fake mp4 data")

        # Create a real output file so os.replace succeeds
        output = str(tmp_path / "reel.mp4")
        with patch("os.replace"):
            with patch("pathlib.Path.exists", return_value=True):
                concat_clips([str(clip1), str(clip2)], output)

        # Verify concat demuxer was used
        cmd = mock_run.call_args[0][0]
        assert "-f" in cmd
        assert "concat" in cmd

    @patch("subprocess.run")
    def test_concat_empty_list_returns_false(self, mock_run, tmp_path):
        from src.assembly.encoder import concat_clips
        result = concat_clips([], str(tmp_path / "reel.mp4"))
        assert result is False
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_validate_clip_calls_ffprobe(self, mock_run, tmp_path):
        from src.assembly.encoder import validate_clip
        mock_run.return_value = MagicMock(returncode=0, stdout=b"h264,3840,2160")
        result = validate_clip(str(tmp_path / "clip.mp4"))
        assert result is True
        cmd = mock_run.call_args[0][0]
        assert "ffprobe" in cmd[0]

    @patch("subprocess.run")
    def test_validate_clip_invalid_returns_false(self, mock_run, tmp_path):
        from src.assembly.encoder import validate_clip
        mock_run.return_value = MagicMock(returncode=1, stdout=b"")
        assert validate_clip(str(tmp_path / "bad.mp4")) is False
