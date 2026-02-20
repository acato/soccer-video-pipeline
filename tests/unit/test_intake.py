"""
Unit tests for src/ingestion/intake.py

Tests ffprobe metadata extraction and file validation.
All tests use synthetic fixtures — no real footage required.
"""
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ffprobe_output(
    duration: float = 90.0,
    width: int = 3840,
    height: int = 2160,
    fps_str: str = "30/1",
    codec: str = "h264",
    size_bytes: int = 10_000_000_000,
) -> str:
    """Build fake ffprobe JSON output."""
    return json.dumps({
        "streams": [
            {
                "codec_type": "video",
                "codec_name": codec,
                "width": width,
                "height": height,
                "r_frame_rate": fps_str,
                "nb_frames": str(int(float(fps_str.split("/")[0]) / float(fps_str.split("/")[1]) * duration)),
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
            }
        ],
        "format": {
            "duration": str(duration),
            "size": str(size_bytes),
            "filename": "/mnt/nas/match.mp4",
        }
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFfprobeParser:
    """Test ffprobe output parsing logic (mocked subprocess)."""

    @pytest.mark.unit
    def test_parse_4k30_h264(self):
        """Standard 4K 30fps H.264 file parses correctly."""
        from src.ingestion.intake import parse_ffprobe_output

        raw = _make_ffprobe_output(duration=5400.0, width=3840, height=2160, fps_str="30/1", codec="h264")
        meta = parse_ffprobe_output(json.loads(raw), "/mnt/nas/match.mp4")

        assert meta.width == 3840
        assert meta.height == 2160
        assert abs(meta.fps - 30.0) < 0.01
        assert meta.codec == "h264"
        assert abs(meta.duration_sec - 5400.0) < 0.1

    @pytest.mark.unit
    def test_parse_4k60_h265(self):
        """4K 60fps H.265 file parses correctly."""
        from src.ingestion.intake import parse_ffprobe_output

        raw = _make_ffprobe_output(duration=3600.0, width=3840, height=2160, fps_str="60/1", codec="hevc")
        meta = parse_ffprobe_output(json.loads(raw), "/mnt/nas/match.mp4")

        assert abs(meta.fps - 60.0) < 0.01
        assert meta.codec == "hevc"

    @pytest.mark.unit
    def test_fractional_fps(self):
        """29.97fps (30000/1001) parsed correctly."""
        from src.ingestion.intake import parse_ffprobe_output

        raw = _make_ffprobe_output(fps_str="30000/1001")
        meta = parse_ffprobe_output(json.loads(raw), "/mnt/nas/match.mp4")
        assert abs(meta.fps - 29.97) < 0.01

    @pytest.mark.unit
    def test_no_video_stream_raises(self):
        """Files with no video stream raise ValueError."""
        from src.ingestion.intake import parse_ffprobe_output

        data = {"streams": [{"codec_type": "audio"}], "format": {"duration": "10", "size": "1000", "filename": "x.mp3"}}
        with pytest.raises(ValueError, match="No video stream"):
            parse_ffprobe_output(data, "x.mp3")

    @pytest.mark.unit
    def test_unsupported_codec_raises(self):
        """Non H.264/H.265 codecs raise ValueError."""
        from src.ingestion.intake import parse_ffprobe_output

        raw = _make_ffprobe_output(codec="vp9")
        with pytest.raises(ValueError, match="Unsupported codec"):
            parse_ffprobe_output(json.loads(raw), "match.mp4")


class TestFileValidation:
    """Test file path and format validation."""

    @pytest.mark.unit
    def test_missing_file_raises(self, tmp_path: Path):
        from src.ingestion.intake import validate_video_path

        with pytest.raises(FileNotFoundError):
            validate_video_path(str(tmp_path / "nonexistent.mp4"))

    @pytest.mark.unit
    def test_non_mp4_extension_raises(self, tmp_path: Path):
        from src.ingestion.intake import validate_video_path

        f = tmp_path / "match.avi"
        f.touch()
        with pytest.raises(ValueError, match="extension"):
            validate_video_path(str(f))

    @pytest.mark.unit
    def test_valid_mp4_path_passes(self, tmp_path: Path):
        from src.ingestion.intake import validate_video_path

        f = tmp_path / "match.mp4"
        f.touch()
        assert validate_video_path(str(f)) == str(f)


class TestSha256:
    @pytest.mark.unit
    def test_sha256_deterministic(self, tmp_path: Path):
        from src.ingestion.intake import compute_sha256

        f = tmp_path / "test.mp4"
        f.write_bytes(b"hello world video content")
        h1 = compute_sha256(str(f))
        h2 = compute_sha256(str(f))
        assert h1 == h2
        assert len(h1) == 64

    @pytest.mark.unit
    def test_different_files_different_hash(self, tmp_path: Path):
        from src.ingestion.intake import compute_sha256

        f1 = tmp_path / "a.mp4"
        f2 = tmp_path / "b.mp4"
        f1.write_bytes(b"content one")
        f2.write_bytes(b"content two")
        assert compute_sha256(str(f1)) != compute_sha256(str(f2))
