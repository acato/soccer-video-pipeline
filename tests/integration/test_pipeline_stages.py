"""
Integration tests for pipeline stages using real FFmpeg + synthetic fixtures.
Requires: FFmpeg installed on test machine.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _ffmpeg_available() -> bool:
    return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0


pytestmark = pytest.mark.integration
skip_no_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg not installed")


def _make_video(path: Path, duration: int = 10, width: int = 640, height: int = 360) -> Path:
    """Generate a synthetic test video with FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=green:size={width}x{height}:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "30", "-preset", "ultrafast",
        "-c:a", "aac",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr.decode()}"
    return path


@skip_no_ffmpeg
class TestIntakeWithRealFfprobe:
    def test_extract_metadata_from_synthetic_video(self, tmp_path: Path):
        from src.ingestion.intake import extract_metadata

        video = _make_video(tmp_path / "test.mp4", duration=10, width=1280, height=720)
        vf = extract_metadata(str(video))

        assert vf.width == 1280
        assert vf.height == 720
        assert abs(vf.fps - 30.0) < 0.5
        assert abs(vf.duration_sec - 10.0) < 0.5
        assert vf.codec == "h264"
        assert len(vf.sha256) == 64
        assert vf.size_bytes > 0

    def test_extract_metadata_idempotent(self, tmp_path: Path):
        """Same file = same SHA-256 every time."""
        from src.ingestion.intake import extract_metadata
        video = _make_video(tmp_path / "idem.mp4")
        vf1 = extract_metadata(str(video))
        vf2 = extract_metadata(str(video))
        assert vf1.sha256 == vf2.sha256


@skip_no_ffmpeg
class TestClipExtraction:
    def test_extract_clip_stream_copy(self, tmp_path: Path):
        from src.assembly.encoder import extract_clip, validate_clip

        source = _make_video(tmp_path / "source.mp4", duration=30)
        output = tmp_path / "clip.mp4"

        ok = extract_clip(str(source), 5.0, 15.0, str(output), codec="copy")
        assert ok
        assert output.exists()
        assert validate_clip(str(output))

    def test_extract_clip_duration_correct(self, tmp_path: Path):
        from src.assembly.encoder import extract_clip
        from src.ingestion.intake import extract_metadata

        source = _make_video(tmp_path / "source.mp4", duration=60)
        output = tmp_path / "clip.mp4"

        extract_clip(str(source), 10.0, 25.0, str(output))
        meta = extract_metadata(str(output))
        # Stream copy snaps to prior keyframe; duration may be slightly longer
        assert meta.duration_sec >= 14.5 and meta.duration_sec < 18.0


@skip_no_ffmpeg
class TestClipConcatenation:
    def test_concat_two_clips(self, tmp_path: Path):
        from src.assembly.encoder import concat_clips, validate_clip
        from src.ingestion.intake import extract_metadata

        source = _make_video(tmp_path / "source.mp4", duration=60)

        # Extract two clips
        clip1 = tmp_path / "clip1.mp4"
        clip2 = tmp_path / "clip2.mp4"

        from src.assembly.encoder import extract_clip
        extract_clip(str(source), 5.0, 15.0, str(clip1))
        extract_clip(str(source), 30.0, 40.0, str(clip2))

        reel = tmp_path / "reel.mp4"
        ok = concat_clips([str(clip1), str(clip2)], str(reel))
        assert ok
        assert validate_clip(str(reel))

        meta = extract_metadata(str(reel))
        # Two 10s clips; each may be slightly longer due to keyframe alignment
        assert meta.duration_sec >= 18.0 and meta.duration_sec < 35.0


@skip_no_ffmpeg
class TestEventLogWithRealEvents:
    def test_end_to_end_event_log_write_read(self, tmp_path: Path, sample_events_jsonl: Path):
        """Load the fixture event log and verify read_all works correctly."""
        from src.detection.event_log import EventLog

        log = EventLog(sample_events_jsonl)
        events = log.read_all()
        assert len(events) == 2
        assert events[0].timestamp_start < events[1].timestamp_start

    def test_segmentation_from_fixture_events(self, tmp_path: Path, sample_events_jsonl: Path):
        """Test that segmentation produces correct clips from known events."""
        from src.detection.event_log import EventLog
        from src.segmentation.clipper import compute_clips

        log = EventLog(sample_events_jsonl)
        events = log.read_all()

        gk_clips = compute_clips(events, 5400.0, "goalkeeper", pre_pad=3.0, post_pad=5.0)
        hl_clips = compute_clips(events, 5400.0, "highlights", pre_pad=3.0, post_pad=5.0)

        assert len(gk_clips) == 1
        assert len(hl_clips) == 1

        # GK clip: event at 10.0–12.5, padded to 7.0–17.5
        assert abs(gk_clips[0].start_sec - 7.0) < 0.1
        assert abs(gk_clips[0].end_sec - 17.5) < 0.1

        # Highlights clip: event at 45.0–46.0, padded to 42.0–51.0
        assert abs(hl_clips[0].start_sec - 42.0) < 0.1
        assert abs(hl_clips[0].end_sec - 51.0) < 0.1
