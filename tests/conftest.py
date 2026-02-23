"""
Shared pytest fixtures for all test levels.
Uses only synthetic data — never requires real match footage.
"""
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Environment setup — set required env vars before config import
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True, scope="session")
def set_test_env():
    os.environ.setdefault("NAS_MOUNT_PATH", "/tmp/test-nas-source")
    os.environ.setdefault("NAS_OUTPUT_PATH", "/tmp/test-nas-output")
    os.environ.setdefault("WORKING_DIR", "/tmp/test-soccer-working")
    os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
    os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    yield


# ---------------------------------------------------------------------------
# Temp directories
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    return tmp_path / "working"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Synthetic video fixtures
# ---------------------------------------------------------------------------
SYNTHETIC_VIDEO_CACHE: dict[str, Path] = {}


def _make_synthetic_video(
    width: int, height: int, fps: int, duration: int, output_path: Path
) -> Path:
    """Generate a synthetic colored video using FFmpeg lavfi source."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x1a7a1a:size={width}x{height}:rate={fps}",
        "-f", "lavfi",
        "-i", "sine=frequency=440:sample_rate=48000",
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "64k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode()}")
    return output_path


@pytest.fixture(scope="session")
def sample_video_30s(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """30-second 1280x720 synthetic video (session-scoped for speed)."""
    cache_dir = tmp_path_factory.mktemp("fixtures")
    path = cache_dir / "sample_30s.mp4"
    return _make_synthetic_video(1280, 720, 30, 30, path)


@pytest.fixture(scope="session")
def sample_video_10s(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """10-second 1280x720 synthetic video."""
    cache_dir = tmp_path_factory.mktemp("fixtures")
    path = cache_dir / "sample_10s.mp4"
    return _make_synthetic_video(1280, 720, 30, 10, path)


# ---------------------------------------------------------------------------
# Match config helpers
# ---------------------------------------------------------------------------

def make_match_config():
    """Return a minimal MatchConfig suitable for use in tests."""
    from src.ingestion.models import KitConfig, MatchConfig
    return MatchConfig(
        team=KitConfig(
            team_name="Home FC",
            outfield_color="blue",
            gk_color="neon_yellow",
        ),
        opponent=KitConfig(
            team_name="Away United",
            outfield_color="red",
            gk_color="neon_green",
        ),
    )


@pytest.fixture
def sample_match_config():
    return make_match_config()


# ---------------------------------------------------------------------------
# Event log fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_events_jsonl(tmp_path: Path) -> Path:
    """Write a deterministic sample events.jsonl file."""
    events = [
        {
            "event_id": "aaaa-0001",
            "job_id": "job-0001",
            "source_file": "match.mp4",
            "event_type": "shot_stop_diving",
            "timestamp_start": 10.0,
            "timestamp_end": 12.5,
            "confidence": 0.82,
            "reel_targets": ["keeper"],
            "player_track_id": 1,
            "is_goalkeeper_event": True,
            "frame_start": 300,
            "frame_end": 375,
            "reviewed": False,
            "review_override": None,
            "metadata": {},
        },
        {
            "event_id": "bbbb-0002",
            "job_id": "job-0001",
            "source_file": "match.mp4",
            "event_type": "goal",
            "timestamp_start": 45.0,
            "timestamp_end": 46.0,
            "confidence": 0.91,
            "reel_targets": ["highlights"],
            "player_track_id": 7,
            "is_goalkeeper_event": False,
            "frame_start": 1350,
            "frame_end": 1380,
            "reviewed": False,
            "review_override": None,
            "metadata": {},
        },
    ]
    path = tmp_path / "events.jsonl"
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


# ---------------------------------------------------------------------------
# Mock config
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("NAS_MOUNT_PATH", str(tmp_path / "nas-source"))
    monkeypatch.setenv("NAS_OUTPUT_PATH", str(tmp_path / "nas-output"))
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "working"))
    (tmp_path / "nas-source").mkdir()
    (tmp_path / "nas-output").mkdir()
    (tmp_path / "working").mkdir()
