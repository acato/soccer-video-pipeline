"""
End-to-end tests: full pipeline from video submission to reel output.

Requires:
  - Full Docker Compose stack running (make up)
  - FFmpeg installed
  - API reachable at http://localhost:8080

Usage:
  pytest tests/e2e/ -m e2e --timeout=600
"""
import os
import subprocess
import time
from pathlib import Path

import pytest
import httpx


API_BASE = "http://localhost:8080"
# Local path to the directory bind-mounted into the API/worker as NAS source.
# Must match NAS_MOUNT_PATH in infra/.env (the host-side bind mount path).
_NAS_SOURCE = Path(os.getenv("NAS_MOUNT_PATH", "/tmp/e2e-nas-source"))


def _api_available() -> bool:
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def _make_test_video(filename: str, duration: int = 120) -> str:
    """Create a synthetic video in the NAS source dir; return the relative filename."""
    _NAS_SOURCE.mkdir(parents=True, exist_ok=True)
    path = _NAS_SOURCE / filename
    if not path.exists():
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=green:size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "30", "-preset", "ultrafast",
            "-c:a", "aac",
            str(path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    return filename


pytestmark = pytest.mark.e2e


def _ffmpeg_available() -> bool:
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


skip_no_api = pytest.mark.skipif(not _api_available(), reason="API not running")
skip_no_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg not installed")


@skip_no_api
@skip_no_ffmpeg
class TestE2EGoalkeeperReel:
    def test_submit_and_wait_complete(self):
        """Submit a 2-minute synthetic video, wait for completion."""
        nas_path = _make_test_video("e2e_full_match.mp4", duration=120)

        # Submit job
        resp = httpx.post(f"{API_BASE}/jobs", json={
            "nas_path": nas_path,
            "match_config": {
                "team": {"team_name": "Home FC", "outfield_color": "blue", "gk_color": "neon_yellow"},
                "opponent": {"team_name": "Away United", "outfield_color": "red", "gk_color": "neon_green"},
            },
            "reel_types": ["keeper", "highlights"],
        })
        assert resp.status_code == 201
        job_id = resp.json()["job_id"]

        # Poll for completion (max 5 minutes)
        for _ in range(60):
            time.sleep(5)
            status_resp = httpx.get(f"{API_BASE}/jobs/{job_id}/status")
            status = status_resp.json()["status"]
            if status == "complete":
                break
            if status == "failed":
                pytest.fail(f"Job failed: {status_resp.json()}")
        else:
            pytest.fail("Job did not complete within timeout")

        # Verify job completed
        job_resp = httpx.get(f"{API_BASE}/jobs/{job_id}")
        job = job_resp.json()
        assert job["status"] == "complete"
        # Output paths only populated when a real detector is used (USE_NULL_DETECTOR=false)
        if os.getenv("USE_NULL_DETECTOR", "false").lower() not in ("1", "true", "yes"):
            assert "keeper" in job["output_paths"]


@skip_no_api
@skip_no_ffmpeg
class TestE2EIdempotency:
    def test_same_file_returns_same_job(self):
        """Submitting the same file twice returns the same job_id."""
        nas_path = _make_test_video("e2e_idem_match.mp4", duration=30)

        _mc = {
            "team": {"team_name": "Home FC", "outfield_color": "blue", "gk_color": "neon_yellow"},
            "opponent": {"team_name": "Away United", "outfield_color": "red", "gk_color": "neon_green"},
        }
        resp1 = httpx.post(f"{API_BASE}/jobs", json={"nas_path": nas_path, "match_config": _mc})
        resp2 = httpx.post(f"{API_BASE}/jobs", json={"nas_path": nas_path, "match_config": _mc})

        assert resp1.json()["job_id"] == resp2.json()["job_id"]
