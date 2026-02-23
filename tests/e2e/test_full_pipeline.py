"""
End-to-end tests: full pipeline from video submission to reel output.

Requires:
  - Full Docker Compose stack running (make up)
  - FFmpeg installed
  - API reachable at http://localhost:8080

Usage:
  pytest tests/e2e/ -m e2e --timeout=600
"""
import subprocess
import time
from pathlib import Path

import pytest
import httpx


API_BASE = "http://localhost:8080"


def _api_available() -> bool:
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def _make_test_video(path: Path, duration: int = 120) -> Path:
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
    return path


pytestmark = pytest.mark.e2e
skip_no_api = pytest.mark.skipif(not _api_available(), reason="API not running")


@skip_no_api
class TestE2EGoalkeeperReel:
    def test_submit_and_wait_complete(self, tmp_path: Path):
        """Submit a 2-minute synthetic video, wait for completion."""
        video = _make_test_video(tmp_path / "match.mp4", duration=120)

        # Submit job
        resp = httpx.post(f"{API_BASE}/jobs", json={
            "nas_path": str(video),
            "reel_types": ["keeper_a"],
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

        # Verify output exists
        job_resp = httpx.get(f"{API_BASE}/jobs/{job_id}")
        job = job_resp.json()
        assert job["status"] == "complete"
        assert "keeper_a" in job["output_paths"]


@skip_no_api
class TestE2EIdempotency:
    def test_same_file_returns_same_job(self, tmp_path: Path):
        """Submitting the same file twice returns the same job_id."""
        video = _make_test_video(tmp_path / "match_idem.mp4", duration=30)

        resp1 = httpx.post(f"{API_BASE}/jobs", json={"nas_path": str(video)})
        resp2 = httpx.post(f"{API_BASE}/jobs", json={"nas_path": str(video)})

        assert resp1.json()["job_id"] == resp2.json()["job_id"]
