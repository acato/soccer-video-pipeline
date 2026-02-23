"""
End-to-end test: full pipeline from file submission to GK reel output.

Requires:
  - Full docker-compose stack running
  - Synthetic test fixture available
  - FFmpeg installed

This test uses a NullDetector override to run the pipeline without real ML models,
validating all infrastructure (queue, storage, assembly) without GPU requirements.
"""
import os
import time
from pathlib import Path

import pytest
import httpx


API_URL = os.getenv("API_URL", "http://localhost:8080")
MAX_WAIT_SEC = 300


@pytest.mark.e2e
class TestGoalkeeperReelE2E:

    @pytest.fixture(scope="class", autouse=True)
    def ensure_fixture_exists(self, tmp_path_factory):
        """Generate a synthetic test video if not already present."""
        import subprocess
        fixture_path = Path("/tmp/e2e_fixtures")
        fixture_path.mkdir(exist_ok=True)
        video = fixture_path / "e2e_match.mp4"
        if not video.exists():
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "color=c=green:size=1280x720:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440",
                "-t", "120",  # 2 minutes
                "-c:v", "libx264", "-crf", "28",
                "-c:a", "aac", str(video)
            ], check=True, capture_output=True)
        return video

    def test_api_health(self):
        r = httpx.get(f"{API_URL}/health", timeout=10)
        assert r.status_code == 200

    def test_submit_and_complete(self):
        """Submit 2-minute synthetic match, wait for completion, verify reel exists."""
        r = httpx.post(
            f"{API_URL}/jobs",
            json={
                "nas_path": "e2e_match.mp4",
                "match_config": {
                    "team": {"team_name": "Home FC", "outfield_color": "blue", "gk_color": "neon_yellow"},
                    "opponent": {"team_name": "Away United", "outfield_color": "red", "gk_color": "neon_green"},
                },
                "reel_types": ["keeper"],
            },
            timeout=30,
        )
        assert r.status_code == 201
        job_id = r.json()["job_id"]

        # Poll until complete or timeout
        start = time.time()
        while time.time() - start < MAX_WAIT_SEC:
            status_r = httpx.get(f"{API_URL}/jobs/{job_id}/status", timeout=10)
            assert status_r.status_code == 200
            status = status_r.json()

            if status["status"] == "complete":
                break
            if status["status"] == "failed":
                pytest.fail(f"Job failed: {status.get('error')}")

            time.sleep(5)
        else:
            pytest.fail(f"Job did not complete within {MAX_WAIT_SEC}s")

        # Verify reel info accessible
        reel_r = httpx.get(f"{API_URL}/reels/{job_id}/keeper", timeout=10)
        assert reel_r.status_code == 200
        reel_info = reel_r.json()
        assert reel_info["size_bytes"] > 0

    def test_idempotency(self):
        """Running same job twice returns same output."""
        _mc = {
            "team": {"team_name": "Home FC", "outfield_color": "blue", "gk_color": "neon_yellow"},
            "opponent": {"team_name": "Away United", "outfield_color": "red", "gk_color": "neon_green"},
        }
        r1 = httpx.post(
            f"{API_URL}/jobs",
            json={"nas_path": "e2e_match.mp4", "match_config": _mc},
            timeout=30,
        )
        r2 = httpx.post(
            f"{API_URL}/jobs",
            json={"nas_path": "e2e_match.mp4", "match_config": _mc},
            timeout=30,
        )
        assert r1.json()["job_id"] == r2.json()["job_id"]
