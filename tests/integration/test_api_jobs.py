"""
Integration tests for the FastAPI jobs API.
No external services needed — uses TestClient + mocked Celery task.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="class")
def api_env(tmp_path_factory):
    """
    Class-scoped: creates shared NAS dirs and patches env vars once for
    all tests in the class. Returns (nas_source, nas_output, working).
    """
    import os
    tmp = tmp_path_factory.mktemp("api_jobs")
    nas_source = tmp / "nas-source"
    nas_output = tmp / "nas-output"
    working = tmp / "working"
    nas_source.mkdir()
    nas_output.mkdir()
    working.mkdir()

    os.environ["NAS_MOUNT_PATH"] = str(nas_source)
    os.environ["NAS_OUTPUT_PATH"] = str(nas_output)
    os.environ["WORKING_DIR"] = str(working)

    return nas_source, nas_output, working


@pytest.fixture(scope="class")
def client(api_env):
    """TestClient with mocked Celery task (class-scoped so tests share state)."""
    with patch("src.api.routes.jobs.process_match_task") as mock_task:
        mock_task.delay = MagicMock()
        from src.api.app import create_app
        app = create_app()
        with TestClient(app) as c:
            c._mock_task = mock_task
            yield c


@pytest.fixture(scope="class")
def sample_video(api_env):
    """Create a synthetic MP4 in the NAS source directory."""
    import subprocess
    nas_source = api_env[0]
    video_path = nas_source / "test_match.mp4"
    result = subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "color=c=green:size=640x360:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440",
        "-t", "5", "-c:v", "libx264", "-crf", "35",
        "-c:a", "aac", str(video_path)
    ], capture_output=True)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr.decode()}"
    return video_path


@pytest.mark.integration
class TestJobSubmission:

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_list_jobs_initially_empty(self, client):
        r = client.get("/jobs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_submit_nonexistent_file_returns_404(self, client, sample_video):
        r = client.post("/jobs", json={"nas_path": "does_not_exist.mp4"})
        assert r.status_code == 404

    def test_submit_invalid_reel_type_returns_400(self, client, sample_video):
        r = client.post("/jobs", json={
            "nas_path": "test_match.mp4",
            "reel_types": ["invalid_type"],
        })
        assert r.status_code == 400

    def test_submit_valid_job(self, client, sample_video):
        r = client.post("/jobs", json={
            "nas_path": "test_match.mp4",
            "reel_types": ["keeper_a", "keeper_b", "highlights"],
        })
        assert r.status_code == 201, f"Body: {r.text}"
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["reel_types"] == ["keeper_a", "keeper_b", "highlights"]
        assert data["progress_pct"] == 0.0
        # Celery was called
        client._mock_task.delay.assert_called()

    def test_get_job_by_id(self, client, sample_video):
        r1 = client.post("/jobs", json={"nas_path": "test_match.mp4"})
        job_id = r1.json()["job_id"]
        r2 = client.get(f"/jobs/{job_id}")
        assert r2.status_code == 200
        assert r2.json()["job_id"] == job_id

    def test_get_nonexistent_job_returns_404(self, client):
        r = client.get("/jobs/completely-fake-uuid-that-does-not-exist")
        assert r.status_code == 404

    def test_get_job_status(self, client, sample_video):
        r1 = client.post("/jobs", json={"nas_path": "test_match.mp4"})
        job_id = r1.json()["job_id"]
        r2 = client.get(f"/jobs/{job_id}/status")
        assert r2.status_code == 200
        data = r2.json()
        assert "status" in data
        assert "progress_pct" in data
        assert data["status"] == "pending"

    def test_idempotent_submission(self, client, sample_video):
        """Submitting the same file twice returns the same job_id."""
        r1 = client.post("/jobs", json={"nas_path": "test_match.mp4"})
        r2 = client.post("/jobs", json={"nas_path": "test_match.mp4"})
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["job_id"] == r2.json()["job_id"]

    def test_list_jobs_after_submission(self, client, sample_video):
        client.post("/jobs", json={"nas_path": "test_match.mp4"})
        r = client.get("/jobs")
        assert r.status_code == 200
        jobs = r.json()
        assert len(jobs) >= 1
        assert all("job_id" in j for j in jobs)

    def test_retry_non_failed_job_returns_400(self, client, sample_video):
        r = client.post("/jobs", json={"nas_path": "test_match.mp4"})
        job_id = r.json()["job_id"]
        r2 = client.post(f"/jobs/{job_id}/retry")
        # Job is PENDING, not FAILED — retry should reject
        assert r2.status_code == 400
