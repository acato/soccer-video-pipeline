"""
Integration tests for the FastAPI application.
Requires: No external services (uses TestClient + mocked Celery).
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory, set_test_env):
    """Create FastAPI TestClient with mocked config paths."""
    tmp = tmp_path_factory.mktemp("api_test")
    nas_source = tmp / "nas-source"
    nas_output = tmp / "nas-output"
    working = tmp / "working"
    nas_source.mkdir()
    nas_output.mkdir()
    working.mkdir()

    import os
    os.environ["NAS_MOUNT_PATH"] = str(nas_source)
    os.environ["NAS_OUTPUT_PATH"] = str(nas_output)
    os.environ["WORKING_DIR"] = str(working)

    from src.api.app import create_app
    return TestClient(create_app())


@pytest.mark.integration
class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.integration
class TestJobsEndpoints:
    def test_list_jobs_empty(self, client):
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_nonexistent_job_404(self, client):
        resp = client.get("/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_submit_job_file_not_found(self, client):
        resp = client.post("/jobs", json={
            "nas_path": "does_not_exist.mp4",
            "reel_types": ["keeper_a"],
        })
        assert resp.status_code == 404

    def test_submit_job_invalid_reel_type(self, client, tmp_path):
        resp = client.post("/jobs", json={
            "nas_path": "match.mp4",
            "reel_types": ["invalid_reel"],
        })
        assert resp.status_code == 400

    @patch("src.api.routes.jobs.extract_metadata")
    @patch("src.api.routes.jobs.process_match_task")
    def test_submit_job_success(self, mock_task, mock_metadata, client):
        from src.ingestion.models import VideoFile
        mock_metadata.return_value = VideoFile(
            path="/mnt/nas/match.mp4",
            filename="match.mp4",
            duration_sec=5400.0,
            fps=30.0,
            width=3840,
            height=2160,
            codec="h264",
            size_bytes=15_000_000_000,
            sha256="b" * 64,
        )
        mock_task.delay = MagicMock()

        with patch("src.api.routes.jobs.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.side_effect = lambda *a, **kw: Path(*a, **kw)  # real Path except exists()
            # Simpler: just create the file in the NAS source dir
            import os
            nas = os.environ.get("NAS_MOUNT_PATH", "/tmp/test-nas-source")
            os.makedirs(nas, exist_ok=True)
            open(os.path.join(nas, "match.mp4"), "w").close()

            resp = client.post("/jobs", json={
                "nas_path": "match.mp4",
                "reel_types": ["keeper_a", "keeper_b", "highlights"],
            })
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert "job_id" in data
        mock_task.delay.assert_called_once()

    @patch("src.api.routes.jobs.extract_metadata")
    @patch("src.api.routes.jobs.process_match_task")
    def test_submit_duplicate_returns_existing_job(self, mock_task, mock_metadata, client):
        """Submitting the same SHA-256 twice returns the existing job."""
        from src.ingestion.models import VideoFile
        import os
        shared_hash = "c" * 64
        mock_metadata.return_value = VideoFile(
            path="/mnt/nas/match2.mp4", filename="match2.mp4",
            duration_sec=5400.0, fps=30.0, width=3840, height=2160,
            codec="h264", size_bytes=10_000_000_000, sha256=shared_hash,
        )
        mock_task.delay = MagicMock()

        nas = os.environ.get("NAS_MOUNT_PATH", "/tmp/test-nas-source")
        os.makedirs(nas, exist_ok=True)
        open(os.path.join(nas, "match2.mp4"), "w").close()

        resp1 = client.post("/jobs", json={"nas_path": "match2.mp4"})
        resp2 = client.post("/jobs", json={"nas_path": "match2.mp4"})

        assert resp1.status_code == 201
        assert resp2.json()["job_id"] == resp1.json()["job_id"]
        assert mock_task.delay.call_count == 1

    def test_job_status_endpoint(self, client):
        resp = client.get("/jobs/nonexistent/status")
        assert resp.status_code == 404
