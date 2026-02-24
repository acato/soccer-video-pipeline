"""Unit tests for src/api/routes/files.py"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def files_client(tmp_path, monkeypatch):
    # Create fake video files in a temp NAS directory
    for name in ["game1.mp4", "game2.mkv", "notes.txt", "clip.mov"]:
        (tmp_path / name).write_text("fake")
    monkeypatch.setenv("NAS_MOUNT_PATH", str(tmp_path))
    monkeypatch.setenv("NAS_OUTPUT_PATH", str(tmp_path / "out"))

    from src.api.routes.files import router
    app = FastAPI()
    app.include_router(router, prefix="/files")
    return TestClient(app)


@pytest.mark.unit
class TestListFiles:
    def test_returns_video_files_only(self, files_client):
        r = files_client.get("/files")
        assert r.status_code == 200
        files = r.json()
        assert "game1.mp4" in files
        assert "game2.mkv" in files
        assert "clip.mov" in files
        assert "notes.txt" not in files

    def test_returns_sorted(self, files_client):
        r = files_client.get("/files")
        files = r.json()
        assert files == sorted(files)

    def test_empty_directory(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("NAS_MOUNT_PATH", str(empty))
        monkeypatch.setenv("NAS_OUTPUT_PATH", str(tmp_path / "out"))

        from src.api.routes.files import router
        app = FastAPI()
        app.include_router(router, prefix="/files")
        client = TestClient(app)
        r = client.get("/files")
        assert r.status_code == 200
        assert r.json() == []

    def test_missing_nas_returns_503(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NAS_MOUNT_PATH", str(tmp_path / "nonexistent"))
        monkeypatch.setenv("NAS_OUTPUT_PATH", str(tmp_path / "out"))

        from src.api.routes.files import router
        app = FastAPI()
        app.include_router(router, prefix="/files")
        client = TestClient(app)
        r = client.get("/files")
        assert r.status_code == 503
