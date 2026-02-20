"""Unit tests for src/api/routes/ui.py"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def ui_client():
    from src.api.routes.ui import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.unit
class TestMonitoringUI:
    def test_ui_returns_html(self, ui_client):
        r = ui_client.get("/ui")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_ui_contains_pipeline_branding(self, ui_client):
        r = ui_client.get("/ui")
        assert "Soccer Pipeline" in r.text

    def test_ui_contains_jobs_api_calls(self, ui_client):
        r = ui_client.get("/ui")
        assert "/jobs" in r.text

    def test_ui_contains_status_rendering(self, ui_client):
        r = ui_client.get("/ui")
        # Should have JS to render status badges
        assert "complete" in r.text
        assert "failed" in r.text

    def test_ui_is_self_contained(self, ui_client):
        """No external CDN dependencies."""
        r = ui_client.get("/ui")
        # No external script sources
        assert "cdn.jsdelivr" not in r.text
        assert "unpkg.com" not in r.text
