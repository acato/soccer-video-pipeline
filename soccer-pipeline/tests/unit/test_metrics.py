"""Unit tests for src/api/metrics.py"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


@pytest.fixture
def metrics_client():
    from src.api.metrics import router, increment, _metrics
    # Reset counters
    for k in _metrics:
        _metrics[k] = 0.0

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.unit
class TestPrometheusMetrics:
    def test_metrics_endpoint_returns_200(self, metrics_client):
        r = metrics_client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type(self, metrics_client):
        r = metrics_client.get("/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_metrics_contains_job_counters(self, metrics_client):
        r = metrics_client.get("/metrics")
        assert "soccer_pipeline_jobs_submitted_total" in r.text
        assert "soccer_pipeline_jobs_completed_total" in r.text
        assert "soccer_pipeline_jobs_failed_total" in r.text

    def test_increment_counter(self, metrics_client):
        from src.api.metrics import increment, _metrics
        _metrics["jobs_completed_total"] = 0
        increment("jobs_completed_total", 3)
        assert _metrics["jobs_completed_total"] == 3.0

    def test_increment_reflects_in_response(self, metrics_client):
        from src.api.metrics import increment, _metrics
        _metrics["jobs_submitted_total"] = 0
        increment("jobs_submitted_total", 5)
        r = metrics_client.get("/metrics")
        assert "soccer_pipeline_jobs_submitted_total 5" in r.text

    def test_increment_unknown_metric_ignored(self):
        from src.api.metrics import increment
        # Should not raise
        increment("nonexistent_metric_xyz")
