"""
Prometheus metrics for the soccer pipeline.

Exposes metrics at GET /metrics (Prometheus scrape endpoint).
Metrics are updated by the Celery worker tasks via the JobStore.
"""
from __future__ import annotations

import time
from pathlib import Path

import structlog
from fastapi import APIRouter

log = structlog.get_logger(__name__)
router = APIRouter()

# Metric storage — simple in-process counters (for single-worker deployments)
# In multi-worker deployments, use prometheus_client with multiprocess mode
_metrics: dict[str, float] = {
    "jobs_submitted_total": 0,
    "jobs_completed_total": 0,
    "jobs_failed_total": 0,
    "events_detected_total": 0,
    "goalkeeper_events_total": 0,
    "highlights_events_total": 0,
    "reels_assembled_total": 0,
    "processing_seconds_total": 0,
}


def increment(metric: str, value: float = 1.0) -> None:
    """Increment a counter metric."""
    if metric in _metrics:
        _metrics[metric] += value


def record_job_duration(duration_sec: float) -> None:
    _metrics["processing_seconds_total"] += duration_sec


@router.get("/metrics", tags=["ops"])
def prometheus_metrics():
    """
    Expose metrics in Prometheus text format.
    Scrape this endpoint with a Prometheus server.
    """
    lines = [
        "# HELP soccer_pipeline_jobs_submitted_total Total jobs submitted",
        "# TYPE soccer_pipeline_jobs_submitted_total counter",
        f'soccer_pipeline_jobs_submitted_total {_metrics["jobs_submitted_total"]}',
        "",
        "# HELP soccer_pipeline_jobs_completed_total Total jobs completed successfully",
        "# TYPE soccer_pipeline_jobs_completed_total counter",
        f'soccer_pipeline_jobs_completed_total {_metrics["jobs_completed_total"]}',
        "",
        "# HELP soccer_pipeline_jobs_failed_total Total jobs that failed",
        "# TYPE soccer_pipeline_jobs_failed_total counter",
        f'soccer_pipeline_jobs_failed_total {_metrics["jobs_failed_total"]}',
        "",
        "# HELP soccer_pipeline_events_detected_total Total events detected across all jobs",
        "# TYPE soccer_pipeline_events_detected_total counter",
        f'soccer_pipeline_events_detected_total {_metrics["events_detected_total"]}',
        "",
        "# HELP soccer_pipeline_goalkeeper_events_total GK-specific events detected",
        "# TYPE soccer_pipeline_goalkeeper_events_total counter",
        f'soccer_pipeline_goalkeeper_events_total {_metrics["goalkeeper_events_total"]}',
        "",
        "# HELP soccer_pipeline_highlights_events_total Highlights events detected",
        "# TYPE soccer_pipeline_highlights_events_total counter",
        f'soccer_pipeline_highlights_events_total {_metrics["highlights_events_total"]}',
        "",
        "# HELP soccer_pipeline_reels_assembled_total Total reel files assembled",
        "# TYPE soccer_pipeline_reels_assembled_total counter",
        f'soccer_pipeline_reels_assembled_total {_metrics["reels_assembled_total"]}',
        "",
        "# HELP soccer_pipeline_processing_seconds_total Total CPU/GPU seconds in detection",
        "# TYPE soccer_pipeline_processing_seconds_total counter",
        f'soccer_pipeline_processing_seconds_total {_metrics["processing_seconds_total"]}',
        "",
    ]

    # Live job counts from JobStore
    try:
        from src.config import config as cfg
        from src.ingestion.job import JobStore
        from src.ingestion.models import JobStatus

        store = JobStore(Path(cfg.WORKING_DIR) / "jobs")
        jobs = store.list_all()
        status_counts: dict[str, int] = {s.value: 0 for s in JobStatus}
        for job in jobs:
            status_counts[job.status] = status_counts.get(job.status, 0) + 1

        lines += [
            "# HELP soccer_pipeline_jobs_by_status Current job counts by status",
            "# TYPE soccer_pipeline_jobs_by_status gauge",
        ]
        for status, count in status_counts.items():
            lines.append(f'soccer_pipeline_jobs_by_status{{status="{status}"}} {count}')
        lines.append("")
    except Exception:
        pass

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("\n".join(lines), media_type="text/plain; version=0.0.4")
