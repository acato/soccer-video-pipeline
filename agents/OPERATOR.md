# Agent: OPERATOR

## Role
You own runtime health, observability, alerting, retry policies, and operational
runbooks. You ensure the pipeline runs reliably at 2am when no human is watching.

## Monitoring Stack

```
Celery → Flower (task monitoring UI, port 5555)
SQLite → periodic health check queries
FFmpeg processes → process monitor (psutil)
NAS → df -h polling every 60s
Logs → structlog → stdout → optional: Loki / log file rotation
```

## Health Check Endpoints

```python
# GET /api/v1/health
{
  "status": "healthy" | "degraded" | "unhealthy",
  "components": {
    "nas_mount": "ok" | "unreachable",
    "redis": "ok" | "unreachable",
    "celery_workers": {"active": 2, "status": "ok"},
    "scratch_gb_free": 45.2,
    "gpu_vram_free_mb": 6144,
    "jobs_pending": 3,
    "jobs_failed_last_hour": 0
  }
}

# GET /api/v1/jobs/{job_id}/status
# GET /api/v1/jobs/              → list all jobs, filterable by status
```

## Celery Worker Configuration

```python
# celery_config.py
CELERY_TASK_ROUTES = {
    "ingest.*":           {"queue": "io"},        # CPU, I/O bound
    "analysis.*":         {"queue": "gpu"},       # GPU required
    "event_detection.*":  {"queue": "cpu"},       # CPU, DB-heavy
    "gk_reel.*":          {"queue": "cpu"},
    "highlights.*":       {"queue": "cpu"},
    "render.*":           {"queue": "io"},        # CPU + I/O
}

CELERY_TASK_ACKS_LATE = True           # Don't ack until task completes
CELERY_TASK_REJECT_ON_WORKER_LOST = True
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_EXPIRES = 604800         # 7 days
```

## Retry Policies

| Task | Max Retries | Backoff | On Exhaustion |
|---|---|---|---|
| ingest_file | 3 | 60s | Mark FAILED, alert |
| analysis_pass | 2 | 300s | Mark FAILED, alert |
| render_reel | 3 | 120s | Mark FAILED, purge scratch, alert |
| nas_write | 5 | 30s | Mark FAILED, alert — DO NOT lose output |

## Alerting Rules

Implement as Celery beat periodic tasks:

```python
@celery_app.task
def check_alerts():
    alerts = []
    
    # NAS health
    if not nas_client.is_mounted():
        alerts.append(Alert("CRITICAL", "NAS unmounted"))
    
    if nas_client.scratch_budget_remaining_gb() < 10:
        alerts.append(Alert("WARNING", f"Scratch low: {gb:.1f}GB"))
    
    # Job health
    failed = db.count_jobs(status="FAILED", since=timedelta(hours=1))
    if failed > 2:
        alerts.append(Alert("WARNING", f"{failed} jobs failed in last hour"))
    
    # Stale jobs
    stale = db.count_jobs(status="RUNNING", older_than=timedelta(hours=4))
    if stale:
        alerts.append(Alert("WARNING", f"{stale} jobs running >4h"))
    
    for alert in alerts:
        log.warning("alert", **alert.dict())
        # Hook: send to webhook, email, or Slack if configured
```

## Operational Runbooks

### RB-001: NAS Unreachable
1. Check Docker bind mount: `docker exec worker df -h /mnt/nas`
2. Check NAS connectivity: `ping <nas_ip>`
3. If NAS is up but mount stale: `docker compose restart worker`
4. All RUNNING jobs will auto-retry when NAS recovers (ACKS_LATE=True)

### RB-002: GPU OOM / Worker Crash
1. `docker compose logs worker --tail=100`
2. Check VRAM: `nvidia-smi`
3. Reduce `DETECTION_BATCH_SIZE` in config and restart worker
4. Failed jobs will auto-retry

### RB-003: Scratch Disk Full
1. `docker compose exec worker python -m soccer_pipeline.ops.purge_scratch --dry-run`
2. Review: any DONE jobs with scratch not purged? → `--force-purge`
3. Check for orphaned temp files: `find /mnt/nas/scratch -mtime +1 -name "*.mp4"`
4. After purge: jobs will resume automatically

### RB-004: Job Stuck in NEEDS_REVIEW
1. `GET /api/v1/review/pending` — list pending review items
2. Review in UI (or via API): `POST /api/v1/review/{event_id} {"verdict": "accept"}`
3. Job will automatically continue to reel assembly

## Docker Compose Services

```yaml
services:
  redis:
    image: redis:7-alpine
    
  worker-gpu:
    build: docker/Dockerfile.worker
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - /path/to/nas:/mnt/nas   # ← configure for your NAS mount
    command: celery -A soccer_pipeline.celery_app worker -Q gpu -c 1
    
  worker-cpu:
    build: docker/Dockerfile.worker
    volumes:
      - /path/to/nas:/mnt/nas
    command: celery -A soccer_pipeline.celery_app worker -Q io,cpu -c 4
    
  beat:
    build: docker/Dockerfile.worker
    command: celery -A soccer_pipeline.celery_app beat
    
  api:
    build: docker/Dockerfile.api
    ports:
      - "8080:8080"
    
  flower:
    image: mher/flower:2.0
    ports:
      - "5555:5555"
```

## Key Metrics to Track

- `pipeline.job.duration_s` — histogram per job type
- `pipeline.analysis.fps` — gauge, per pass
- `pipeline.events.detected` — counter per event type per job
- `pipeline.render.duration_s` — histogram
- `pipeline.scratch.gb_used` — gauge
