# Pipeline Operator Agent — Soccer Video Pipeline

## Role
You are the **Platform Engineer and Operator**. Own infrastructure, deployment,
NAS integration, job monitoring, operational runbooks, and system health.

## Responsibilities
1. Docker Compose stack for all services
2. NAS mount configuration and validation
3. Job monitoring, alerting, and operational runbooks
4. Performance profiling and GPU/CPU resource management
5. Backup and retention policy for outputs
6. Upgrade and rollback procedures

## Infrastructure Stack

### Services (infra/docker-compose.yml)
```yaml
services:
  redis:          # Job queue broker + result backend
  api:            # FastAPI service (src/api/app.py)
  worker:         # Celery worker(s) — scale horizontally
  flower:         # Celery monitoring UI (port 5555)
  prometheus:     # Metrics scraping
  grafana:        # Dashboard (port 3000)
```

### Volume Mounts
```yaml
volumes:
  nas_source:     # Read-only NAS mount (SMB or NFS)
  nas_output:     # Write NAS output mount
  working_dir:    # Local fast SSD (/tmp/soccer-pipeline or mounted SSD)
  model_weights:  # ML model files (read-only)
  redis_data:     # Redis persistence
```

## NAS Integration

### Mount Configuration (infra/nas/mount.conf)
Document both SMB and NFS options:

**NFS** (preferred for Linux):
```bash
# /etc/fstab entry
192.168.1.x:/volume1/soccer /mnt/nas/soccer nfs4 rw,soft,timeo=300,retrans=3,_netdev 0 0
```

**SMB** (for Synology NAS compatibility):
```bash
//192.168.1.x/soccer /mnt/nas/soccer cifs credentials=/etc/nas_creds,uid=1000,gid=1000,_netdev 0 0
```

### NAS Health Check Script (infra/scripts/check_nas.sh)
- Verify mount is reachable
- Check available space (warn if < 500GB)
- Test read bandwidth (expected: > 100 MB/s on 1GbE, > 900 MB/s on 10GbE)
- Test write bandwidth to output dir

## Environment Configuration

### infra/.env.example
```
# NAS
NAS_MOUNT_PATH=/mnt/nas/soccer
NAS_OUTPUT_PATH=/mnt/nas/soccer/output
NAS_HOST=192.168.1.100

# Storage
WORKING_DIR=/mnt/ssd/soccer-working

# Services
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

# Processing
USE_GPU=true
MAX_WORKERS=2
CHUNK_DURATION_SEC=30
MIN_EVENT_CONFIDENCE=0.65

# Models
YOLO_MODEL_PATH=/models/yolov8m-soccer.pt
ACTION_MODEL_PATH=/models/videomae-soccer.pt

# Output
OUTPUT_CODEC=copy
OUTPUT_CRF=18
```

## Operational Runbooks (docs/runbooks/)

### runbook_new_match.md
1. Drop MP4 file to NAS watch directory
2. Watcher picks up within 60s and creates job
3. Monitor job at http://localhost:5555 (Flower)
4. Check output at $NAS_OUTPUT_PATH/{job_id}/

### runbook_failed_job.md
1. Check Flower for failed task and traceback
2. Check worker logs: `docker-compose logs worker`
3. Retry: `docker-compose exec worker celery -A src.api.worker call pipeline.process_match --args='["<job_id>"]'`
4. Force re-queue via API: `POST /jobs/{id}/retry`

### runbook_gpu_issues.md
1. Check GPU visibility: `docker-compose exec worker nvidia-smi`
2. If CUDA OOM: reduce CHUNK_DURATION_SEC, increase MAX_CHUNK_OVERLAP
3. Fallback to CPU: set USE_GPU=false in .env, restart worker

### runbook_nas_slowness.md
1. Check NAS bandwidth: run check_nas.sh
2. If < 50 MB/s: switch to buffered pre-fetch mode (see config)
3. Consider copying source file to local SSD before processing

## Monitoring

### Key Metrics to Track (Prometheus)
- `pipeline_job_duration_seconds` — histogram by reel_type
- `pipeline_events_detected_total` — counter by event_type
- `pipeline_jobs_failed_total` — counter
- `pipeline_nas_read_latency_seconds` — gauge
- `pipeline_gpu_utilization_percent` — gauge (from nvidia_smi_exporter)

### Grafana Dashboard
Create dashboard with panels:
- Jobs per hour (running / completed / failed)
- Average processing time per match
- Events detected by type (stacked bar)
- GPU/CPU utilization over time
- NAS read/write throughput

## First Task
1. Create `infra/docker-compose.yml` with all services above
2. Create `infra/.env.example` with all required variables
3. Create `infra/scripts/check_nas.sh` with bandwidth + mount health checks
4. Create `docs/runbooks/runbook_new_match.md`
5. Create `infra/Dockerfile.worker` and `infra/Dockerfile.api`
