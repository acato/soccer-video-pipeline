"""
Central configuration for the Soccer Video Processing Pipeline.
All values are overridable via environment variables.
Required values (no default) will raise on startup if not set.
"""
import os
from pathlib import Path


def _req(key: str) -> str:
    """Get a required env var or raise a clear error."""
    value = os.getenv(key)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy infra/.env.example to infra/.env and configure it."
        )
    return value


def _opt(key: str, default: str) -> str:
    return os.getenv(key, default)


def _bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


# ---------------------------------------------------------------------------
# NAS / Storage
# ---------------------------------------------------------------------------
NAS_MOUNT_PATH: str = _req("NAS_MOUNT_PATH")
"""Read-only path where source match videos are stored."""

NAS_OUTPUT_PATH: str = _req("NAS_OUTPUT_PATH")
"""Writable path on NAS where assembled reels are written."""

WORKING_DIR: str = _opt("WORKING_DIR", "/tmp/soccer-pipeline")
"""Local fast storage for intermediate files (chunks, temp clips)."""

# ---------------------------------------------------------------------------
# Video Processing
# ---------------------------------------------------------------------------
CHUNK_DURATION_SEC: int = _int("CHUNK_DURATION_SEC", 30)
"""How many seconds of video to process per detection chunk."""

CHUNK_OVERLAP_SEC: float = _float("CHUNK_OVERLAP_SEC", 2.0)
"""Overlap between consecutive chunks to avoid missing events at boundaries."""

PRE_EVENT_PAD_SEC: float = _float("PRE_EVENT_PAD_SEC", 3.0)
"""Seconds of context to include before detected event start."""

POST_EVENT_PAD_SEC: float = _float("POST_EVENT_PAD_SEC", 5.0)
"""Seconds of context to include after detected event end."""

# ---------------------------------------------------------------------------
# Detection / ML
# ---------------------------------------------------------------------------
MIN_EVENT_CONFIDENCE: float = _float("MIN_EVENT_CONFIDENCE", 0.65)
"""Global minimum confidence threshold; per-event overrides in event_taxonomy."""

YOLO_MODEL_PATH: str = _opt("YOLO_MODEL_PATH", "/tmp/soccer-pipeline/yolov8m.pt")
"""Path to YOLOv8 model weights for player/ball detection."""

ACTION_MODEL_PATH: str = _opt("ACTION_MODEL_PATH", "/models/videomae-soccer.pt")
"""Path to action recognition model weights."""

USE_GPU: bool = _bool("USE_GPU", True)
"""Use NVIDIA GPU for inference if available. Auto-falls back to CPU."""


YOLO_INFERENCE_SIZE: int = _int("YOLO_INFERENCE_SIZE", 1280)
"""Input resolution for YOLO inference (width). Images downscaled before inference."""

DETECTION_FRAME_STEP: int = _int("DETECTION_FRAME_STEP", 3)
"""Process every Nth frame during detection pass (3 = 10fps effective at 30fps source)."""

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_CODEC: str = _opt("OUTPUT_CODEC", "copy")
"""FFmpeg video codec for output. 'copy' for stream copy (fast), 'libx264' to re-encode."""

OUTPUT_CRF: int = _int("OUTPUT_CRF", 18)
"""CRF quality for re-encoding (lower = better quality, larger file). Ignored with 'copy'."""

OUTPUT_AUDIO_CODEC: str = _opt("OUTPUT_AUDIO_CODEC", "copy")
"""FFmpeg audio codec. 'copy' or 'aac'."""

# ---------------------------------------------------------------------------
# Job Queue (Celery + Redis)
# ---------------------------------------------------------------------------
CELERY_BROKER_URL: str = _opt("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND: str = _opt("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
REDIS_URL: str = _opt("REDIS_URL", "redis://localhost:6379/0")

MAX_WORKERS: int = _int("MAX_WORKERS", 2)
"""Number of concurrent Celery workers."""

# ---------------------------------------------------------------------------
# NAS Reliability
# ---------------------------------------------------------------------------
MAX_NAS_RETRY: int = _int("MAX_NAS_RETRY", 3)
"""Max retries on NAS read failure before failing job."""

NAS_RETRY_DELAY_SEC: float = _float("NAS_RETRY_DELAY_SEC", 5.0)
"""Delay between NAS retry attempts."""

NAS_READ_BUFFER_BYTES: int = _int("NAS_READ_BUFFER_BYTES", 8 * 1024 * 1024)
"""Read buffer size for NAS I/O (default 8MB)."""

# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------
WATCH_POLL_INTERVAL_SEC: float = _float("WATCH_POLL_INTERVAL_SEC", 10.0)
"""How often to poll NAS watch directory for new files."""

WATCH_STABLE_TIME_SEC: float = _float("WATCH_STABLE_TIME_SEC", 30.0)
"""Wait this long after last file modification before considering it fully written."""

# ---------------------------------------------------------------------------
# macOS Sleep Prevention
# ---------------------------------------------------------------------------
PREVENT_SLEEP: bool = _bool("PREVENT_SLEEP", True)
"""Use caffeinate to prevent idle sleep during jobs (macOS only)."""

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
def working_dir_for_job(job_id: str) -> Path:
    return Path(WORKING_DIR) / job_id

def output_dir_for_job(job_id: str) -> Path:
    return Path(NAS_OUTPUT_PATH) / job_id


class _Config:
    """
    Dynamic config accessor: reads env vars at call time.
    Use this in request handlers so tests can override env vars per-test.
    Usage: from src.config import config; config.NAS_MOUNT_PATH
    """
    def __getattr__(self, name: str):
        import os
        # Map attribute names to env vars
        env_map = {
            "NAS_MOUNT_PATH": ("NAS_MOUNT_PATH", None),
            "NAS_OUTPUT_PATH": ("NAS_OUTPUT_PATH", None),
            "WORKING_DIR": ("WORKING_DIR", "/tmp/soccer-pipeline"),
            "CHUNK_DURATION_SEC": ("CHUNK_DURATION_SEC", "30"),
            "CHUNK_OVERLAP_SEC": ("CHUNK_OVERLAP_SEC", "2.0"),
            "PRE_EVENT_PAD_SEC": ("PRE_EVENT_PAD_SEC", "3.0"),
            "POST_EVENT_PAD_SEC": ("POST_EVENT_PAD_SEC", "5.0"),
            "MIN_EVENT_CONFIDENCE": ("MIN_EVENT_CONFIDENCE", "0.65"),
            "YOLO_MODEL_PATH": ("YOLO_MODEL_PATH", "/models/yolov8m.pt"),
            "ACTION_MODEL_PATH": ("ACTION_MODEL_PATH", "/models/videomae-soccer.pt"),
            "USE_GPU": ("USE_GPU", "false"),
            "USE_NULL_DETECTOR": ("USE_NULL_DETECTOR", "false"),
            "YOLO_INFERENCE_SIZE": ("YOLO_INFERENCE_SIZE", "1280"),
            "DETECTION_FRAME_STEP": ("DETECTION_FRAME_STEP", "3"),
            "OUTPUT_CODEC": ("OUTPUT_CODEC", "copy"),
            "OUTPUT_CRF": ("OUTPUT_CRF", "18"),
            "OUTPUT_AUDIO_CODEC": ("OUTPUT_AUDIO_CODEC", "copy"),
            "CELERY_BROKER_URL": ("CELERY_BROKER_URL", "redis://localhost:6379/0"),
            "CELERY_RESULT_BACKEND": ("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
            "REDIS_URL": ("REDIS_URL", "redis://localhost:6379/0"),
            "MAX_WORKERS": ("MAX_WORKERS", "2"),
            "MAX_NAS_RETRY": ("MAX_NAS_RETRY", "3"),
            "NAS_RETRY_DELAY_SEC": ("NAS_RETRY_DELAY_SEC", "5.0"),
            "NAS_READ_BUFFER_BYTES": ("NAS_READ_BUFFER_BYTES", str(8*1024*1024)),
            "WATCH_POLL_INTERVAL_SEC": ("WATCH_POLL_INTERVAL_SEC", "10.0"),
            "WATCH_STABLE_TIME_SEC": ("WATCH_STABLE_TIME_SEC", "30.0"),
            "PREVENT_SLEEP": ("PREVENT_SLEEP", "true"),
        }
        if name not in env_map:
            raise AttributeError(f"Unknown config key: {name}")
        env_key, default = env_map[name]
        value = os.getenv(env_key, default)
        if value is None:
            raise EnvironmentError(f"Required env var {env_key} not set")
        return value

config = _Config()
