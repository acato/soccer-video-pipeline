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
# Audio Detection (Phase 1)
# ---------------------------------------------------------------------------
AUDIO_ENABLED: bool = _bool("AUDIO_ENABLED", True)
"""Enable audio-based whistle/energy detection. Fails open if no audio stream."""

AUDIO_BANDPASS_LOW_HZ: int = _int("AUDIO_BANDPASS_LOW_HZ", 2000)
"""Low end of whistle bandpass filter (Hz)."""

AUDIO_BANDPASS_HIGH_HZ: int = _int("AUDIO_BANDPASS_HIGH_HZ", 4000)
"""High end of whistle bandpass filter (Hz)."""

AUDIO_MIN_WHISTLE_SEC: float = _float("AUDIO_MIN_WHISTLE_SEC", 0.2)
"""Minimum whistle duration to count as a real whistle (seconds)."""

AUDIO_SURGE_STDDEV: float = _float("AUDIO_SURGE_STDDEV", 3.5)
"""Energy surge threshold in standard deviations above rolling mean."""

# ---------------------------------------------------------------------------
# Visual Candidate Generation (Phase 2)
# ---------------------------------------------------------------------------
VISUAL_SCAN_INTERVAL_SEC: float = _float("VISUAL_SCAN_INTERVAL_SEC", 15.0)
"""Spot-check interval for full-scan fallback when no audio (seconds)."""

# ---------------------------------------------------------------------------
# VLM Verification (Phase 3)
# ---------------------------------------------------------------------------
VLM_ENABLED: bool = _bool("VLM_ENABLED", False)
"""Use Claude API as VLM verification backend."""

ANTHROPIC_API_KEY: str = _opt("ANTHROPIC_API_KEY", "")
"""API key for Anthropic Claude. Required when VLM_ENABLED=true."""

VLM_MODEL: str = _opt("VLM_MODEL", "claude-sonnet-4-20250514")
"""Claude model to use for VLM verification."""

VLM_FRAME_WIDTH: int = _int("VLM_FRAME_WIDTH", 1280)
"""Resize width for extracted VLM frames (smaller = cheaper API calls)."""

VLM_MIN_CONFIDENCE: float = _float("VLM_MIN_CONFIDENCE", 0.6)
"""Minimum VLM confidence to confirm an event."""

VLLM_ENABLED: bool = _bool("VLLM_ENABLED", False)
"""Use vLLM-hosted vision model (Qwen3-VL) as VLM verification backend."""

VLLM_URL: str = _opt("VLLM_URL", "http://10.10.2.222:8000")
"""vLLM server URL (OpenAI-compatible API)."""

VLLM_MODEL: str = _opt("VLLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct-FP8")
"""Model name as registered in vLLM."""

VLLM_MIN_CONFIDENCE: float = _float("VLLM_MIN_CONFIDENCE", 0.5)
"""Minimum confidence to keep a classified event."""

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
            "USE_BALL_TOUCH_DETECTOR": ("USE_BALL_TOUCH_DETECTOR", "false"),
            # Audio detection (Phase 1)
            "AUDIO_ENABLED": ("AUDIO_ENABLED", "true"),
            "AUDIO_BANDPASS_LOW_HZ": ("AUDIO_BANDPASS_LOW_HZ", "2000"),
            "AUDIO_BANDPASS_HIGH_HZ": ("AUDIO_BANDPASS_HIGH_HZ", "4000"),
            "AUDIO_MIN_WHISTLE_SEC": ("AUDIO_MIN_WHISTLE_SEC", "0.2"),
            "AUDIO_SURGE_STDDEV": ("AUDIO_SURGE_STDDEV", "3.5"),
            # Visual candidate generation (Phase 2)
            "VISUAL_SCAN_INTERVAL_SEC": ("VISUAL_SCAN_INTERVAL_SEC", "15.0"),
            # VLM verification (Phase 3)
            "VLM_ENABLED": ("VLM_ENABLED", "false"),
            "ANTHROPIC_API_KEY": ("ANTHROPIC_API_KEY", ""),
            "VLM_MODEL": ("VLM_MODEL", "claude-sonnet-4-20250514"),
            "VLM_FRAME_WIDTH": ("VLM_FRAME_WIDTH", "1280"),
            "VLM_MIN_CONFIDENCE": ("VLM_MIN_CONFIDENCE", "0.6"),
            "VLLM_ENABLED": ("VLLM_ENABLED", "false"),
            "VLLM_URL": ("VLLM_URL", "http://10.10.2.222:8000"),
            "VLLM_MODEL": ("VLLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct-FP8"),
            "VLLM_MIN_CONFIDENCE": ("VLLM_MIN_CONFIDENCE", "0.5"),
        }
        if name not in env_map:
            raise AttributeError(f"Unknown config key: {name}")
        env_key, default = env_map[name]
        value = os.getenv(env_key, default)
        if value is None:
            raise EnvironmentError(f"Required env var {env_key} not set")
        return value

config = _Config()
