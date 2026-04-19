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
# VLM (Vision Language Model) Classification
# ---------------------------------------------------------------------------
VLM_ENABLED: bool = _bool("VLM_ENABLED", False)
"""Use Claude VLM to verify GK save events post-detection."""

ANTHROPIC_API_KEY: str = _opt("ANTHROPIC_API_KEY", "")
"""API key for Anthropic Claude. Required when VLM_ENABLED=true."""

VLM_MODEL: str = _opt("VLM_MODEL", "claude-sonnet-4-20250514")
"""Claude model to use for VLM classification."""

VLM_FRAME_WIDTH: int = _int("VLM_FRAME_WIDTH", 640)
"""Resize width for extracted VLM frames (smaller = cheaper API calls)."""

VLM_MIN_CONFIDENCE: float = _float("VLM_MIN_CONFIDENCE", 0.6)
"""Minimum VLM confidence to keep an event."""

USE_VLM_DETECTION: bool = _bool("USE_VLM_DETECTION", False)
"""Use VLM-first two-pass detection instead of YOLO heuristic pipeline."""

VLM_FRAME_INTERVAL: float = _float("VLM_FRAME_INTERVAL", 3.0)
"""Seconds between frames in VLM coarse scan."""

VLM_DETECT_FRAME_WIDTH: int = _int("VLM_DETECT_FRAME_WIDTH", 960)
"""Resize width for VLM detection frames (larger than verification frames)."""

# ---------------------------------------------------------------------------
# vLLM (local inference server)
# ---------------------------------------------------------------------------
VLLM_ENABLED: bool = _bool("VLLM_ENABLED", False)
"""Use local vLLM server for VLM classification instead of Anthropic API."""

VLLM_URL: str = _opt("VLLM_URL", "http://10.10.2.222:8000")
"""Base URL of the vLLM OpenAI-compatible server."""

VLLM_MODEL: str = _opt("VLLM_MODEL", "qwen3-vl-32b")
"""Model name served by vLLM (must match --served-model-name)."""

VLLM_MIN_CONFIDENCE: float = _float("VLLM_MIN_CONFIDENCE", 0.5)
"""Minimum confidence for vLLM classification results."""

# ---------------------------------------------------------------------------
# Two-Tier VLM Classification
# ---------------------------------------------------------------------------
TIERED_VLM_ENABLED: bool = _bool("TIERED_VLM_ENABLED", False)
"""Enable two-tier VLM: Tier 1 (8B fast) + Tier 2 (32B accurate)."""

TIER1_MODEL: str = _opt("TIER1_MODEL", "soccer-8b-lora")
"""Model name for Tier 1 (fast classification). Must match vLLM --served-model-name."""

TIER1_MODEL_PATH: str = _opt("TIER1_MODEL_PATH", "")
"""HuggingFace or local path for Tier 1 model (used by swap script)."""

TIER1_LORA_PATH: str = _opt("TIER1_LORA_PATH", "")
"""LoRA adapter path for Tier 1 (if applicable)."""

TIER2_MODEL: str = _opt("TIER2_MODEL", "qwen3-vl-32b-fp8")
"""Model name for Tier 2 (accurate review). Must match vLLM --served-model-name."""

TIER2_MODEL_PATH: str = _opt("TIER2_MODEL_PATH", "Qwen/Qwen3-VL-32B-Instruct-FP8")
"""HuggingFace or local path for Tier 2 model (used by swap script)."""

TIER1_MIN_CONFIDENCE: float = _float("TIER1_MIN_CONFIDENCE", 0.6)
"""Below this confidence, Tier 1 verdicts escalate to Tier 2."""

TIER1_BROKEN_THRESHOLD: float = _float("TIER1_BROKEN_THRESHOLD", 3.0)
"""If any label exceeds expected_rate * this multiplier, Tier 1 is broken."""

TIER2_SPOT_CHECK_RATE: float = _float("TIER2_SPOT_CHECK_RATE", 0.10)
"""Fraction of confident Tier 1 verdicts to spot-check with Tier 2."""

TIER2_ESCALATION_CAP: float = _float("TIER2_ESCALATION_CAP", 0.50)
"""Max fraction of verdicts to escalate. Exceeding this = Tier 1 broken."""

MODEL_SWAP_SCRIPT: str = _opt("MODEL_SWAP_SCRIPT", "")
"""Path to shell script for swapping vLLM models between tiers."""

MODEL_SWAP_TIMEOUT_SEC: int = _int("MODEL_SWAP_TIMEOUT_SEC", 120)
"""Max seconds to wait for model swap to complete."""

# ---------------------------------------------------------------------------
# Dual-Pass VLM Detection (8B triage + 32B classify)
# ---------------------------------------------------------------------------
DUAL_PASS_ENABLED: bool = _bool("DUAL_PASS_ENABLED", False)
"""Enable dual-pass VLM: 8B triage scan → model swap → 32B classify."""

DUAL_PASS_TIER1_NAME: str = _opt("DUAL_PASS_TIER1_NAME", "qwen3-vl-8b")
"""8B model --served-model-name for triage pass."""

DUAL_PASS_TIER1_PATH: str = _opt("DUAL_PASS_TIER1_PATH", "Qwen/Qwen3-VL-8B-Instruct")
"""8B model HuggingFace path for triage pass."""

DUAL_PASS_TIER2_NAME: str = _opt("DUAL_PASS_TIER2_NAME", "qwen3-vl-32b-fp8")
"""32B model --served-model-name for classification pass."""

DUAL_PASS_TIER2_PATH: str = _opt("DUAL_PASS_TIER2_PATH", "Qwen/Qwen3-VL-32B-Instruct-FP8")
"""32B model HuggingFace path for classification pass."""

DUAL_PASS_TRIAGE_STEP: float = _float("DUAL_PASS_TRIAGE_STEP", 6.0)
"""Seconds between sliding window steps in 8B triage scan."""

DUAL_PASS_SWAP_SCRIPT: str = _opt("DUAL_PASS_SWAP_SCRIPT", "")
"""Path to model swap script (defaults to scripts/swap_vllm_model.sh)."""

SINGLE_PASS_32B: bool = _bool("SINGLE_PASS_32B", False)
"""Single-pass mode: 32B classifies directly from frames (no triage/observe split)."""

SINGLE_PASS_STEP_SEC: float = _float("SINGLE_PASS_STEP_SEC", 10.0)
"""Seconds between sliding window steps in single-pass mode."""

SINGLE_PASS_WINDOW_SEC: float = _float("SINGLE_PASS_WINDOW_SEC", 15.0)
"""Window duration in seconds for single-pass classification."""

SINGLE_PASS_FRAMES: int = _int("SINGLE_PASS_FRAMES", 5)
"""Number of frames per window in single-pass mode."""

YOLO_GROUNDING_ENABLED: bool = _bool("YOLO_GROUNDING_ENABLED", False)
"""Run #33 breakthrough: YOLO spatial-grounding gate on VLM events.
Rejects throw_in/corner_kick/goal_kick detections where the ball isn't
actually near the required field landmark."""

YOLO_GROUNDING_FAIL_OPEN: bool = _bool("YOLO_GROUNDING_FAIL_OPEN", True)
"""Keep events when YOLO can't find a ball (preserves recall)."""

YOLO_GROUNDING_FRAMES: int = _int("YOLO_GROUNDING_FRAMES", 3)
"""Frames to sample per candidate for grounding."""

YOLO_GROUNDING_FRAME_SPAN_SEC: float = _float("YOLO_GROUNDING_FRAME_SPAN_SEC", 2.0)
"""Total span of sampled frames around the event timestamp (seconds)."""

YOLO_GROUNDING_INFERENCE_SIZE: int = _int("YOLO_GROUNDING_INFERENCE_SIZE", 640)
"""YOLO input resolution for grounding inference (smaller = faster)."""

YOLO_GROUNDING_BALL_CONF: float = _float("YOLO_GROUNDING_BALL_CONF", 0.15)
"""Minimum YOLO confidence to accept a ball detection."""

YOLO_BALL_CLASS_ID: int = _int("YOLO_BALL_CLASS_ID", 32)
"""Class index of 'ball' in the YOLO model. COCO default is 32 (sports_ball);
soccer-tuned models (e.g. uisikdag/yolo-v8-football-players-detection) use 0."""

YOLO_PERSON_CLASS_IDS: str = _opt("YOLO_PERSON_CLASS_IDS", "0")
"""Comma-separated class indices counted as persons. COCO default: '0'.
Soccer-tuned models: '1,2,3' (goalkeeper, player, referee)."""

YOLO_GK_CLASS_IDS: str = _opt("YOLO_GK_CLASS_IDS", "")
"""Comma-separated class indices for goalkeeper-specific tracking. Empty by
default. Soccer-tuned models set '1'. Tracked for the Run #36 GK-action gate."""

YOLO_GK_PROXIMITY_THRESHOLD: float = _float("YOLO_GK_PROXIMITY_THRESHOLD", 0.20)
"""Max normalized Euclidean ball-to-GK distance for a GK-action event
(catch, shot_stop_*, punch) to be accepted by the grounding gate. 0.20 ≈
20% of the frame diagonal — generous first pass."""

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
            "VLLM_ENABLED": ("VLLM_ENABLED", "false"),
            "VLLM_URL": ("VLLM_URL", "http://10.10.2.222:8000"),
            "VLLM_MODEL": ("VLLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct-FP8"),
            "VLLM_MIN_CONFIDENCE": ("VLLM_MIN_CONFIDENCE", "0.5"),
            "VLM_ENABLED": ("VLM_ENABLED", "false"),
            "ANTHROPIC_API_KEY": ("ANTHROPIC_API_KEY", ""),
            "VLM_MODEL": ("VLM_MODEL", "claude-sonnet-4-20250514"),
            "VLM_FRAME_WIDTH": ("VLM_FRAME_WIDTH", "640"),
            "VLM_MIN_CONFIDENCE": ("VLM_MIN_CONFIDENCE", "0.6"),
            "USE_VLM_DETECTION": ("USE_VLM_DETECTION", "false"),
            "VLM_FRAME_INTERVAL": ("VLM_FRAME_INTERVAL", "3.0"),
            "VLM_DETECT_FRAME_WIDTH": ("VLM_DETECT_FRAME_WIDTH", "960"),
            # Two-tier VLM
            "TIERED_VLM_ENABLED": ("TIERED_VLM_ENABLED", "false"),
            "TIER1_MODEL": ("TIER1_MODEL", "soccer-8b-lora"),
            "TIER1_MODEL_PATH": ("TIER1_MODEL_PATH", ""),
            "TIER1_LORA_PATH": ("TIER1_LORA_PATH", ""),
            "TIER2_MODEL": ("TIER2_MODEL", "qwen3-vl-32b-fp8"),
            "TIER2_MODEL_PATH": ("TIER2_MODEL_PATH", "Qwen/Qwen3-VL-32B-Instruct-FP8"),
            "TIER1_MIN_CONFIDENCE": ("TIER1_MIN_CONFIDENCE", "0.6"),
            "TIER1_BROKEN_THRESHOLD": ("TIER1_BROKEN_THRESHOLD", "3.0"),
            "TIER2_SPOT_CHECK_RATE": ("TIER2_SPOT_CHECK_RATE", "0.10"),
            "TIER2_ESCALATION_CAP": ("TIER2_ESCALATION_CAP", "0.50"),
            "MODEL_SWAP_SCRIPT": ("MODEL_SWAP_SCRIPT", ""),
            "MODEL_SWAP_TIMEOUT_SEC": ("MODEL_SWAP_TIMEOUT_SEC", "120"),
            # Dual-pass VLM
            "DUAL_PASS_ENABLED": ("DUAL_PASS_ENABLED", "false"),
            "DUAL_PASS_TIER1_NAME": ("DUAL_PASS_TIER1_NAME", "qwen3-vl-8b"),
            "DUAL_PASS_TIER1_PATH": ("DUAL_PASS_TIER1_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
            "DUAL_PASS_TIER2_NAME": ("DUAL_PASS_TIER2_NAME", "qwen3-vl-32b-fp8"),
            "DUAL_PASS_TIER2_PATH": ("DUAL_PASS_TIER2_PATH", "Qwen/Qwen3-VL-32B-Instruct-FP8"),
            "DUAL_PASS_TRIAGE_STEP": ("DUAL_PASS_TRIAGE_STEP", "6.0"),
            "DUAL_PASS_SWAP_SCRIPT": ("DUAL_PASS_SWAP_SCRIPT", ""),
            # Single-pass 32B
            "SINGLE_PASS_32B": ("SINGLE_PASS_32B", "false"),
            "SINGLE_PASS_STEP_SEC": ("SINGLE_PASS_STEP_SEC", "10.0"),
            "SINGLE_PASS_WINDOW_SEC": ("SINGLE_PASS_WINDOW_SEC", "15.0"),
            "SINGLE_PASS_FRAMES": ("SINGLE_PASS_FRAMES", "5"),
            # YOLO spatial grounding (Run #33 breakthrough)
            "YOLO_GROUNDING_ENABLED": ("YOLO_GROUNDING_ENABLED", "false"),
            "YOLO_GROUNDING_FAIL_OPEN": ("YOLO_GROUNDING_FAIL_OPEN", "true"),
            "YOLO_GROUNDING_FRAMES": ("YOLO_GROUNDING_FRAMES", "3"),
            "YOLO_GROUNDING_FRAME_SPAN_SEC": ("YOLO_GROUNDING_FRAME_SPAN_SEC", "2.0"),
            "YOLO_GROUNDING_INFERENCE_SIZE": ("YOLO_GROUNDING_INFERENCE_SIZE", "640"),
            "YOLO_GROUNDING_BALL_CONF": ("YOLO_GROUNDING_BALL_CONF", "0.15"),
            "YOLO_BALL_CLASS_ID": ("YOLO_BALL_CLASS_ID", "32"),
            "YOLO_PERSON_CLASS_IDS": ("YOLO_PERSON_CLASS_IDS", "0"),
            "YOLO_GK_CLASS_IDS": ("YOLO_GK_CLASS_IDS", ""),
            "YOLO_GK_PROXIMITY_THRESHOLD": ("YOLO_GK_PROXIMITY_THRESHOLD", "0.20"),
        }
        if name not in env_map:
            raise AttributeError(f"Unknown config key: {name}")
        env_key, default = env_map[name]
        value = os.getenv(env_key, default)
        if value is None:
            raise EnvironmentError(f"Required env var {env_key} not set")
        return value

config = _Config()
