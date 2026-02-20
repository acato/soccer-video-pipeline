"""
Video intake: validate files, extract metadata via ffprobe, compute SHA-256.

All operations are NAS-safe:
  - Buffered reads (no mmap)
  - Retries with exponential backoff on I/O errors
  - Never modifies source files
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import structlog

from src.ingestion.models import VideoFile

log = structlog.get_logger(__name__)

SUPPORTED_CODECS = {"h264", "hevc"}
SUPPORTED_EXTENSIONS = {".mp4", ".MP4"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_video_path(path: str) -> str:
    """
    Validate that the given path is an accessible MP4 file.
    Raises FileNotFoundError or ValueError on failure.
    Returns the path unchanged on success.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if p.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{p.suffix}' — expected one of {SUPPORTED_EXTENSIONS}"
        )
    return str(path)


def extract_metadata(path: str, max_retries: int = 3) -> VideoFile:
    """
    Run ffprobe on the given path and return a VideoFile model.
    Retries on transient NAS errors with exponential backoff.
    """
    validate_video_path(path)

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = _run_ffprobe(path)
            vf = parse_ffprobe_output(raw, path)
            vf_with_hash = VideoFile(
                **vf.model_dump(exclude={"sha256"}),
                sha256=compute_sha256(path),
            )
            log.info(
                "intake.metadata_extracted",
                path=path,
                duration_sec=vf_with_hash.duration_sec,
                resolution=f"{vf_with_hash.width}x{vf_with_hash.height}",
                fps=vf_with_hash.fps,
                codec=vf_with_hash.codec,
            )
            return vf_with_hash
        except (OSError, IOError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            log.warning("intake.retry", attempt=attempt + 1, wait_sec=wait, error=str(exc))
            time.sleep(wait)

    raise RuntimeError(f"Failed to extract metadata after {max_retries} attempts: {last_exc}")


def parse_ffprobe_output(data: dict[str, Any], path: str) -> VideoFile:
    """
    Parse ffprobe JSON output dict into a VideoFile model.
    Pure function — no I/O. Directly unit-testable.
    """
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError(f"No video stream found in: {path}")

    vs = video_streams[0]
    codec_name = vs.get("codec_name", "").lower()
    if codec_name not in SUPPORTED_CODECS:
        raise ValueError(
            f"Unsupported codec '{codec_name}' in {path} — supported: {SUPPORTED_CODECS}"
        )

    fps = _parse_fps(vs.get("r_frame_rate", "30/1"))
    duration = float(fmt.get("duration", vs.get("duration", 0)))
    size = int(fmt.get("size", 0))

    return VideoFile(
        path=path,
        filename=Path(path).name,
        duration_sec=duration,
        fps=fps,
        width=int(vs.get("width", 0)),
        height=int(vs.get("height", 0)),
        codec=codec_name,
        size_bytes=size,
        sha256="",  # Populated by extract_metadata after this call
    )


def compute_sha256(path: str, buffer_size: int = 8 * 1024 * 1024) -> str:
    """
    Compute SHA-256 of file contents using buffered reads.
    Buffer size defaults to 8MB to avoid RAM exhaustion on large NAS files.
    """
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while chunk := f.read(buffer_size):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_ffprobe(path: str) -> dict[str, Any]:
    """Execute ffprobe and return parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    return json.loads(result.stdout)


def _parse_fps(fps_str: str) -> float:
    """Parse '30/1', '60000/1001', '30' style fps strings."""
    if "/" in fps_str:
        num, den = fps_str.split("/", 1)
        den_val = float(den)
        if den_val == 0:
            return 0.0
        return float(num) / den_val
    return float(fps_str)
