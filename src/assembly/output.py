"""
Output manager: writes final reels from working dir to NAS output path.

Ensures:
  - Atomic writes (temp file → rename)
  - Directory creation
  - Output path convention: {NAS_OUTPUT_PATH}/{job_id}/{reel_type}_reel.mp4
  - Post-write validation via ffprobe
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import structlog

from src.assembly.encoder import validate_clip

log = structlog.get_logger(__name__)

REEL_FILENAME_MAP = {
    "goalkeeper": "goalkeeper_reel.mp4",
    "keeper_a":   "keeper_a_reel.mp4",
    "keeper_b":   "keeper_b_reel.mp4",
    "highlights": "highlights_reel.mp4",
    "player":     "player_reel.mp4",
}


def get_output_path(nas_output_base: str, job_id: str, reel_type: str) -> Path:
    """Return canonical output path for a reel."""
    filename = REEL_FILENAME_MAP.get(reel_type, f"{reel_type}_reel.mp4")
    return Path(nas_output_base) / job_id / filename


def write_reel_to_nas(
    working_reel_path: str | Path,
    nas_output_base: str,
    job_id: str,
    reel_type: str,
    max_retries: int = 3,
) -> str:
    """
    Copy assembled reel from working dir to NAS output, atomically.

    Uses copy + atomic rename to prevent partial writes being visible.
    Validates the output file after write.

    Returns:
        Final NAS output path on success.

    Raises:
        RuntimeError if write or validation fails after retries.
    """
    source = Path(working_reel_path)
    if not source.exists():
        raise FileNotFoundError(f"Working reel not found: {source}")

    dest = get_output_path(nas_output_base, job_id, reel_type)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest_tmp = dest.with_suffix(".tmp.mp4")

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            shutil.copy2(str(source), str(dest_tmp))
            os.replace(str(dest_tmp), str(dest))

            # Validate written file
            if not validate_clip(str(dest)):
                raise RuntimeError(f"Output validation failed: {dest}")

            size_mb = dest.stat().st_size / 1024 / 1024
            log.info(
                "output.reel_written",
                reel_type=reel_type,
                job_id=job_id,
                path=str(dest),
                size_mb=round(size_mb, 1),
            )
            return str(dest)

        except Exception as exc:
            last_exc = exc
            dest_tmp.unlink(missing_ok=True)
            log.warning(
                "output.write_retry",
                attempt=attempt + 1,
                error=str(exc),
                dest=str(dest),
            )

    raise RuntimeError(
        f"Failed to write reel to NAS after {max_retries} attempts: {last_exc}"
    )


def write_job_manifest(
    nas_output_base: str,
    job_id: str,
    output_paths: dict[str, str],
    metadata: dict,
) -> str:
    """
    Write a JSON manifest alongside the reels for downstream tooling.
    Returns path to manifest file.
    """
    import json
    from datetime import datetime, timezone

    manifest = {
        "job_id": job_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "reels": output_paths,
        **metadata,
    }
    manifest_path = Path(nas_output_base) / job_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("output.manifest_written", path=str(manifest_path))
    return str(manifest_path)
