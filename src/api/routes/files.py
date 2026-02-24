"""
List video files available on the NAS mount.

GET /files   Return filenames suitable for the submit-job dropdown.
"""
from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException

log = structlog.get_logger(__name__)
router = APIRouter()

_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".mts", ".m2ts", ".ts"})


@router.get("")
@router.get("/", include_in_schema=False)
def list_files():
    """List video files in the NAS mount directory (non-recursive, top-level only)."""
    from src.config import config as dyn_cfg

    nas_path = Path(dyn_cfg.NAS_MOUNT_PATH)
    if not nas_path.exists():
        raise HTTPException(503, f"NAS mount not available: {nas_path}")

    files = sorted(
        p.name
        for p in nas_path.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
    )
    return files
