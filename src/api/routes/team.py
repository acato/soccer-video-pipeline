"""
Team configuration endpoint.

GET /team — returns the saved team config (name + all kits)
from ~/.soccer-pipeline-team.json (or TEAM_CONFIG_PATH env var).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse

log = structlog.get_logger(__name__)
router = APIRouter()

TEAM_CONFIG_DEFAULT = Path.home() / ".soccer-pipeline-team.json"


def _team_config_path() -> Path:
    return Path(os.getenv("TEAM_CONFIG_PATH", str(TEAM_CONFIG_DEFAULT)))


def load_team_config() -> dict | None:
    """Load team config from disk.  Returns None if not found."""
    path = _team_config_path()
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("team.config_read_error", path=str(path), error=str(exc))
        return None


@router.get("")
@router.get("/", include_in_schema=False)
def get_team():
    """Return the saved team config, or 404 if none exists."""
    config = load_team_config()
    if config is None:
        return JSONResponse(
            {"error": "no_team_config", "message": "No team configured yet. Run ./setup-team.sh to set up your team."},
            status_code=404,
        )
    return config
