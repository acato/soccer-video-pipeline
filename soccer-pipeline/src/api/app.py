"""
FastAPI application factory, health endpoints, and middleware.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import jobs, reels
from src.api.routes.events import router as events_router
from src.api.routes.ui import router as ui_router
from src.api import metrics

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.config import config as cfg
    jobs_dir = Path(cfg.WORKING_DIR) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    log.info("api.startup", working_dir=cfg.WORKING_DIR)
    yield
    log.info("api.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Soccer Video Pipeline API",
        description="Submit match videos for goalkeeper and highlights reel generation",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(jobs.router,    prefix="/jobs",    tags=["jobs"])
    app.include_router(reels.router,   prefix="/reels",   tags=["reels"])
    app.include_router(events_router,  prefix="/events",  tags=["events"])
    app.include_router(metrics.router,                    tags=["ops"])
    app.include_router(ui_router,                         tags=["ui"])

    @app.get("/health", tags=["ops"])
    async def health():
        """Liveness probe — returns 200 if API is running."""
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/ready", tags=["ops"])
    async def ready():
        """Readiness probe — checks NAS mount and job store."""
        from src.config import config as cfg
        issues = []
        if not Path(cfg.NAS_MOUNT_PATH).exists():
            issues.append(f"NAS_MOUNT_PATH not mounted: {cfg.NAS_MOUNT_PATH}")
        if issues:
            from fastapi.responses import JSONResponse
            return JSONResponse({"status": "not_ready", "issues": issues}, status_code=503)
        return {"status": "ready"}

    return app


app = create_app()
