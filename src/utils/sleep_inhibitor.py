"""
Prevent macOS idle sleep while a pipeline job is running.

Uses ``caffeinate -i`` (inhibit idle sleep) so the NAS volume stays mounted
for the full duration of a job.  No-op on non-macOS platforms or when the
``PREVENT_SLEEP`` config flag is disabled.
"""
from __future__ import annotations

import subprocess
import sys
from types import TracebackType

import structlog

log = structlog.get_logger(__name__)


class SleepInhibitor:
    """Context manager that keeps macOS awake via ``caffeinate``."""

    def __init__(self, *, job_id: str = "", enabled: bool = True) -> None:
        self._job_id = job_id
        self._enabled = enabled and sys.platform == "darwin"
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "SleepInhibitor":
        if not self._enabled:
            return self
        try:
            self._proc = subprocess.Popen(
                ["caffeinate", "-i"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info(
                "sleep_inhibitor.engaged",
                job_id=self._job_id,
                pid=self._proc.pid,
            )
        except FileNotFoundError:
            log.warning("sleep_inhibitor.caffeinate_not_found", job_id=self._job_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
            log.info(
                "sleep_inhibitor.released",
                job_id=self._job_id,
                pid=self._proc.pid,
            )
        except ProcessLookupError:
            pass  # already exited
        except Exception as exc:
            log.warning(
                "sleep_inhibitor.cleanup_error",
                job_id=self._job_id,
                error=str(exc),
            )
        finally:
            self._proc = None
