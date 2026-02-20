"""
NAS directory watcher.

Polls the configured NAS_MOUNT_PATH for new MP4 files and submits them as jobs.
Uses polling (not inotify) because NAS mounts often don't support inotify events.

Stability check: a file is only ingested after its size hasn't changed for
WATCH_STABLE_TIME_SEC seconds, preventing partial-write ingestion.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import structlog

log = structlog.get_logger(__name__)


class NASWatcher:
    """
    Polls a directory and calls `on_new_file(path)` for each new stable MP4.

    Tracks seen files via an in-memory set (survives across poll cycles,
    resets on watcher restart — intentional: idempotent job creation handles duplicates).
    """

    def __init__(
        self,
        watch_path: str,
        on_new_file: Callable[[str], None],
        poll_interval_sec: float = 10.0,
        stable_time_sec: float = 30.0,
    ):
        self.watch_path = Path(watch_path)
        self.on_new_file = on_new_file
        self.poll_interval = poll_interval_sec
        self.stable_time = stable_time_sec
        self._seen: set[str] = set()
        self._pending: dict[str, tuple[int, float]] = {}  # path -> (size, first_seen_ts)

    def run_forever(self) -> None:
        """Block and poll indefinitely. Call in a daemon thread or process."""
        log.info("watcher.started", watch_path=str(self.watch_path))
        while True:
            try:
                self._poll()
            except Exception as exc:
                log.error("watcher.poll_error", error=str(exc))
            time.sleep(self.poll_interval)

    def _poll(self) -> None:
        if not self.watch_path.exists():
            log.warning("watcher.path_missing", path=str(self.watch_path))
            return

        now = time.monotonic()
        for entry in self.watch_path.rglob("*.mp4"):
            path_str = str(entry)
            if path_str in self._seen:
                continue

            try:
                stat = entry.stat()
                current_size = stat.st_size
            except OSError:
                continue  # File disappeared — ignore

            if path_str not in self._pending:
                self._pending[path_str] = (current_size, now)
                log.debug("watcher.detected", path=path_str, size_bytes=current_size)
                continue

            prev_size, first_seen = self._pending[path_str]

            if current_size != prev_size:
                # Still growing — reset timer
                self._pending[path_str] = (current_size, now)
                continue

            if now - first_seen >= self.stable_time:
                # Stable — submit
                log.info("watcher.stable_file", path=path_str, size_bytes=current_size)
                self._seen.add(path_str)
                del self._pending[path_str]
                try:
                    self.on_new_file(path_str)
                except Exception as exc:
                    log.error("watcher.submit_error", path=path_str, error=str(exc))
                    self._seen.discard(path_str)  # Allow retry on next poll
