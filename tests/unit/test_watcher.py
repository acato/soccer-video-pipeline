"""Unit tests for src/ingestion/watcher.py"""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.watcher import NASWatcher


@pytest.mark.unit
class TestNASWatcher:
    def test_detects_stable_file(self, tmp_path):
        """A file that doesn't change size for stable_time is submitted."""
        submitted = []
        watcher = NASWatcher(
            watch_path=str(tmp_path),
            on_new_file=lambda p: submitted.append(p),
            poll_interval_sec=0.01,
            stable_time_sec=0.05,
        )

        video = tmp_path / "match.mp4"
        video.write_bytes(b"fake mp4 data" * 1000)

        # First poll: file detected, enters pending
        watcher._poll()
        assert len(submitted) == 0
        assert str(video) in watcher._pending

        # Wait for stability window
        time.sleep(0.1)

        # Second poll: file is stable → submitted
        watcher._poll()
        assert len(submitted) == 1
        assert submitted[0] == str(video)

    def test_growing_file_not_submitted(self, tmp_path):
        """A file that keeps growing is never submitted."""
        submitted = []
        watcher = NASWatcher(
            watch_path=str(tmp_path),
            on_new_file=lambda p: submitted.append(p),
            stable_time_sec=0.05,
        )

        video = tmp_path / "growing.mp4"
        video.write_bytes(b"a" * 1000)
        watcher._poll()

        # Simulate growth between polls
        video.write_bytes(b"a" * 2000)
        time.sleep(0.1)
        watcher._poll()

        assert len(submitted) == 0  # Growing file not yet submitted

    def test_already_seen_file_skipped(self, tmp_path):
        """File submitted once is never re-submitted."""
        count = []
        watcher = NASWatcher(
            watch_path=str(tmp_path),
            on_new_file=lambda p: count.append(p),
            stable_time_sec=0.01,
        )

        video = tmp_path / "seen.mp4"
        video.write_bytes(b"data")

        watcher._poll()
        time.sleep(0.05)
        watcher._poll()  # Submits
        watcher._poll()  # Should skip (already seen)

        assert len(count) == 1

    def test_missing_watch_path_is_graceful(self, tmp_path):
        """Missing watch directory logs warning but doesn't crash."""
        watcher = NASWatcher(
            watch_path=str(tmp_path / "nonexistent"),
            on_new_file=MagicMock(),
            stable_time_sec=0.01,
        )
        # Should not raise
        watcher._poll()

    def test_on_new_file_error_allows_retry(self, tmp_path):
        """If on_new_file raises, the file is removed from _seen so it can be retried."""
        call_count = [0]

        def failing_handler(path):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("NAS write error")

        watcher = NASWatcher(
            watch_path=str(tmp_path),
            on_new_file=failing_handler,
            stable_time_sec=0.01,
        )

        video = tmp_path / "retry.mp4"
        video.write_bytes(b"data")

        # Directly inject into _pending as already-stable to skip timing
        watcher._pending[str(video)] = (video.stat().st_size, 0.0)  # first_seen in the past
        watcher._poll()  # First attempt: raises, file removed from _seen
        assert call_count[0] == 1
        assert str(video) not in watcher._seen  # Failed → not in seen → can retry

        # Re-inject for retry
        watcher._pending[str(video)] = (video.stat().st_size, 0.0)
        watcher._poll()  # Second attempt: succeeds
        assert call_count[0] == 2
        assert str(video) in watcher._seen
