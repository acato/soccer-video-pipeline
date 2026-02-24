"""
Structured event log (JSONL format).

One file per job at: WORKING_DIR/{job_id}/events.jsonl
- Append-only writes (idempotent: duplicate event_ids are deduplicated on read)
- Atomic line appends using line buffering
- Supports streaming read for large matches
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import structlog

from src.detection.models import Event

log = structlog.get_logger(__name__)


class EventLog:
    """
    Append-only JSONL event log for a single job.

    Thread safety: Multiple writers should use separate EventLog instances
    and merge via merge_logs(). Single-writer pattern is assumed per job.
    """

    def __init__(self, log_path: str | Path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event) -> None:
        """Append one event to the log. Creates file if it doesn't exist."""
        with open(self.path, "a", buffering=1) as f:  # Line-buffered
            f.write(event.model_dump_json() + "\n")
        log.debug("event_log.append", event_id=event.event_id, event_type=event.event_type)

    def append_many(self, events: list[Event]) -> None:
        """Bulk append — more efficient than calling append() in a loop."""
        if not events:
            return
        with open(self.path, "a", buffering=1) as f:
            for event in events:
                f.write(event.model_dump_json() + "\n")
        log.info("event_log.bulk_append", count=len(events), path=str(self.path))

    def iter_events(self) -> Iterator[Event]:
        """Stream events from disk without loading all into memory."""
        if not self.path.exists():
            return
        with open(self.path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Event.model_validate_json(line)
                except Exception as exc:
                    log.warning(
                        "event_log.parse_error",
                        line=line_num,
                        error=str(exc),
                        path=str(self.path),
                    )

    def read_all(self) -> list[Event]:
        """Load all events, deduplicating by event_id (last write wins)."""
        seen: dict[str, Event] = {}
        for event in self.iter_events():
            seen[event.event_id] = event
        events = list(seen.values())
        events.sort(key=lambda e: e.timestamp_start)
        return events

    def filter_by_reel(self, reel_type: str) -> list[Event]:
        """Return all events for a specific reel type, sorted by timestamp.

        GK events use sub-roles like keeper_a/keeper_b, so "keeper" matches
        any target starting with "keeper".
        """
        return [
            e for e in self.read_all()
            if any(rt == reel_type or rt.startswith(reel_type + "_") for rt in e.reel_targets)
        ]

    def filter_by_confidence(self, min_confidence: float = 0.65) -> list[Event]:
        """Return only events meeting confidence threshold (respects manual overrides)."""
        return [e for e in self.read_all() if e.should_include(min_confidence)]

    def count(self) -> int:
        """Count events without loading all into memory."""
        if not self.path.exists():
            return 0
        count = 0
        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0


def merge_logs(source_paths: list[Path], dest_path: Path) -> int:
    """
    Merge multiple JSONL event logs into one, deduplicating by event_id.
    Returns number of events written.
    """
    seen: dict[str, Event] = {}
    for src in source_paths:
        log_reader = EventLog(src)
        for event in log_reader.iter_events():
            seen[event.event_id] = event

    events = sorted(seen.values(), key=lambda e: e.timestamp_start)
    dest_log = EventLog(dest_path)
    # Write fresh (overwrite if exists)
    dest_path.write_text("")
    dest_log.append_many(events)
    log.info("event_log.merged", sources=len(source_paths), events_written=len(events))
    return len(events)
