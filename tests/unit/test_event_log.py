"""Unit tests for src/detection/event_log.py"""
import json
from pathlib import Path
import pytest

from src.detection.event_log import EventLog, merge_logs
from src.detection.models import Event, EventType


def _make_event(event_id: str, ts_start: float = 10.0, reel="goalkeeper") -> Event:
    return Event(
        event_id=event_id,
        job_id="job-001",
        source_file="match.mp4",
        event_type=EventType.SHOT_STOP_DIVING,
        timestamp_start=ts_start,
        timestamp_end=ts_start + 2.5,
        confidence=0.82,
        reel_targets=[reel],
        frame_start=int(ts_start * 30),
        frame_end=int((ts_start + 2.5) * 30),
    )


@pytest.mark.unit
class TestEventLog:
    def test_append_and_read(self, tmp_path: Path):
        log = EventLog(tmp_path / "events.jsonl")
        e = _make_event("e001")
        log.append(e)
        events = log.read_all()
        assert len(events) == 1
        assert events[0].event_id == "e001"

    def test_append_many(self, tmp_path: Path):
        log = EventLog(tmp_path / "events.jsonl")
        events = [_make_event(f"e{i:03d}", ts_start=float(i * 10)) for i in range(5)]
        log.append_many(events)
        assert log.count() == 5

    def test_events_sorted_by_timestamp(self, tmp_path: Path):
        log = EventLog(tmp_path / "events.jsonl")
        log.append(_make_event("e_late", ts_start=100.0))
        log.append(_make_event("e_early", ts_start=10.0))
        events = log.read_all()
        assert events[0].event_id == "e_early"
        assert events[1].event_id == "e_late"

    def test_deduplication_last_write_wins(self, tmp_path: Path):
        log = EventLog(tmp_path / "events.jsonl")
        e1 = _make_event("e001")
        log.append(e1)
        # Write same event_id with different confidence
        data = e1.model_dump()
        data["confidence"] = 0.99
        log.append(Event(**data))
        events = log.read_all()
        assert len(events) == 1
        assert events[0].confidence == 0.99  # Last write wins

    def test_filter_by_reel(self, tmp_path: Path):
        log = EventLog(tmp_path / "events.jsonl")
        log.append(_make_event("gk1", reel="goalkeeper"))
        log.append(_make_event("hl1", reel="highlights"))
        gk_events = log.filter_by_reel("goalkeeper")
        hl_events = log.filter_by_reel("highlights")
        assert len(gk_events) == 1 and gk_events[0].event_id == "gk1"
        assert len(hl_events) == 1 and hl_events[0].event_id == "hl1"

    def test_corrupt_line_skipped(self, tmp_path: Path):
        log_path = tmp_path / "events.jsonl"
        # Write one valid + one corrupt line
        valid_event = _make_event("e001")
        log_path.write_text(valid_event.model_dump_json() + "\n{CORRUPT JSON}\n")
        log = EventLog(log_path)
        events = log.read_all()
        assert len(events) == 1  # Corrupt line skipped

    def test_empty_log_returns_empty_list(self, tmp_path: Path):
        log = EventLog(tmp_path / "no_events.jsonl")
        assert log.read_all() == []
        assert log.count() == 0
        assert not log.exists()

    def test_iter_events_streaming(self, tmp_path: Path):
        log = EventLog(tmp_path / "events.jsonl")
        for i in range(10):
            log.append(_make_event(f"e{i:03d}", ts_start=float(i)))
        count = sum(1 for _ in log.iter_events())
        assert count == 10


@pytest.mark.unit
class TestMergeLogs:
    def test_merge_two_logs(self, tmp_path: Path):
        log1 = EventLog(tmp_path / "log1.jsonl")
        log2 = EventLog(tmp_path / "log2.jsonl")
        log1.append(_make_event("e001", ts_start=10.0))
        log2.append(_make_event("e002", ts_start=20.0))

        dest = tmp_path / "merged.jsonl"
        count = merge_logs([tmp_path / "log1.jsonl", tmp_path / "log2.jsonl"], dest)
        assert count == 2

        merged = EventLog(dest)
        events = merged.read_all()
        assert len(events) == 2
        assert events[0].timestamp_start == 10.0

    def test_merge_deduplicates(self, tmp_path: Path):
        log1 = EventLog(tmp_path / "log1.jsonl")
        log2 = EventLog(tmp_path / "log2.jsonl")
        same = _make_event("e001")
        log1.append(same)
        log2.append(same)

        dest = tmp_path / "merged.jsonl"
        count = merge_logs([tmp_path / "log1.jsonl", tmp_path / "log2.jsonl"], dest)
        assert count == 1
