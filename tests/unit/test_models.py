"""
Unit tests for Pydantic data models — serialization, validation, business logic.
"""
import pytest
from src.detection.models import (
    BoundingBox, Detection, Event, EventType, Track,
    EVENT_REEL_MAP, EVENT_CONFIDENCE_THRESHOLDS
)
from src.ingestion.models import Job, JobStatus, VideoFile


@pytest.mark.unit
class TestEventModel:

    def test_should_include_above_threshold(self):
        ev = Event(
            job_id="j1", source_file="m.mp4",
            event_type=EventType.GOAL, timestamp_start=10, timestamp_end=11,
            confidence=0.90, reel_targets=["highlights"],
            frame_start=300, frame_end=330,
        )
        assert ev.should_include() is True

    def test_should_exclude_below_threshold(self):
        ev = Event(
            job_id="j1", source_file="m.mp4",
            event_type=EventType.GOAL, timestamp_start=10, timestamp_end=11,
            confidence=0.50,  # Below GOAL threshold of 0.85
            reel_targets=["highlights"],
            frame_start=300, frame_end=330,
        )
        assert ev.should_include() is False

    def test_review_override_true_forces_include(self):
        ev = Event(
            job_id="j1", source_file="m.mp4",
            event_type=EventType.GOAL, timestamp_start=10, timestamp_end=11,
            confidence=0.10,  # Very low
            reel_targets=["highlights"],
            frame_start=300, frame_end=330,
            review_override=True,
        )
        assert ev.should_include() is True

    def test_review_override_false_forces_exclude(self):
        ev = Event(
            job_id="j1", source_file="m.mp4",
            event_type=EventType.GOAL, timestamp_start=10, timestamp_end=11,
            confidence=0.99,  # Very high
            reel_targets=["highlights"],
            frame_start=300, frame_end=330,
            review_override=False,
        )
        assert ev.should_include() is False

    def test_duration_sec_computed(self):
        ev = Event(
            job_id="j1", source_file="m.mp4",
            event_type=EventType.CATCH, timestamp_start=10.5, timestamp_end=13.5,
            confidence=0.80, reel_targets=["goalkeeper"],
            frame_start=315, frame_end=405,
        )
        assert abs(ev.duration_sec - 3.0) < 0.001

    def test_serialization_roundtrip(self):
        ev = Event(
            job_id="j1", source_file="m.mp4",
            event_type=EventType.SHOT_STOP_DIVING, timestamp_start=100, timestamp_end=102,
            confidence=0.75, reel_targets=["goalkeeper"],
            frame_start=3000, frame_end=3060,
        )
        json_str = ev.model_dump_json()
        ev2 = Event.model_validate_json(json_str)
        assert ev2.event_id == ev.event_id
        assert ev2.event_type == ev.event_type
        assert ev2.confidence == ev.confidence

    def test_all_event_types_in_reel_map(self):
        """Every EventType must have a reel target mapping."""
        for et in EventType:
            assert et in EVENT_REEL_MAP, f"EventType.{et} missing from EVENT_REEL_MAP"

    def test_all_event_types_have_threshold(self):
        """Every EventType must have a confidence threshold."""
        for et in EventType:
            assert et in EVENT_CONFIDENCE_THRESHOLDS, f"EventType.{et} missing from CONFIDENCE_THRESHOLDS"


@pytest.mark.unit
class TestBoundingBox:

    def test_center_x(self):
        bb = BoundingBox(x=0.1, y=0.2, width=0.4, height=0.3)
        assert abs(bb.center_x - 0.3) < 0.001

    def test_center_y(self):
        bb = BoundingBox(x=0.1, y=0.2, width=0.4, height=0.3)
        assert abs(bb.center_y - 0.35) < 0.001

    def test_area(self):
        bb = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.4)
        assert abs(bb.area - 0.2) < 0.001


@pytest.mark.unit
class TestJobModel:

    def _make_video_file(self) -> VideoFile:
        return VideoFile(
            path="/nas/match.mp4", filename="match.mp4",
            duration_sec=5400, fps=30, width=3840, height=2160,
            codec="h264", size_bytes=10_000_000_000, sha256="abc123",
        )

    def test_default_status_is_pending(self):
        job = Job(video_file=self._make_video_file(), reel_types=["goalkeeper"])
        assert job.status == JobStatus.PENDING

    def test_with_status_immutable_update(self):
        job = Job(video_file=self._make_video_file(), reel_types=["goalkeeper"])
        updated = job.with_status(JobStatus.DETECTING, progress=10.0)
        assert updated.status == JobStatus.DETECTING
        assert updated.progress_pct == 10.0
        assert job.status == JobStatus.PENDING  # Original unchanged

    def test_job_id_is_uuid(self):
        import uuid
        job = Job(video_file=self._make_video_file(), reel_types=["goalkeeper"])
        uuid.UUID(job.job_id)  # Raises if invalid

    def test_serialization_roundtrip(self):
        job = Job(video_file=self._make_video_file(), reel_types=["goalkeeper", "highlights"])
        json_str = job.model_dump_json()
        job2 = Job.model_validate_json(json_str)
        assert job2.job_id == job.job_id
        assert job2.reel_types == job.reel_types
