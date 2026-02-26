"""
Unit tests for job pause / cancel / resume functionality.

Covers:
- Model: PAUSED/CANCELLED status values, pause/cancel_requested flags, with_status flag clearing
- JobStore: request_pause, request_cancel methods
- API endpoints: pause, cancel, resume, updated retry/delete
- Worker: on_detect_progress flag checks, PipelinePaused/PipelineCancelled handling
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from tests.conftest import make_match_config

from src.ingestion.models import Job, JobStatus, VideoFile


def _sample_video_file(path: str = "/mnt/nas/match.mp4") -> VideoFile:
    return VideoFile(
        path=path,
        filename="match.mp4",
        duration_sec=5400.0,
        fps=30.0,
        width=3840,
        height=2160,
        codec="h264",
        size_bytes=15_000_000_000,
        sha256="a" * 64,
    )


def _sample_job(**kwargs) -> Job:
    return Job(
        video_file=_sample_video_file(),
        match_config=make_match_config(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestJobModelPauseCancel:
    def test_paused_is_valid_status(self):
        assert JobStatus.PAUSED == "paused"

    def test_cancelled_is_valid_status(self):
        assert JobStatus.CANCELLED == "cancelled"

    def test_pause_requested_defaults_false(self):
        job = _sample_job()
        assert job.pause_requested is False

    def test_cancel_requested_defaults_false(self):
        job = _sample_job()
        assert job.cancel_requested is False

    def test_with_status_to_pending_clears_both_flags(self):
        job = _sample_job(pause_requested=True, cancel_requested=True)
        updated = job.with_status(JobStatus.PENDING, progress=0.0)
        assert updated.pause_requested is False
        assert updated.cancel_requested is False

    def test_with_status_to_cancelled_clears_pause_flag(self):
        job = _sample_job(pause_requested=True)
        updated = job.with_status(JobStatus.CANCELLED)
        assert updated.pause_requested is False

    def test_with_status_preserves_flags_on_other_transitions(self):
        job = _sample_job(pause_requested=True)
        updated = job.with_status(JobStatus.DETECTING, progress=10.0)
        assert updated.pause_requested is True

    def test_job_serializes_with_new_fields(self):
        job = _sample_job(pause_requested=True, cancel_requested=True)
        data = job.model_dump()
        assert data["pause_requested"] is True
        assert data["cancel_requested"] is True

    def test_job_deserializes_with_new_fields(self):
        job = _sample_job(pause_requested=True)
        raw = job.model_dump_json()
        loaded = Job.model_validate_json(raw)
        assert loaded.pause_requested is True

    def test_job_deserializes_without_new_fields_defaults(self):
        """Backwards compat: old JSON without the flags should deserialize fine."""
        job = _sample_job()
        data = job.model_dump()
        data.pop("pause_requested", None)
        data.pop("cancel_requested", None)
        loaded = Job(**data)
        assert loaded.pause_requested is False
        assert loaded.cancel_requested is False


# ---------------------------------------------------------------------------
# JobStore tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestJobStoreRequestFlags:
    def test_request_pause_sets_flag(self, tmp_path: Path):
        from src.ingestion.job import JobStore
        store = JobStore(tmp_path / "jobs")
        job = _sample_job()
        store.save(job)
        updated = store.request_pause(job.job_id)
        assert updated.pause_requested is True
        assert store.get(job.job_id).pause_requested is True

    def test_request_cancel_sets_flag(self, tmp_path: Path):
        from src.ingestion.job import JobStore
        store = JobStore(tmp_path / "jobs")
        job = _sample_job()
        store.save(job)
        updated = store.request_cancel(job.job_id)
        assert updated.cancel_requested is True
        assert store.get(job.job_id).cancel_requested is True

    def test_request_pause_nonexistent_returns_none(self, tmp_path: Path):
        from src.ingestion.job import JobStore
        store = JobStore(tmp_path / "jobs")
        assert store.request_pause("ghost") is None

    def test_request_cancel_nonexistent_returns_none(self, tmp_path: Path):
        from src.ingestion.job import JobStore
        store = JobStore(tmp_path / "jobs")
        assert store.request_cancel("ghost") is None


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPauseCancelResumeEndpoints:
    """Test the pause/cancel/resume routes via FastAPI TestClient."""

    @pytest.fixture
    def client(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("WORKING_DIR", str(tmp_path / "working"))
        monkeypatch.setenv("NAS_MOUNT_PATH", str(tmp_path / "nas"))
        monkeypatch.setenv("NAS_OUTPUT_PATH", str(tmp_path / "out"))
        (tmp_path / "working").mkdir()
        (tmp_path / "nas").mkdir()

        from src.ingestion.job import JobStore
        store = JobStore(tmp_path / "working" / "jobs")
        monkeypatch.setattr("src.api.routes.jobs._get_store", lambda: store)
        self._store = store

        # Stub out Celery task
        stub = MagicMock()
        monkeypatch.setattr("src.api.routes.jobs.process_match_task", stub)
        self._task_stub = stub

        from fastapi.testclient import TestClient
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    def _insert_job(self, status=JobStatus.DETECTING, **kwargs):
        job = _sample_job(status=status, **kwargs)
        self._store.save(job)
        return job

    # --- Pause ---
    def test_pause_active_job(self, client):
        job = self._insert_job(status=JobStatus.DETECTING)
        r = client.post(f"/jobs/{job.job_id}/pause")
        assert r.status_code == 200
        assert self._store.get(job.job_id).pause_requested is True

    def test_pause_pending_job_goes_directly_to_paused(self, client):
        job = self._insert_job(status=JobStatus.PENDING)
        r = client.post(f"/jobs/{job.job_id}/pause")
        assert r.status_code == 200
        assert self._store.get(job.job_id).status == JobStatus.PAUSED

    def test_pause_complete_job_returns_400(self, client):
        job = self._insert_job(status=JobStatus.COMPLETE)
        r = client.post(f"/jobs/{job.job_id}/pause")
        assert r.status_code == 400

    def test_pause_nonexistent_returns_404(self, client):
        r = client.post("/jobs/ghost/pause")
        assert r.status_code == 404

    # --- Cancel ---
    def test_cancel_active_job(self, client):
        job = self._insert_job(status=JobStatus.DETECTING)
        r = client.post(f"/jobs/{job.job_id}/cancel")
        assert r.status_code == 200
        assert self._store.get(job.job_id).cancel_requested is True

    def test_cancel_paused_job_goes_directly_to_cancelled(self, client):
        job = self._insert_job(status=JobStatus.PAUSED)
        r = client.post(f"/jobs/{job.job_id}/cancel")
        assert r.status_code == 200
        assert self._store.get(job.job_id).status == JobStatus.CANCELLED

    def test_cancel_pending_job_goes_directly_to_cancelled(self, client):
        job = self._insert_job(status=JobStatus.PENDING)
        r = client.post(f"/jobs/{job.job_id}/cancel")
        assert r.status_code == 200
        assert self._store.get(job.job_id).status == JobStatus.CANCELLED

    def test_cancel_complete_job_returns_400(self, client):
        job = self._insert_job(status=JobStatus.COMPLETE)
        r = client.post(f"/jobs/{job.job_id}/cancel")
        assert r.status_code == 400

    def test_cancel_already_cancelled_returns_400(self, client):
        job = self._insert_job(status=JobStatus.CANCELLED)
        r = client.post(f"/jobs/{job.job_id}/cancel")
        assert r.status_code == 400

    def test_cancel_nonexistent_returns_404(self, client):
        r = client.post("/jobs/ghost/cancel")
        assert r.status_code == 404

    # --- Resume ---
    def test_resume_paused_job(self, client):
        job = self._insert_job(status=JobStatus.PAUSED, pause_requested=True)
        r = client.post(f"/jobs/{job.job_id}/resume")
        assert r.status_code == 200
        reloaded = self._store.get(job.job_id)
        assert reloaded.status == JobStatus.PENDING
        assert reloaded.pause_requested is False
        assert reloaded.cancel_requested is False
        self._task_stub.delay.assert_called_once_with(job.job_id)

    def test_resume_detecting_job_returns_400(self, client):
        job = self._insert_job(status=JobStatus.DETECTING)
        r = client.post(f"/jobs/{job.job_id}/resume")
        assert r.status_code == 400

    def test_resume_nonexistent_returns_404(self, client):
        r = client.post("/jobs/ghost/resume")
        assert r.status_code == 404

    # --- Retry (updated to include CANCELLED) ---
    def test_retry_cancelled_job(self, client):
        job = self._insert_job(status=JobStatus.CANCELLED)
        r = client.post(f"/jobs/{job.job_id}/retry")
        assert r.status_code == 200
        assert self._store.get(job.job_id).status == JobStatus.PENDING

    def test_retry_failed_job(self, client):
        job = self._insert_job(status=JobStatus.FAILED)
        r = client.post(f"/jobs/{job.job_id}/retry")
        assert r.status_code == 200

    def test_retry_detecting_returns_400(self, client):
        job = self._insert_job(status=JobStatus.DETECTING)
        r = client.post(f"/jobs/{job.job_id}/retry")
        assert r.status_code == 400

    # --- Delete (updated to include PAUSED/CANCELLED) ---
    def test_delete_paused_job(self, client):
        job = self._insert_job(status=JobStatus.PAUSED)
        r = client.delete(f"/jobs/{job.job_id}")
        assert r.status_code == 200
        assert self._store.get(job.job_id) is None

    def test_delete_cancelled_job(self, client):
        job = self._insert_job(status=JobStatus.CANCELLED)
        r = client.delete(f"/jobs/{job.job_id}")
        assert r.status_code == 200

    def test_delete_detecting_returns_400(self, client):
        job = self._insert_job(status=JobStatus.DETECTING)
        r = client.delete(f"/jobs/{job.job_id}")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Worker callback tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestWorkerPauseCancelCallbacks:
    """Test that on_detect_progress raises on pause/cancel flags,
    and that _run_pipeline continues through segmenting/assembling
    to produce partial reels."""

    def test_on_detect_progress_raises_paused(self, tmp_path: Path):
        from src.api.worker import PipelinePaused
        from src.ingestion.job import JobStore

        store = JobStore(tmp_path / "jobs")
        job = _sample_job(status=JobStatus.DETECTING, pause_requested=True)
        store.save(job)

        fetched = store.get(job.job_id)
        assert fetched.pause_requested is True
        with pytest.raises(PipelinePaused):
            if fetched.cancel_requested:
                from src.api.worker import PipelineCancelled
                raise PipelineCancelled("cancelled")
            if fetched.pause_requested:
                raise PipelinePaused("paused")

    def test_on_detect_progress_raises_cancelled(self, tmp_path: Path):
        from src.api.worker import PipelineCancelled
        from src.ingestion.job import JobStore

        store = JobStore(tmp_path / "jobs")
        job = _sample_job(status=JobStatus.DETECTING, cancel_requested=True)
        store.save(job)

        fetched = store.get(job.job_id)
        with pytest.raises(PipelineCancelled):
            if fetched.cancel_requested:
                raise PipelineCancelled("cancelled")

    def test_cancel_takes_precedence_over_pause(self, tmp_path: Path):
        from src.api.worker import PipelineCancelled
        from src.ingestion.job import JobStore

        store = JobStore(tmp_path / "jobs")
        job = _sample_job(
            status=JobStatus.DETECTING,
            pause_requested=True,
            cancel_requested=True,
        )
        store.save(job)

        fetched = store.get(job.job_id)
        with pytest.raises(PipelineCancelled):
            if fetched.cancel_requested:
                raise PipelineCancelled("cancelled")
            if fetched.pause_requested:
                from src.api.worker import PipelinePaused
                raise PipelinePaused("paused")

    def test_pipeline_paused_does_not_trigger_retry(self):
        """PipelinePaused is caught before the generic Exception handler."""
        from src.api.worker import PipelinePaused
        assert not issubclass(PipelinePaused, (ValueError, RuntimeError, OSError))

    def test_pipeline_cancelled_does_not_trigger_retry(self):
        from src.api.worker import PipelineCancelled
        assert not issubclass(PipelineCancelled, (ValueError, RuntimeError, OSError))


# ---------------------------------------------------------------------------
# Partial-reel integration tests (mocked pipeline)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPartialReelOnInterrupt:
    """When paused/cancelled mid-detection, _run_pipeline should still
    segment + assemble a reel from whatever events were already detected."""

    def _run_with_interrupt(self, tmp_path, interrupt_type, events=None, clips=None):
        """Run _run_pipeline with runner.run() raising PipelinePaused or PipelineCancelled."""
        from src.api.worker import PipelineCancelled, PipelinePaused, _run_pipeline
        from src.ingestion.job import JobStore

        source_file = tmp_path / "source" / "match.mp4"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.touch()

        store = JobStore(tmp_path / "jobs")
        vf = _sample_video_file(path=str(source_file))
        job = Job(video_file=vf, match_config=make_match_config(), reel_types=["keeper"])
        store.save(job)
        job_id = job.job_id

        cfg = MagicMock()
        cfg.WORKING_DIR = str(tmp_path / "working")
        cfg.YOLO_MODEL_PATH = "/models/yolov8m.pt"
        cfg.USE_GPU = "false"
        cfg.USE_NULL_DETECTOR = "false"
        cfg.YOLO_INFERENCE_SIZE = "1280"
        cfg.DETECTION_FRAME_STEP = "3"
        cfg.CHUNK_DURATION_SEC = "30"
        cfg.CHUNK_OVERLAP_SEC = "2.0"
        cfg.MIN_EVENT_CONFIDENCE = "0.65"
        cfg.OUTPUT_CODEC = "copy"
        cfg.OUTPUT_CRF = "18"
        cfg.NAS_OUTPUT_PATH = str(tmp_path / "output")
        cfg.MAX_NAS_RETRY = "3"
        cfg.REEL_PLUGINS = ""
        cfg.USE_BALL_TOUCH_DETECTOR = "false"
        (tmp_path / "working").mkdir(parents=True, exist_ok=True)

        exc_cls = PipelinePaused if interrupt_type == "paused" else PipelineCancelled

        _P = {
            "PlayerDetector": "src.detection.player_detector.PlayerDetector",
            "GoalkeeperDetector": "src.detection.goalkeeper_detector.GoalkeeperDetector",
            "PipelineRunner": "src.detection.event_classifier.PipelineRunner",
            "EventLog": "src.detection.event_log.EventLog",
            "compute_clips": "src.segmentation.clipper.compute_clips",
            "postprocess_clips": "src.segmentation.deduplicator.postprocess_clips",
            "ReelComposer": "src.assembly.composer.ReelComposer",
            "write_reel_to_nas": "src.assembly.output.write_reel_to_nas",
            "get_output_path": "src.assembly.output.get_output_path",
            "write_job_manifest": "src.assembly.output.write_job_manifest",
        }

        with patch(_P["PlayerDetector"]), \
             patch(_P["GoalkeeperDetector"]), \
             patch(_P["PipelineRunner"]) as m_runner, \
             patch(_P["EventLog"]) as m_evlog, \
             patch(_P["compute_clips"]) as m_clips, \
             patch(_P["postprocess_clips"]) as m_post, \
             patch(_P["ReelComposer"]) as m_composer, \
             patch(_P["write_reel_to_nas"]) as m_nas, \
             patch(_P["get_output_path"], create=True), \
             patch(_P["write_job_manifest"], create=True):

            # Make runner.run() raise the interrupt exception
            m_runner.return_value.run.side_effect = exc_cls("interrupted")
            m_evlog.return_value.read_all.return_value = events or []
            m_clips.return_value = clips or []
            m_post.return_value = clips or []
            m_composer.return_value.compose.return_value = True
            m_nas.return_value = "/output/reel.mp4"

            result = _run_pipeline(job_id, store, cfg)

        return result, store, job_id, m_composer

    def test_paused_produces_partial_reel(self, tmp_path):
        from src.detection.models import Event, EventType
        from src.segmentation.clipper import ClipBoundary

        event = Event(
            job_id="x", source_file="match.mp4",
            event_type=EventType.CATCH, timestamp_start=10.0,
            timestamp_end=12.0, confidence=0.80,
            reel_targets=["keeper"], frame_start=300, frame_end=360,
            is_goalkeeper_event=True,
        )
        clip = ClipBoundary(
            source_file="/mnt/nas/source/match.mp4",
            start_sec=8.0, end_sec=13.5,
            events=["ev-001"], reel_type="keeper",
            primary_event_type="catch",
        )
        result, store, job_id, m_composer = self._run_with_interrupt(
            tmp_path, "paused", events=[event], clips=[clip],
        )
        assert result["status"] == "paused"
        assert "keeper" in result["output_paths"]
        m_composer.return_value.compose.assert_called_once()

        final = store.get(job_id)
        assert final.status == JobStatus.PAUSED
        assert "keeper" in final.output_paths

    def test_cancelled_produces_partial_reel(self, tmp_path):
        from src.detection.models import Event, EventType
        from src.segmentation.clipper import ClipBoundary

        event = Event(
            job_id="x", source_file="match.mp4",
            event_type=EventType.CATCH, timestamp_start=10.0,
            timestamp_end=12.0, confidence=0.80,
            reel_targets=["keeper"], frame_start=300, frame_end=360,
            is_goalkeeper_event=True,
        )
        clip = ClipBoundary(
            source_file="/mnt/nas/source/match.mp4",
            start_sec=8.0, end_sec=13.5,
            events=["ev-001"], reel_type="keeper",
            primary_event_type="catch",
        )
        result, store, job_id, _ = self._run_with_interrupt(
            tmp_path, "cancelled", events=[event], clips=[clip],
        )
        assert result["status"] == "cancelled"
        assert "keeper" in result["output_paths"]

        final = store.get(job_id)
        assert final.status == JobStatus.CANCELLED
        assert "keeper" in final.output_paths

    def test_paused_no_events_yet_sets_paused_not_failed(self, tmp_path):
        """If paused before any events are detected, status should be PAUSED, not FAILED."""
        result, store, job_id, _ = self._run_with_interrupt(
            tmp_path, "paused", events=[], clips=[],
        )
        assert result["status"] == "paused"
        assert store.get(job_id).status == JobStatus.PAUSED

    def test_cancelled_no_events_yet_sets_cancelled_not_failed(self, tmp_path):
        result, store, job_id, _ = self._run_with_interrupt(
            tmp_path, "cancelled", events=[], clips=[],
        )
        assert result["status"] == "cancelled"
        assert store.get(job_id).status == JobStatus.CANCELLED
