"""Tests for per-job game_start_sec feature."""
import json
import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.models import Job, VideoFile


def _make_video_file(**kwargs):
    defaults = dict(
        path="/nas/match.mp4",
        filename="match.mp4",
        duration_sec=6000.0,
        fps=30.0,
        width=3840,
        height=2160,
        codec="h264",
        size_bytes=1_000_000,
        sha256="abc123",
    )
    defaults.update(kwargs)
    return VideoFile(**defaults)


@pytest.mark.unit
class TestJobGameStartSec:
    def test_default_zero(self):
        job = Job(video_file=_make_video_file())
        assert job.game_start_sec == 0.0

    def test_explicit_value(self):
        job = Job(video_file=_make_video_file(), game_start_sec=300.0)
        assert job.game_start_sec == 300.0

    def test_round_trip_serialization(self):
        job = Job(video_file=_make_video_file(), game_start_sec=270.5)
        data = json.loads(job.model_dump_json())
        restored = Job(**data)
        assert restored.game_start_sec == 270.5

    def test_with_status_preserves_game_start_sec(self):
        from src.ingestion.models import JobStatus
        job = Job(video_file=_make_video_file(), game_start_sec=180.0)
        updated = job.with_status(JobStatus.DETECTING, progress=50.0)
        assert updated.game_start_sec == 180.0


@pytest.mark.unit
class TestCreateJobGameStartSec:
    def test_create_job_passes_game_start_sec(self, tmp_path):
        from src.ingestion.job import JobStore, create_job
        store = JobStore(tmp_path / "jobs")
        vf = _make_video_file()
        job = create_job(vf, ["keeper"], store, game_start_sec=300.0)
        assert job.game_start_sec == 300.0
        # Verify persisted
        loaded = store.get(job.job_id)
        assert loaded.game_start_sec == 300.0

    def test_create_job_default_zero(self, tmp_path):
        from src.ingestion.job import JobStore, create_job
        store = JobStore(tmp_path / "jobs")
        vf = _make_video_file()
        job = create_job(vf, ["keeper"], store)
        assert job.game_start_sec == 0.0


@pytest.mark.unit
class TestChunkStartsWithGameStart:
    def _make_runner(self, duration=600.0, chunk_sec=30, game_start_sec=0.0):
        from src.detection.event_classifier import PipelineRunner
        vf = _make_video_file(duration_sec=duration)
        runner = PipelineRunner(
            job_id="test",
            video_file=vf,
            player_detector=MagicMock(),
            gk_detector=MagicMock(),
            event_log=MagicMock(),
            chunk_sec=chunk_sec,
            game_start_sec=game_start_sec,
        )
        return runner

    def test_default_starts_at_zero(self):
        runner = self._make_runner(duration=100, chunk_sec=30)
        starts = runner._chunk_starts(100)
        assert starts[0] == 0.0

    def test_game_start_skips_warmup(self):
        runner = self._make_runner(duration=600, chunk_sec=30, game_start_sec=300.0)
        starts = runner._chunk_starts(600)
        assert starts[0] == 300.0
        assert all(s >= 300.0 for s in starts)

    def test_game_start_chunk_count(self):
        # 600s video, skip first 300s, 30s chunks → 10 chunks
        runner = self._make_runner(duration=600, chunk_sec=30, game_start_sec=300.0)
        starts = runner._chunk_starts(600)
        assert len(starts) == 10

    def test_game_start_beyond_duration_yields_empty(self):
        runner = self._make_runner(duration=100, chunk_sec=30, game_start_sec=200.0)
        starts = runner._chunk_starts(100)
        assert starts == []


@pytest.mark.unit
class TestSubmitJobRequestGameStartSec:
    def test_request_model_accepts_game_start_sec(self):
        from src.api.routes.jobs import SubmitJobRequest
        req = SubmitJobRequest(nas_path="/video.mp4", game_start_sec=300.0)
        assert req.game_start_sec == 300.0

    def test_request_model_default_zero(self):
        from src.api.routes.jobs import SubmitJobRequest
        req = SubmitJobRequest(nas_path="/video.mp4")
        assert req.game_start_sec == 0.0
