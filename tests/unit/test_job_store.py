"""Unit tests for src/ingestion/job.py (JobStore)"""
from pathlib import Path
import pytest
from src.ingestion.job import JobStore, create_job
from src.ingestion.models import Job, JobStatus, VideoFile
from tests.conftest import make_match_config


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


@pytest.mark.unit
class TestJobStore:
    def test_save_and_get(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        job = _sample_job()
        store.save(job)
        retrieved = store.get(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_returns_none(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        assert store.get("does-not-exist") is None

    def test_update_status(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        job = _sample_job()
        store.save(job)
        updated = store.update_status(job.job_id, JobStatus.DETECTING, progress=15.0)
        assert updated.status == JobStatus.DETECTING
        assert updated.progress_pct == 15.0
        assert store.get(job.job_id).status == JobStatus.DETECTING

    def test_update_status_not_found_returns_none(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        assert store.update_status("ghost-id", JobStatus.FAILED) is None

    def test_list_all_sorted_newest_first(self, tmp_path: Path):
        import time
        store = JobStore(tmp_path / "jobs")
        job1 = _sample_job()
        time.sleep(0.01)
        job2 = _sample_job()
        store.save(job1)
        store.save(job2)
        jobs = store.list_all()
        assert jobs[0].job_id == job2.job_id

    def test_save_is_atomic(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        store.save(_sample_job())
        assert list((tmp_path / "jobs").glob("*.tmp")) == []

    def test_path_traversal_prevented(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        safe_id = store._path("../../etc/passwd")
        assert str(safe_id).startswith(str(tmp_path / "jobs"))

    def test_job_with_status_immutable_update(self, tmp_path: Path):
        job = _sample_job()
        updated = job.with_status(JobStatus.COMPLETE, progress=100.0)
        assert updated.status == JobStatus.COMPLETE
        assert job.status == JobStatus.PENDING

    def test_with_status_clears_error_on_non_failed(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        job = _sample_job()
        store.save(job)
        store.update_status(job.job_id, JobStatus.FAILED, error="something broke")
        assert store.get(job.job_id).error == "something broke"
        store.update_status(job.job_id, JobStatus.DETECTING, progress=5.0)
        reloaded = store.get(job.job_id)
        assert reloaded.status == JobStatus.DETECTING
        assert reloaded.error is None

    def test_with_status_preserves_error_on_failed(self, tmp_path: Path):
        job = _sample_job(error="original error")
        assert job.with_status(JobStatus.FAILED).error == "original error"

    def test_with_status_explicit_error_on_any_status(self, tmp_path: Path):
        job = _sample_job()
        assert job.with_status(JobStatus.DETECTING, error="forced error").error == "forced error"

    def test_output_paths_updated(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        job = _sample_job()
        store.save(job)
        store.update_status(
            job.job_id, JobStatus.COMPLETE,
            output_paths={"keeper": "/nas/output/job1/keeper_reel.mp4"}
        )
        assert "keeper" in store.get(job.job_id).output_paths

    def test_match_config_persisted(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        job = _sample_job()
        store.save(job)
        reloaded = store.get(job.job_id)
        assert reloaded.match_config.team.team_name == "Home FC"
        assert reloaded.match_config.opponent.gk_color == "neon_green"
