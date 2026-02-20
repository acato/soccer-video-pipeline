"""Unit tests for src/ingestion/job.py (JobStore)"""
from pathlib import Path
import pytest
from src.ingestion.job import JobStore, create_job
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


@pytest.mark.unit
class TestJobStore:
    def test_save_and_get(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        vf = _sample_video_file()
        job = Job(video_file=vf)
        store.save(job)
        retrieved = store.get(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_returns_none(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        assert store.get("does-not-exist") is None

    def test_update_status(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        job = Job(video_file=_sample_video_file())
        store.save(job)
        updated = store.update_status(job.job_id, JobStatus.DETECTING, progress=15.0)
        assert updated.status == JobStatus.DETECTING
        assert updated.progress_pct == 15.0
        # Verify persisted
        reloaded = store.get(job.job_id)
        assert reloaded.status == JobStatus.DETECTING

    def test_update_status_not_found_returns_none(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        result = store.update_status("ghost-id", JobStatus.FAILED)
        assert result is None

    def test_list_all_sorted_newest_first(self, tmp_path: Path):
        import time
        store = JobStore(tmp_path / "jobs")
        job1 = Job(video_file=_sample_video_file())
        time.sleep(0.01)
        job2 = Job(video_file=_sample_video_file())
        store.save(job1)
        store.save(job2)
        jobs = store.list_all()
        assert jobs[0].job_id == job2.job_id  # Newest first

    def test_save_is_atomic(self, tmp_path: Path):
        """Save should write to .tmp then rename — no partial writes visible."""
        store = JobStore(tmp_path / "jobs")
        job = Job(video_file=_sample_video_file())
        store.save(job)
        # Verify no .tmp file left behind
        tmp_files = list((tmp_path / "jobs").glob("*.tmp"))
        assert tmp_files == []

    def test_path_traversal_prevented(self, tmp_path: Path):
        """Job IDs with ../ should not escape the jobs directory."""
        store = JobStore(tmp_path / "jobs")
        safe_id = store._path("../../etc/passwd")
        # Path must stay inside the jobs directory (no directory escape)
        assert str(safe_id).startswith(str(tmp_path / "jobs"))

    def test_job_with_status_immutable_update(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        job = Job(video_file=_sample_video_file())
        updated = job.with_status(JobStatus.COMPLETE, progress=100.0)
        assert updated.status == JobStatus.COMPLETE
        assert job.status == JobStatus.PENDING  # Original unchanged

    def test_output_paths_updated(self, tmp_path: Path):
        store = JobStore(tmp_path / "jobs")
        job = Job(video_file=_sample_video_file())
        store.save(job)
        store.update_status(
            job.job_id, JobStatus.COMPLETE,
            output_paths={"goalkeeper": "/nas/output/job1/goalkeeper_reel.mp4"}
        )
        reloaded = store.get(job.job_id)
        assert "goalkeeper" in reloaded.output_paths
