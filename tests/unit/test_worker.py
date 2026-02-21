"""
Unit tests for src/api/worker.py

Tests pipeline orchestration logic, config type casting, ReelComposer
instantiation, and Celery app exposure.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


def _make_video_file():
    """Build a minimal VideoFile for testing."""
    from src.ingestion.models import VideoFile
    return VideoFile(
        path="/mnt/nas/source/match.mp4",
        filename="match.mp4",
        duration_sec=5400.0,
        fps=30.0,
        width=3840,
        height=2160,
        codec="h264",
        size_bytes=20_000_000_000,
        sha256="abc123",
    )


def _make_job(job_id="test-job-001", reel_types=None):
    """Build a minimal Job for testing."""
    from src.ingestion.models import Job
    return Job(
        job_id=job_id,
        video_file=_make_video_file(),
        reel_types=reel_types or ["goalkeeper"],
    )


def _make_string_config(tmp_path):
    """Simulate _Config dynamic accessor: all values returned as strings."""
    cfg = MagicMock()
    cfg.WORKING_DIR = str(tmp_path / "working")
    cfg.YOLO_MODEL_PATH = "/models/yolov8m.pt"
    cfg.USE_GPU = "false"
    cfg.YOLO_INFERENCE_SIZE = "1280"
    cfg.DETECTION_FRAME_STEP = "3"
    cfg.CHUNK_DURATION_SEC = "30"
    cfg.CHUNK_OVERLAP_SEC = "2.0"
    cfg.MIN_EVENT_CONFIDENCE = "0.65"
    cfg.PRE_EVENT_PAD_SEC = "3.0"
    cfg.POST_EVENT_PAD_SEC = "5.0"
    cfg.OUTPUT_CODEC = "copy"
    cfg.OUTPUT_CRF = "18"
    cfg.OUTPUT_AUDIO_CODEC = "copy"
    cfg.NAS_OUTPUT_PATH = str(tmp_path / "output")
    cfg.MAX_NAS_RETRY = "3"
    return cfg


# Patch targets at source modules where _run_pipeline imports them from
_P = {
    "PlayerDetector":       "src.detection.player_detector.PlayerDetector",
    "GoalkeeperDetector":   "src.detection.goalkeeper_detector.GoalkeeperDetector",
    "PipelineRunner":       "src.detection.event_classifier.PipelineRunner",
    "EventLog":             "src.detection.event_log.EventLog",
    "compute_clips":        "src.segmentation.clipper.compute_clips",
    "postprocess_clips":    "src.segmentation.deduplicator.postprocess_clips",
    "ReelComposer":         "src.assembly.composer.ReelComposer",
    "write_reel_to_nas":    "src.assembly.output.write_reel_to_nas",
    "get_output_path":      "src.assembly.output.get_output_path",
    "write_job_manifest":   "src.assembly.output.write_job_manifest",
}


def _run_pipeline_with_mocks(tmp_path, job=None, cfg=None, events=None, clips=None):
    """
    Run _run_pipeline with all external dependencies mocked.
    Returns dict of mock objects keyed by name.
    """
    from src.api.worker import _run_pipeline

    job = job or _make_job()
    cfg = cfg or _make_string_config(tmp_path)
    store = MagicMock()
    store.get.return_value = job
    (tmp_path / "working").mkdir(parents=True, exist_ok=True)

    mocks = {}
    with patch(_P["PlayerDetector"]) as m_pd, \
         patch(_P["GoalkeeperDetector"]) as m_gk, \
         patch(_P["PipelineRunner"]) as m_runner, \
         patch(_P["EventLog"]) as m_evlog, \
         patch(_P["compute_clips"]) as m_clips, \
         patch(_P["postprocess_clips"]) as m_post, \
         patch(_P["ReelComposer"]) as m_composer, \
         patch(_P["write_reel_to_nas"]) as m_nas, \
         patch(_P["get_output_path"], create=True) as m_out, \
         patch(_P["write_job_manifest"], create=True) as m_manifest:

        m_runner.return_value.run.return_value = 0 if not events else len(events)
        m_evlog.return_value.read_all.return_value = events or []
        m_clips.return_value = clips or []
        m_post.return_value = clips or []
        m_composer.return_value.compose.return_value = True
        m_nas.return_value = "/output/reel.mp4"

        mocks = {
            "PlayerDetector": m_pd,
            "GoalkeeperDetector": m_gk,
            "PipelineRunner": m_runner,
            "EventLog": m_evlog,
            "compute_clips": m_clips,
            "postprocess_clips": m_post,
            "ReelComposer": m_composer,
            "write_reel_to_nas": m_nas,
            "store": store,
        }

        result = _run_pipeline(job.job_id, store, cfg)
        mocks["result"] = result

    return mocks


@pytest.mark.unit
class TestRunPipelineTypeCasting:
    """Verify _run_pipeline properly casts string config values to numeric types."""

    def test_player_detector_receives_int_inference_size(self, tmp_path):
        mocks = _run_pipeline_with_mocks(tmp_path)
        pd_call = mocks["PlayerDetector"].call_args
        assert isinstance(pd_call.kwargs["inference_size"], int)
        assert pd_call.kwargs["inference_size"] == 1280

    def test_player_detector_receives_int_frame_step(self, tmp_path):
        mocks = _run_pipeline_with_mocks(tmp_path)
        pd_call = mocks["PlayerDetector"].call_args
        assert isinstance(pd_call.kwargs["frame_step"], int)
        assert pd_call.kwargs["frame_step"] == 3

    def test_player_detector_receives_bool_use_gpu(self, tmp_path):
        mocks = _run_pipeline_with_mocks(tmp_path)
        pd_call = mocks["PlayerDetector"].call_args
        assert isinstance(pd_call.kwargs["use_gpu"], bool)
        assert pd_call.kwargs["use_gpu"] is False

    def test_pipeline_runner_receives_int_chunk_sec(self, tmp_path):
        mocks = _run_pipeline_with_mocks(tmp_path)
        pr_call = mocks["PipelineRunner"].call_args
        assert isinstance(pr_call.kwargs["chunk_sec"], int)
        assert pr_call.kwargs["chunk_sec"] == 30

    def test_pipeline_runner_receives_float_overlap(self, tmp_path):
        mocks = _run_pipeline_with_mocks(tmp_path)
        pr_call = mocks["PipelineRunner"].call_args
        assert isinstance(pr_call.kwargs["overlap_sec"], float)
        assert pr_call.kwargs["overlap_sec"] == 2.0

    def test_pipeline_runner_receives_float_min_confidence(self, tmp_path):
        mocks = _run_pipeline_with_mocks(tmp_path)
        pr_call = mocks["PipelineRunner"].call_args
        assert isinstance(pr_call.kwargs["min_confidence"], float)
        assert pr_call.kwargs["min_confidence"] == 0.65

    def test_compute_clips_receives_float_padding(self, tmp_path):
        from src.detection.models import Event, EventType

        event = Event(
            job_id="test-job-001", source_file="match.mp4",
            event_type=EventType.CATCH, timestamp_start=10.0,
            timestamp_end=12.0, confidence=0.80,
            reel_targets=["goalkeeper"], frame_start=300, frame_end=360,
        )
        mocks = _run_pipeline_with_mocks(tmp_path, events=[event])
        cc_call = mocks["compute_clips"].call_args
        assert isinstance(cc_call.kwargs["pre_pad"], float)
        assert cc_call.kwargs["pre_pad"] == 3.0
        assert isinstance(cc_call.kwargs["post_pad"], float)
        assert cc_call.kwargs["post_pad"] == 5.0


@pytest.mark.unit
class TestRunPipelineReelComposer:
    """Verify ReelComposer is instantiated correctly per reel type."""

    def _make_clip(self):
        from src.segmentation.clipper import ClipBoundary
        return ClipBoundary(
            source_file="/mnt/nas/source/match.mp4",
            start_sec=7.0, end_sec=17.0,
            events=["ev-001"], reel_type="goalkeeper",
            primary_event_type="catch",
        )

    def test_composer_receives_job_id_and_reel_type(self, tmp_path):
        clip = self._make_clip()
        mocks = _run_pipeline_with_mocks(tmp_path, clips=[clip])
        composer_call = mocks["ReelComposer"].call_args
        assert composer_call.kwargs["job_id"] == "test-job-001"
        assert composer_call.kwargs["reel_type"] == "goalkeeper"

    def test_composer_receives_codec_and_int_crf(self, tmp_path):
        clip = self._make_clip()
        mocks = _run_pipeline_with_mocks(tmp_path, clips=[clip])
        composer_call = mocks["ReelComposer"].call_args
        assert composer_call.kwargs["codec"] == "copy"
        assert isinstance(composer_call.kwargs["crf"], int)
        assert composer_call.kwargs["crf"] == 18

    def test_compose_method_called(self, tmp_path):
        """Ensure worker calls composer.compose(), not the old compose_reel()."""
        clip = self._make_clip()
        mocks = _run_pipeline_with_mocks(tmp_path, clips=[clip])
        mocks["ReelComposer"].return_value.compose.assert_called_once()

    def test_no_clips_skips_composer(self, tmp_path):
        """When no clips produced, ReelComposer is not instantiated."""
        mocks = _run_pipeline_with_mocks(tmp_path, clips=[])
        mocks["ReelComposer"].assert_not_called()


@pytest.mark.unit
class TestRunPipelineUseGpuCasting:
    """Verify USE_GPU string → bool conversion handles all env var formats."""

    @pytest.mark.parametrize("value,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("Yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("random", False),
    ])
    def test_use_gpu_string_to_bool(self, value, expected, tmp_path):
        cfg = _make_string_config(tmp_path)
        cfg.USE_GPU = value
        mocks = _run_pipeline_with_mocks(tmp_path, cfg=cfg)
        pd_call = mocks["PlayerDetector"].call_args
        assert pd_call.kwargs["use_gpu"] is expected


@pytest.mark.unit
class TestWorkerModuleExports:
    """Verify module-level exports for Celery discovery."""

    def test_process_match_available(self):
        from src.api.worker import process_match
        assert process_match is not None

    def test_process_match_task_available(self):
        from src.api.worker import process_match_task
        assert process_match_task is not None

    def test_process_match_has_name(self):
        from src.api.worker import process_match
        assert hasattr(process_match, "name")
        assert process_match.name == "soccer_pipeline.tasks.process_match"

    def test_app_attribute_exposed(self):
        """Celery -A src.api.worker requires a module-level 'app' attribute."""
        from src.api import worker
        assert hasattr(worker, "app")

    def test_stub_task_delay_raises(self):
        from src.api.worker import _StubTask
        stub = _StubTask()
        with pytest.raises(RuntimeError, match="Celery is not installed"):
            stub.delay("some-job")

    def test_stub_task_call_raises(self):
        from src.api.worker import _StubTask
        stub = _StubTask()
        with pytest.raises(RuntimeError, match="Celery not installed"):
            stub("some-job")


@pytest.mark.unit
class TestRunPipelineJobCompletion:
    """Verify pipeline updates job status through all stages."""

    def _make_clip(self):
        from src.segmentation.clipper import ClipBoundary
        return ClipBoundary(
            source_file="/mnt/nas/source/match.mp4",
            start_sec=7.0, end_sec=17.0,
            events=["ev-001"], reel_type="goalkeeper",
            primary_event_type="catch",
        )

    def test_pipeline_reaches_complete_status(self, tmp_path):
        from src.ingestion.models import JobStatus
        clip = self._make_clip()
        mocks = _run_pipeline_with_mocks(tmp_path, clips=[clip])
        result = mocks["result"]
        assert result["job_id"] == "test-job-001"

        status_calls = [
            c for c in mocks["store"].update_status.call_args_list
            if c.args[1] == JobStatus.COMPLETE
        ]
        assert len(status_calls) == 1
        assert status_calls[0].kwargs["progress"] == 100.0

    def test_pipeline_transitions_through_all_stages(self, tmp_path):
        from src.ingestion.models import JobStatus
        clip = self._make_clip()
        mocks = _run_pipeline_with_mocks(tmp_path, clips=[clip])
        statuses = [c.args[1] for c in mocks["store"].update_status.call_args_list]
        assert JobStatus.DETECTING in statuses
        assert JobStatus.SEGMENTING in statuses
        assert JobStatus.ASSEMBLING in statuses
        assert JobStatus.COMPLETE in statuses

    def test_pipeline_returns_output_paths(self, tmp_path):
        from src.segmentation.clipper import ClipBoundary
        clip = ClipBoundary(
            source_file="/mnt/nas/source/match.mp4",
            start_sec=7.0, end_sec=17.0,
            events=["ev-001"], reel_type="goalkeeper",
            primary_event_type="catch",
        )
        mocks = _run_pipeline_with_mocks(tmp_path, clips=[clip])
        result = mocks["result"]
        assert "goalkeeper" in result["output_paths"]

    def test_pipeline_fails_when_no_reels_produced(self, tmp_path):
        """When detection finds 0 events, pipeline should mark FAILED, not COMPLETE."""
        from src.ingestion.models import JobStatus

        mocks = _run_pipeline_with_mocks(tmp_path, events=[], clips=[])
        result = mocks["result"]
        assert result["output_paths"] == {}

        # Should have called FAILED, not COMPLETE
        status_calls = mocks["store"].update_status.call_args_list
        final_call = status_calls[-1]
        assert final_call.args[1] == JobStatus.FAILED
        assert "No reels produced" in final_call.kwargs["error"]

        # Verify COMPLETE was never called
        complete_calls = [
            c for c in status_calls if c.args[1] == JobStatus.COMPLETE
        ]
        assert len(complete_calls) == 0
