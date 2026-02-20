"""Unit tests for src/assembly/output.py"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.assembly.output import get_output_path, write_job_manifest, REEL_FILENAME_MAP


@pytest.mark.unit
class TestGetOutputPath:
    def test_goalkeeper_path(self):
        p = get_output_path("/nas/out", "job-001", "goalkeeper")
        assert str(p) == "/nas/out/job-001/goalkeeper_reel.mp4"

    def test_highlights_path(self):
        p = get_output_path("/nas/out", "job-001", "highlights")
        assert str(p) == "/nas/out/job-001/highlights_reel.mp4"

    def test_player_path(self):
        p = get_output_path("/nas/out", "job-001", "player")
        assert str(p) == "/nas/out/job-001/player_reel.mp4"

    def test_unknown_reel_type_uses_generic_name(self):
        p = get_output_path("/nas/out", "job-001", "custom_reel")
        assert "custom_reel" in str(p)

    def test_job_id_in_path(self):
        p = get_output_path("/out", "abc-123-def", "goalkeeper")
        assert "abc-123-def" in str(p)

    def test_returns_path_object(self):
        p = get_output_path("/out", "j1", "goalkeeper")
        assert isinstance(p, Path)

    def test_all_reel_types_have_mappings(self):
        for reel_type in ["goalkeeper", "highlights", "player"]:
            assert reel_type in REEL_FILENAME_MAP


@pytest.mark.unit
class TestWriteJobManifest:
    def test_manifest_written_to_correct_path(self, tmp_path):
        output_paths = {"goalkeeper": "/out/job1/goalkeeper_reel.mp4"}
        manifest_path = write_job_manifest(
            str(tmp_path), "job-001", output_paths,
            metadata={"source": "match.mp4", "duration_sec": 5400.0},
        )
        assert Path(manifest_path).exists()
        assert manifest_path.endswith("manifest.json")

    def test_manifest_contains_job_id(self, tmp_path):
        write_job_manifest(str(tmp_path), "job-999", {}, {})
        manifest_path = tmp_path / "job-999" / "manifest.json"
        data = json.loads(manifest_path.read_text())
        assert data["job_id"] == "job-999"

    def test_manifest_contains_reels(self, tmp_path):
        paths = {"goalkeeper": "/reel.mp4"}
        write_job_manifest(str(tmp_path), "j1", paths, {})
        data = json.loads((tmp_path / "j1" / "manifest.json").read_text())
        assert data["reels"]["goalkeeper"] == "/reel.mp4"

    def test_manifest_contains_completed_at(self, tmp_path):
        write_job_manifest(str(tmp_path), "j1", {}, {})
        data = json.loads((tmp_path / "j1" / "manifest.json").read_text())
        assert "completed_at" in data

    def test_manifest_directory_created(self, tmp_path):
        nested = tmp_path / "deep" / "path"
        write_job_manifest(str(nested), "j1", {}, {})
        assert (nested / "j1" / "manifest.json").exists()
