"""Unit tests for pipeline_cli.py.

Uses httpx.MockTransport to simulate API responses — no running server needed.
"""
import json
from unittest.mock import patch

import httpx
import pytest

# Import from infra/scripts — add to path since it's not a package.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra" / "scripts"))
from pipeline_cli import main, _progress_bar, build_parser  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: canned API responses
# ---------------------------------------------------------------------------

SAMPLE_JOB = {
    "job_id": "abc-123",
    "status": "DETECTING",
    "progress_pct": 42.0,
    "error": None,
    "video_file": {
        "filename": "game_2025_01_15.mp4",
        "duration_sec": 5400.0,
        "width": 3840,
        "height": 2160,
        "sha256": "deadbeef",
    },
    "reel_types": ["keeper", "highlights"],
    "output_paths": {},
    "created_at": "2025-06-01T10:00:00",
    "updated_at": "2025-06-01T10:05:00",
}

COMPLETE_JOB = {
    **SAMPLE_JOB,
    "status": "COMPLETE",
    "progress_pct": 100.0,
    "output_paths": {
        "keeper": "/output/abc-123/keeper_reel.mp4",
        "highlights": "/output/abc-123/highlights_reel.mp4",
    },
}

FAILED_JOB = {
    **SAMPLE_JOB,
    "status": "FAILED",
    "progress_pct": 30.0,
    "error": "Detection crashed: out of memory",
}


def _mock_transport(handler):
    """Build an httpx.MockTransport from a request handler function."""
    return httpx.MockTransport(handler)


def _patch_client(handler):
    """Patch pipeline_cli._client to return a client using MockTransport."""
    def factory(api_url, timeout):
        return httpx.Client(
            transport=_mock_transport(handler),
            base_url=api_url,
            timeout=timeout,
        )
    return patch("pipeline_cli._client", side_effect=factory)


# ---------------------------------------------------------------------------
# TestSubmitCommand
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestSubmitCommand:
    def test_submit_success(self, capsys):
        def handler(req):
            assert req.method == "POST"
            assert req.url.path == "/jobs"
            body = json.loads(req.content)
            assert body["nas_path"] == "matches/game.mp4"
            return httpx.Response(201, json=SAMPLE_JOB)

        with _patch_client(handler):
            code = main(["submit", "matches/game.mp4"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Job submitted successfully" in out
        assert "abc-123" in out

    def test_submit_json_mode(self, capsys):
        def handler(req):
            return httpx.Response(201, json=SAMPLE_JOB)

        with _patch_client(handler):
            code = main(["--json", "submit", "matches/game.mp4"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["job_id"] == "abc-123"

    def test_submit_custom_reels(self, capsys):
        def handler(req):
            body = json.loads(req.content)
            assert body["reel_types"] == ["keeper"]
            return httpx.Response(201, json=SAMPLE_JOB)

        with _patch_client(handler):
            code = main(["submit", "game.mp4", "--reel", "keeper"])
        assert code == 0

    def test_submit_invalid_reel(self, capsys):
        with _patch_client(lambda r: httpx.Response(200, json={})):
            code = main(["submit", "game.mp4", "--reel", "badtype"])
        assert code == 1
        err = capsys.readouterr().err
        assert "Invalid reel type" in err

    def test_submit_404(self, capsys):
        def handler(req):
            return httpx.Response(404, json={"detail": "Video not found on NAS"})

        with _patch_client(handler):
            code = main(["submit", "missing.mp4"])
        assert code == 1
        err = capsys.readouterr().err
        assert "Not found" in err

    def test_submit_server_error(self, capsys):
        def handler(req):
            return httpx.Response(500, json={"detail": "Internal error"})

        with _patch_client(handler):
            code = main(["submit", "game.mp4"])
        assert code == 1
        err = capsys.readouterr().err
        assert "500" in err


# ---------------------------------------------------------------------------
# TestStatusCommand
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestStatusCommand:
    def test_status_success_human(self, capsys):
        def handler(req):
            assert req.url.path == "/jobs/abc-123"
            return httpx.Response(200, json=SAMPLE_JOB)

        with _patch_client(handler):
            code = main(["status", "abc-123"])
        assert code == 0
        out = capsys.readouterr().out
        assert "DETECTING" in out
        assert "abc-123" in out

    def test_status_json(self, capsys):
        def handler(req):
            return httpx.Response(200, json=SAMPLE_JOB)

        with _patch_client(handler):
            code = main(["--json", "status", "abc-123"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "DETECTING"

    def test_status_404(self, capsys):
        def handler(req):
            return httpx.Response(404, json={"detail": "Job not found"})

        with _patch_client(handler):
            code = main(["status", "nonexistent"])
        assert code == 1

    def test_status_shows_progress_bar(self, capsys):
        def handler(req):
            return httpx.Response(200, json=SAMPLE_JOB)

        with _patch_client(handler):
            code = main(["status", "abc-123"])
        assert code == 0
        out = capsys.readouterr().out
        assert "█" in out
        assert "42%" in out

    def test_status_failed_shows_error(self, capsys):
        def handler(req):
            return httpx.Response(200, json=FAILED_JOB)

        with _patch_client(handler):
            code = main(["status", "abc-123"])
        assert code == 0
        out = capsys.readouterr().out
        assert "out of memory" in out


# ---------------------------------------------------------------------------
# TestListCommand
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestListCommand:
    def test_list_success_human(self, capsys):
        def handler(req):
            assert req.url.path == "/jobs"
            assert req.url.params.get("limit") == "20"
            return httpx.Response(200, json=[SAMPLE_JOB, COMPLETE_JOB])

        with _patch_client(handler):
            code = main(["list"])
        assert code == 0
        out = capsys.readouterr().out
        assert "abc-123" in out
        assert "DETECTING" in out
        assert "COMPLETE" in out

    def test_list_json(self, capsys):
        def handler(req):
            return httpx.Response(200, json=[SAMPLE_JOB])

        with _patch_client(handler):
            code = main(["--json", "list"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 1

    def test_list_empty(self, capsys):
        def handler(req):
            return httpx.Response(200, json=[])

        with _patch_client(handler):
            code = main(["list"])
        assert code == 0
        out = capsys.readouterr().out
        assert "No jobs found" in out

    def test_list_custom_limit(self, capsys):
        def handler(req):
            assert req.url.params.get("limit") == "5"
            return httpx.Response(200, json=[])

        with _patch_client(handler):
            code = main(["list", "--limit", "5"])
        assert code == 0


# ---------------------------------------------------------------------------
# TestRetryCommand
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestRetryCommand:
    def test_retry_success(self, capsys):
        retried = {**FAILED_JOB, "status": "PENDING", "progress_pct": 0, "error": None}

        def handler(req):
            assert req.method == "POST"
            assert req.url.path == "/jobs/abc-123/retry"
            return httpx.Response(200, json=retried)

        with _patch_client(handler):
            code = main(["retry", "abc-123"])
        assert code == 0
        out = capsys.readouterr().out
        assert "re-queued" in out

    def test_retry_not_failed(self, capsys):
        def handler(req):
            return httpx.Response(400, json={"detail": "Job must be FAILED to retry"})

        with _patch_client(handler):
            code = main(["retry", "abc-123"])
        assert code == 1
        err = capsys.readouterr().err
        assert "Bad request" in err

    def test_retry_not_found(self, capsys):
        def handler(req):
            return httpx.Response(404, json={"detail": "Job not found"})

        with _patch_client(handler):
            code = main(["retry", "nonexistent"])
        assert code == 1


# ---------------------------------------------------------------------------
# TestConnectionErrors
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestConnectionErrors:
    def test_server_not_running(self, capsys):
        def handler(req):
            raise httpx.ConnectError("Connection refused")

        with _patch_client(handler):
            code = main(["list"])
        assert code == 1
        err = capsys.readouterr().err
        assert "Cannot connect" in err

    def test_timeout(self, capsys):
        def handler(req):
            raise httpx.ReadTimeout("timed out")

        with _patch_client(handler):
            code = main(["--timeout", "1", "list"])
        assert code == 1
        err = capsys.readouterr().err
        assert "timed out" in err

    def test_custom_api_url(self, capsys):
        def handler(req):
            return httpx.Response(200, json=[])

        with _patch_client(handler) as mock:
            code = main(["--api-url", "http://myhost:9090", "list"])
            assert code == 0
            call_args = mock.call_args
            assert call_args[0][0] == "http://myhost:9090"


# ---------------------------------------------------------------------------
# TestProgressBar
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestProgressBar:
    def test_zero_percent(self):
        bar = _progress_bar(0)
        assert bar.startswith("[░")
        assert "0%" in bar

    def test_fifty_percent(self):
        bar = _progress_bar(50)
        assert "█" in bar
        assert "░" in bar
        assert "50%" in bar

    def test_hundred_percent(self):
        bar = _progress_bar(100)
        assert "░" not in bar
        assert "100%" in bar


# ---------------------------------------------------------------------------
# TestArgParsing
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestArgParsing:
    def test_no_command_shows_help(self, capsys):
        code = main([])
        assert code == 1

    def test_submit_requires_path(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["submit"])

    def test_status_requires_job_id(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["status"])

    def test_retry_requires_job_id(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["retry"])

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.limit == 20
        assert args.json is False
        assert args.timeout == 10.0

    def test_reel_type_parsing(self):
        parser = build_parser()
        args = parser.parse_args(["submit", "game.mp4", "--reel", "keeper,highlights"])
        assert args.reel == "keeper,highlights"

    def test_api_url_from_env(self, monkeypatch):
        monkeypatch.setenv("PIPELINE_API_URL", "http://custom:1234")
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.api_url == "http://custom:1234"
