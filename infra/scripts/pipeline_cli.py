#!/usr/bin/env python3
"""CLI for the Soccer Video Processing Pipeline.

Standalone script — no src/ imports. Communicates with the API via HTTP only.

Usage:
    pipeline_cli.py submit <nas_path> [--reel goalkeeper,highlights] [--json]
    pipeline_cli.py status <job_id>   [--watch] [--interval 5]      [--json]
    pipeline_cli.py list              [--limit 20]                   [--json]
    pipeline_cli.py retry  <job_id>                                  [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

DEFAULT_API_URL = "http://localhost:8080"
VALID_REEL_TYPES = {"goalkeeper", "highlights", "player"}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _progress_bar(pct: float, width: int = 30) -> str:
    """Render a text progress bar like [████████░░░░░░░░░░░░] 40%."""
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:.0f}%"


def _print_job(job: dict) -> None:
    """Pretty-print a single job record."""
    print(f"Job:      {job['job_id']}")
    print(f"Status:   {job['status']}")
    print(f"Progress: {_progress_bar(job.get('progress_pct', 0))}")
    if job.get("video_file"):
        vf = job["video_file"]
        print(f"File:     {vf.get('filename', 'N/A')}")
        dur = vf.get("duration_sec")
        if dur is not None:
            mins, secs = divmod(int(dur), 60)
            print(f"Duration: {mins}m{secs:02d}s")
    if job.get("reel_types"):
        print(f"Reels:    {', '.join(job['reel_types'])}")
    if job.get("error"):
        print(f"Error:    {job['error']}")
    if job.get("output_paths"):
        print("Outputs:")
        for reel, path in job["output_paths"].items():
            print(f"  {reel}: {path}")


def _print_job_table(jobs: list[dict]) -> None:
    """Print a compact table of jobs."""
    if not jobs:
        print("No jobs found.")
        return
    # Header
    print(f"{'JOB ID':<40} {'STATUS':<12} {'PROGRESS':<15} {'FILE'}")
    print("-" * 90)
    for j in jobs:
        job_id = j.get("job_id", "")[:38]
        status = j.get("status", "")
        pct = j.get("progress_pct", 0)
        filename = ""
        if j.get("video_file"):
            filename = j["video_file"].get("filename", "")
        bar = _progress_bar(pct, width=10)
        print(f"{job_id:<40} {status:<12} {bar:<15} {filename}")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _client(api_url: str, timeout: float) -> httpx.Client:
    return httpx.Client(base_url=api_url, timeout=timeout)


def _handle_response(resp: httpx.Response, json_mode: bool) -> dict | list:
    """Handle HTTP response, raising friendly errors for common failures."""
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        if resp.status_code == 404:
            _print_err(f"Error: Not found — {detail}")
        elif resp.status_code == 400:
            _print_err(f"Error: Bad request — {detail}")
        elif resp.status_code == 409:
            _print_err(f"Error: Conflict — {detail}")
        else:
            _print_err(f"Error: Server returned {resp.status_code} — {detail}")
        raise SystemExit(1)
    return resp.json()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_submit(client: httpx.Client, args: argparse.Namespace) -> int:
    body: dict = {"nas_path": args.nas_path}
    if args.reel:
        reel_types = [r.strip() for r in args.reel.split(",")]
        for rt in reel_types:
            if rt not in VALID_REEL_TYPES:
                _print_err(
                    f"Error: Invalid reel type '{rt}'. "
                    f"Valid types: {', '.join(sorted(VALID_REEL_TYPES))}"
                )
                return 1
        body["reel_types"] = reel_types
    resp = client.post("/jobs", json=body)
    data = _handle_response(resp, args.json)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print("Job submitted successfully.")
        _print_job(data)
    return 0


def cmd_status(client: httpx.Client, args: argparse.Namespace) -> int:
    def _fetch():
        resp = client.get(f"/jobs/{args.job_id}")
        return _handle_response(resp, args.json)

    if args.watch:
        try:
            while True:
                data = _fetch()
                if args.json:
                    print(json.dumps(data, indent=2))
                else:
                    # Clear previous output (simple approach for terminals)
                    print(f"\rStatus: {data['status']}  "
                          f"{_progress_bar(data.get('progress_pct', 0))}",
                          end="", flush=True)
                if data.get("status") in ("COMPLETE", "FAILED"):
                    if not args.json:
                        print()  # newline after progress
                        _print_job(data)
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            return 0
    else:
        data = _fetch()
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            _print_job(data)
    return 0


def cmd_list(client: httpx.Client, args: argparse.Namespace) -> int:
    resp = client.get("/jobs", params={"limit": args.limit})
    data = _handle_response(resp, args.json)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        _print_job_table(data)
    return 0


def cmd_retry(client: httpx.Client, args: argparse.Namespace) -> int:
    resp = client.post(f"/jobs/{args.job_id}/retry")
    data = _handle_response(resp, args.json)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"Job {args.job_id} re-queued.")
        print(f"Status: {data.get('status', 'PENDING')}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline_cli",
        description="Soccer Video Processing Pipeline CLI",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("PIPELINE_API_URL", DEFAULT_API_URL),
        help=f"API base URL (default: {DEFAULT_API_URL}, or $PIPELINE_API_URL)",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Output raw JSON (for scripting)",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0,
        help="HTTP request timeout in seconds (default: 10)",
    )

    sub = parser.add_subparsers(dest="command")

    # submit
    p_submit = sub.add_parser("submit", help="Submit a video for processing")
    p_submit.add_argument("nas_path", help="Relative path to video on NAS")
    p_submit.add_argument(
        "--reel", default=None,
        help="Comma-separated reel types (default: goalkeeper,highlights)",
    )

    # status
    p_status = sub.add_parser("status", help="Check job status")
    p_status.add_argument("job_id", help="Job ID")
    p_status.add_argument(
        "--watch", action="store_true", default=False,
        help="Poll until job completes",
    )
    p_status.add_argument(
        "--interval", type=int, default=5,
        help="Poll interval in seconds (default: 5)",
    )

    # list
    p_list = sub.add_parser("list", help="List all jobs")
    p_list.add_argument(
        "--limit", type=int, default=20,
        help="Max jobs to return (default: 20)",
    )

    # retry
    p_retry = sub.add_parser("retry", help="Retry a failed job")
    p_retry.add_argument("job_id", help="Job ID")

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "submit": cmd_submit,
    "status": cmd_status,
    "list": cmd_list,
    "retry": cmd_retry,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    handler = COMMANDS[args.command]
    try:
        client = _client(args.api_url, args.timeout)
        return handler(client, args)
    except httpx.ConnectError:
        _print_err(f"Error: Cannot connect to API at {args.api_url}")
        _print_err("Is the pipeline running? Try: make up")
        return 1
    except httpx.TimeoutException:
        _print_err(f"Error: Request timed out after {args.timeout}s")
        return 1
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    raise SystemExit(main())
