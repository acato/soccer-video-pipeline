"""
Frame sampler: extract JPEG frames at fixed intervals using FFmpeg.

Thin pre-processing layer for the VLM-first detection pipeline.
No YOLO, no tracking -- just frame extraction.
"""
from __future__ import annotations

import subprocess
from typing import NamedTuple, Optional

import structlog

log = structlog.get_logger(__name__)


class SampledFrame(NamedTuple):
    """A single frame extracted from the source video."""
    timestamp_sec: float
    jpeg_bytes: bytes


class FrameSampler:
    """Extract JPEG frames at fixed intervals from a video file."""

    def __init__(self, video_path: str, frame_width: int = 960):
        self._video_path = video_path
        self._frame_width = frame_width

    def sample(
        self,
        duration_sec: float,
        interval_sec: float = 3.0,
        start_sec: float = 0.0,
    ) -> list[SampledFrame]:
        """Extract frames at ``interval_sec`` intervals.

        Args:
            duration_sec: Total video duration (or end time).
            interval_sec: Seconds between sampled frames.
            start_sec: Where to begin sampling.

        Returns:
            List of SampledFrame ordered by timestamp.
        """
        frames: list[SampledFrame] = []
        ts = start_sec
        while ts < duration_sec:
            jpeg = self._extract_single_frame(ts)
            if jpeg:
                frames.append(SampledFrame(timestamp_sec=ts, jpeg_bytes=jpeg))
            ts += interval_sec

        log.info(
            "frame_sampler.complete",
            total_frames=len(frames),
            interval_sec=interval_sec,
            duration_sec=duration_sec,
        )
        return frames

    def sample_range(
        self,
        center_sec: float,
        window_sec: float = 15.0,
        interval_sec: float = 1.0,
        duration_sec: float = float("inf"),
    ) -> list[SampledFrame]:
        """Extract frames in a window around ``center_sec`` (for boundary refinement).

        Args:
            center_sec: Center of the window.
            window_sec: Half-width of the window in seconds.
            interval_sec: Seconds between frames within the window.
            duration_sec: Video duration (clamp upper bound).

        Returns:
            List of SampledFrame ordered by timestamp.
        """
        start = max(0.0, center_sec - window_sec)
        end = min(duration_sec, center_sec + window_sec)

        frames: list[SampledFrame] = []
        ts = start
        while ts <= end:
            jpeg = self._extract_single_frame(ts)
            if jpeg:
                frames.append(SampledFrame(timestamp_sec=ts, jpeg_bytes=jpeg))
            ts += interval_sec

        return frames

    def _extract_single_frame(self, timestamp: float) -> Optional[bytes]:
        """Extract a single JPEG frame at the given timestamp via FFmpeg."""
        cmd = [
            "ffmpeg",
            "-ss", f"{timestamp:.3f}",
            "-i", self._video_path,
            "-vframes", "1",
            "-vf", f"scale={self._frame_width}:-1",
            "-f", "image2",
            "-c:v", "mjpeg",
            "-q:v", "5",
            "pipe:1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0 and result.stdout:
                return result.stdout
            log.warning("frame_sampler.ffmpeg_failed", ts=timestamp, rc=result.returncode)
            return None
        except subprocess.TimeoutExpired:
            log.warning("frame_sampler.ffmpeg_timeout", ts=timestamp)
            return None
