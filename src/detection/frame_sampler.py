"""
Frame sampler: extract JPEG frames at fixed intervals using FFmpeg.

Thin pre-processing layer for the VLM-first detection pipeline.
No YOLO, no tracking -- just frame extraction.

Two extraction modes:
  - Single-frame: one FFmpeg call per frame (slow, used for small ranges)
  - Batch: single FFmpeg pass with fps filter + JPEG pipe (fast, for full-game scans)
"""
from __future__ import annotations

import subprocess
from typing import NamedTuple, Optional

import structlog

log = structlog.get_logger(__name__)

# JPEG markers for splitting pipe output
_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


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

        Uses batch extraction (single FFmpeg pass) for efficiency.
        Falls back to per-frame extraction if batch fails.

        Args:
            duration_sec: Total video duration (or end time).
            interval_sec: Seconds between sampled frames.
            start_sec: Where to begin sampling.

        Returns:
            List of SampledFrame ordered by timestamp.
        """
        expected_count = int((duration_sec - start_sec) / interval_sec)

        # Try batch extraction first (for large ranges, much faster)
        if expected_count > 10:
            jpegs = self._extract_batch(
                start_sec=start_sec,
                end_sec=duration_sec,
                fps=1.0 / interval_sec,
            )
            if jpegs and len(jpegs) >= expected_count * 0.8:
                result = []
                for i, jpeg_bytes in enumerate(jpegs):
                    ts = start_sec + i * interval_sec
                    if ts >= duration_sec:
                        break
                    result.append(SampledFrame(timestamp_sec=ts, jpeg_bytes=jpeg_bytes))
                log.info(
                    "frame_sampler.batch_complete",
                    total_frames=len(result),
                    interval_sec=interval_sec,
                    duration_sec=duration_sec,
                )
                return result
            log.warning("frame_sampler.batch_insufficient",
                        got=len(jpegs) if jpegs else 0, expected=expected_count)

        # Per-frame extraction (used for small ranges or batch fallback)
        return self._sample_sequential(duration_sec, interval_sec, start_sec)

    def _sample_sequential(
        self,
        duration_sec: float,
        interval_sec: float,
        start_sec: float,
    ) -> list[SampledFrame]:
        """Fallback: extract frames one at a time."""
        frames: list[SampledFrame] = []
        ts = start_sec
        while ts < duration_sec:
            jpeg = self._extract_single_frame(ts)
            if jpeg:
                frames.append(SampledFrame(timestamp_sec=ts, jpeg_bytes=jpeg))
            ts += interval_sec
        log.info(
            "frame_sampler.sequential_complete",
            total_frames=len(frames),
            interval_sec=interval_sec,
        )
        return frames

    def sample_range(
        self,
        center_sec: float,
        window_sec: float = 15.0,
        interval_sec: float = 1.0,
        duration_sec: float = float("inf"),
    ) -> list[SampledFrame]:
        """Extract frames in a window around ``center_sec``.

        For small ranges (< 60s), uses batch extraction.

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

        expected_count = int((end - start) / interval_sec) + 1

        # Use batch for larger ranges
        if expected_count > 10:
            jpegs = self._extract_batch(
                start_sec=start,
                end_sec=end,
                fps=1.0 / interval_sec,
            )
            if jpegs and len(jpegs) >= expected_count * 0.8:
                result = []
                for i, jpeg_bytes in enumerate(jpegs):
                    ts = start + i * interval_sec
                    if ts > end:
                        break
                    result.append(SampledFrame(timestamp_sec=ts, jpeg_bytes=jpeg_bytes))
                return result

        # Sequential fallback (small ranges or batch failure)
        frames: list[SampledFrame] = []
        ts = start
        while ts <= end:
            jpeg = self._extract_single_frame(ts)
            if jpeg:
                frames.append(SampledFrame(timestamp_sec=ts, jpeg_bytes=jpeg))
            ts += interval_sec
        return frames

    def _extract_batch(
        self,
        start_sec: float,
        end_sec: float,
        fps: float = 1.0,
    ) -> list[bytes]:
        """Extract frames via a single FFmpeg pass using fps filter + pipe.

        Returns a list of JPEG byte blobs, or empty list on failure.
        """
        duration = end_sec - start_sec
        if duration <= 0:
            return []

        # Expected frame count (generous estimate for timeout)
        expected = int(duration * fps) + 10
        timeout = max(30, int(duration * 0.5))  # ~0.5s processing per second of video

        cmd = [
            "ffmpeg",
            "-ss", f"{start_sec:.3f}",
            "-i", self._video_path,
            "-t", f"{duration:.3f}",
            "-vf", f"fps={fps},scale={self._frame_width}:-1",
            "-f", "image2pipe",
            "-c:v", "mjpeg",
            "-q:v", "5",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
            )
            if result.returncode != 0 or not result.stdout:
                stderr = result.stderr.decode("utf-8", errors="replace")[-200:]
                log.warning("frame_sampler.batch_ffmpeg_failed",
                            rc=result.returncode, stderr=stderr,
                            start=start_sec, end=end_sec)
                return []

            # Split JPEG stream by SOI/EOI markers
            jpegs = self._split_jpeg_stream(result.stdout)
            log.info("frame_sampler.batch_extracted",
                     frames=len(jpegs), start=start_sec, end=end_sec,
                     bytes_total=len(result.stdout))
            return jpegs

        except subprocess.TimeoutExpired:
            log.warning("frame_sampler.batch_timeout",
                        timeout=timeout, start=start_sec, end=end_sec)
            return []

    @staticmethod
    def _split_jpeg_stream(data: bytes) -> list[bytes]:
        """Split a concatenated JPEG stream into individual images.

        Scans for JPEG SOI (0xFFD8) markers to find frame boundaries.
        """
        frames = []
        pos = 0
        length = len(data)

        while pos < length:
            # Find next SOI marker
            soi = data.find(_JPEG_SOI, pos)
            if soi == -1:
                break

            # Find next SOI after this one (marks end of current frame)
            next_soi = data.find(_JPEG_SOI, soi + 2)
            if next_soi == -1:
                # Last frame — take everything to end
                frame = data[soi:]
            else:
                frame = data[soi:next_soi]

            if len(frame) > 100:  # Skip tiny fragments
                frames.append(frame)

            pos = next_soi if next_soi != -1 else length

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
