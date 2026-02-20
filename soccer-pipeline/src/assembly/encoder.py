"""
FFmpeg-based clip extraction and reel assembly.

Design principles:
  - Stream copy (no re-encode) is always attempted first — fastest and lossless
  - Re-encode fallback used when stream copy fails (e.g., non-keyframe boundaries)
  - Never loads video frames into Python memory
  - All operations are atomic (write to .tmp, rename to final)
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def extract_clip(
    source_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    codec: str = "copy",
    crf: int = 18,
    audio_codec: str = "copy",
) -> bool:
    """
    Extract a single clip from source video using FFmpeg.

    Attempts stream copy first. Falls back to re-encode if stream copy fails
    (common when start_sec doesn't land on a keyframe).

    Args:
        source_path: Full path to source MP4
        start_sec: Clip start time (seconds from video start)
        end_sec: Clip end time
        output_path: Destination MP4 path
        codec: Video codec ('copy' or 'libx264' or 'libx265')
        crf: Quality for re-encode (lower = better, 18 is near-lossless)
        audio_codec: Audio codec ('copy' or 'aac')

    Returns:
        True on success, False on failure.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        log.warning("encoder.invalid_duration", start=start_sec, end=end_sec)
        return False

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(output_path_obj.with_suffix(".tmp.mp4"))

    # First attempt: stream copy (fast, lossless, may have imprecise boundaries)
    success = _run_ffmpeg_extract(
        source_path, start_sec, duration, tmp_path,
        video_codec="copy", audio_codec=audio_codec,
    )

    if not success and codec != "copy":
        log.info("encoder.stream_copy_failed_reencode", output=output_path)
        success = _run_ffmpeg_extract(
            source_path, start_sec, duration, tmp_path,
            video_codec=codec, audio_codec="aac",
            extra_args=["-crf", str(crf), "-preset", "fast"],
        )

    if success and Path(tmp_path).exists():
        os.replace(tmp_path, output_path)
        size_mb = Path(output_path).stat().st_size / 1024 / 1024
        log.info(
            "encoder.clip_extracted",
            output=output_path,
            start_sec=start_sec,
            end_sec=end_sec,
            size_mb=round(size_mb, 1),
        )
        return True

    # Cleanup temp
    Path(tmp_path).unlink(missing_ok=True)
    log.error("encoder.clip_extraction_failed", source=source_path, start=start_sec, end=end_sec)
    return False


def concat_clips(
    clip_paths: list[str],
    output_path: str,
    add_timestamps: bool = False,
    source_fps: float = 30.0,
) -> bool:
    """
    Concatenate multiple MP4 clips into a single reel using FFmpeg concat demuxer.

    All clips must have the same codec, resolution, and frame rate.
    (Guaranteed if all were extracted from the same source with stream copy.)

    Args:
        clip_paths: Ordered list of clip paths to concatenate
        output_path: Final output MP4 path
        add_timestamps: If True, burn match timestamp into each clip
        source_fps: Used for timestamp calculation if add_timestamps=True

    Returns:
        True on success, False on failure.
    """
    if not clip_paths:
        log.warning("encoder.concat_empty_list")
        return False

    valid_clips = [p for p in clip_paths if Path(p).exists() and Path(p).stat().st_size > 0]
    if not valid_clips:
        log.error("encoder.no_valid_clips", attempted=len(clip_paths))
        return False

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = str(output_path_obj.with_suffix(".tmp.mp4"))

    # Write concat list file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    ) as concat_file:
        concat_file_path = concat_file.name
        for clip in valid_clips:
            # Escape single quotes in paths
            safe_path = str(clip).replace("'", "'\\''")
            concat_file.write(f"file '{safe_path}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-c", "copy",             # Stream copy — all clips must match
            "-movflags", "+faststart", # Web-optimized MP4
            tmp_output,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)

        if result.returncode != 0:
            log.error(
                "encoder.concat_failed",
                error=result.stderr.decode()[:500],
                clip_count=len(valid_clips),
            )
            return False

        os.replace(tmp_output, output_path)
        try:
            size_mb = Path(output_path).stat().st_size / 1024 / 1024
        except OSError:
            size_mb = 0.0
        log.info(
            "encoder.reel_assembled",
            output=output_path,
            clips=len(valid_clips),
            size_mb=round(size_mb, 1),
        )
        return True

    finally:
        Path(concat_file_path).unlink(missing_ok=True)
        Path(tmp_output).unlink(missing_ok=True)


def validate_clip(path: str) -> bool:
    """
    Quick FFprobe check that a clip is a valid, playable MP4.
    Returns True if valid, False otherwise.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height",
        "-of", "csv=p=0",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    return result.returncode == 0 and bool(result.stdout.strip())


# ── Internal helpers ───────────────────────────────────────────────────────

def _run_ffmpeg_extract(
    source: str,
    start: float,
    duration: float,
    output: str,
    video_codec: str = "copy",
    audio_codec: str = "copy",
    extra_args: list[str] = None,
) -> bool:
    """Run FFmpeg to extract a clip segment. Returns True on success.

    Seek strategy:
    - Stream copy: input-side seek (fast, keyframe-aligned) then output -t for
      duration. Start may be ~0.5s early due to keyframe snap — acceptable for
      sports reels. End is precisely at requested duration from seek point.
    - Re-encode: output-side seek (slower, frame-accurate).
    """
    if video_codec == "copy":
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",     # Fast input seek (snaps to prior keyframe)
            "-i", source,
            "-t", f"{duration:.3f}",   # Duration from seek point
            "-c:v", video_codec,
            "-c:a", audio_codec,
            "-avoid_negative_ts", "make_zero",
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", source,
            "-ss", f"{start:.3f}",     # Accurate output-side seek for re-encode
            "-t", f"{duration:.3f}",
            "-c:v", video_codec,
            "-c:a", audio_codec,
        ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["-movflags", "+faststart", output])

    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        log.debug(
            "encoder.ffmpeg_failed",
            codec=video_codec,
            error=result.stderr.decode()[:300],
        )
        return False
    return True
