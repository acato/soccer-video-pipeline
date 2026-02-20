"""
Reel composer: orchestrates clip extraction → concatenation → output for one reel type.

Responsibilities:
  - Extract individual clips from source video
  - Write clips to working dir (local SSD)
  - Concatenate clips into final reel
  - Return path to assembled reel
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import structlog

from src.assembly.encoder import concat_clips, extract_clip, validate_clip
from src.segmentation.clipper import ClipBoundary, clips_stats

log = structlog.get_logger(__name__)


class ReelComposer:
    """
    Composes a single output reel from a list of ClipBoundary objects.

    All intermediate clips are written to working_dir and cleaned up after
    the final reel is assembled. The final reel is written atomically.
    """

    def __init__(
        self,
        job_id: str,
        reel_type: str,
        working_dir: str | Path,
        codec: str = "copy",
        crf: int = 18,
    ):
        self.job_id = job_id
        self.reel_type = reel_type
        self.working_dir = Path(working_dir)
        self.codec = codec
        self.crf = crf
        self._clips_dir = self.working_dir / job_id / "clips" / reel_type
        self._clips_dir.mkdir(parents=True, exist_ok=True)

    def compose(
        self,
        clips: list[ClipBoundary],
        output_path: str | Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """
        Extract all clips and concatenate into the output reel.

        Args:
            clips: Ordered list of clip boundaries (sorted by start_sec)
            output_path: Final MP4 destination path
            progress_callback: Called with progress 0–100 during extraction

        Returns:
            True on success, False on failure.
        """
        if not clips:
            log.warning("composer.no_clips", job_id=self.job_id, reel_type=self.reel_type)
            return False

        stats = clips_stats(clips)
        log.info(
            "composer.start",
            job_id=self.job_id,
            reel_type=self.reel_type,
            clip_count=stats["count"],
            total_duration_sec=stats["total_duration_sec"],
        )

        t0 = time.monotonic()
        extracted_paths: list[str] = []
        failed = 0

        for idx, clip in enumerate(clips):
            clip_output = str(self._clips_dir / f"clip_{idx:04d}.mp4")
            ok = extract_clip(
                source_path=clip.source_file,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                output_path=clip_output,
                codec=self.codec,
                crf=self.crf,
            )
            if ok and validate_clip(clip_output):
                extracted_paths.append(clip_output)
            else:
                failed += 1
                log.warning(
                    "composer.clip_failed",
                    clip_idx=idx,
                    start=clip.start_sec,
                    end=clip.end_sec,
                    primary_event=clip.primary_event_type,
                )

            if progress_callback:
                progress_callback((idx + 1) / len(clips) * 90)  # Reserve 10% for concat

        if not extracted_paths:
            log.error("composer.all_clips_failed", job_id=self.job_id)
            return False

        if failed > 0:
            log.warning(
                "composer.some_clips_failed",
                failed=failed,
                succeeded=len(extracted_paths),
            )

        # Sort extracted clips by their numeric index (preserves temporal order)
        extracted_paths.sort()

        log.info("composer.concatenating", clip_count=len(extracted_paths))
        success = concat_clips(
            clip_paths=extracted_paths,
            output_path=str(output_path),
        )

        if progress_callback:
            progress_callback(100.0)

        elapsed = time.monotonic() - t0
        if success:
            log.info(
                "composer.complete",
                job_id=self.job_id,
                reel_type=self.reel_type,
                output=str(output_path),
                elapsed_sec=round(elapsed, 1),
            )
        else:
            log.error("composer.concat_failed", job_id=self.job_id, reel_type=self.reel_type)

        return success

    def cleanup_working_clips(self) -> int:
        """Delete intermediate clip files from working dir. Returns count deleted."""
        deleted = 0
        for f in self._clips_dir.glob("clip_*.mp4"):
            f.unlink(missing_ok=True)
            deleted += 1
        log.debug("composer.cleanup", deleted=deleted, dir=str(self._clips_dir))
        return deleted
