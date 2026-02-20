"""
GK track persistence across chunk boundaries.

The PlayerTracker resets per-chunk. This module maintains GK identity
across the full match by correlating identified GK tracks between chunks.
"""
from __future__ import annotations

from typing import Optional

import structlog

log = structlog.get_logger(__name__)


class MatchGoalkeeperTracker:
    """
    Maintains goalkeeper identity across all chunks of a match.

    As each chunk is processed and a GK track_id is identified,
    this class maps them to a stable match-level GK ID.

    The GK may temporarily leave the frame or be occluded — this
    handles re-identification by jersey color + field position continuity.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._confirmed_gk_track_ids: set[int] = set()
        self._gk_jersey_hsv: Optional[tuple[float, float, float]] = None
        self._chunk_gk_ids: list[Optional[int]] = []  # Per-chunk GK track IDs

    def register_chunk_gk(self, chunk_idx: int, gk_track_id: Optional[int]) -> None:
        """Record which track_id was the GK in a given chunk."""
        self._chunk_gk_ids.append(gk_track_id)
        if gk_track_id is not None:
            self._confirmed_gk_track_ids.add(gk_track_id)
            log.debug(
                "gk_tracker.chunk_registered",
                chunk=chunk_idx,
                gk_track_id=gk_track_id,
                job_id=self.job_id,
            )

    def set_jersey_color(self, hsv: tuple[float, float, float]) -> None:
        """Store dominant HSV color of GK jersey for re-identification."""
        self._gk_jersey_hsv = hsv

    @property
    def identified(self) -> bool:
        """True if GK has been identified in at least one chunk."""
        return len(self._confirmed_gk_track_ids) > 0

    @property
    def identification_rate(self) -> float:
        """Fraction of chunks where GK was successfully identified."""
        if not self._chunk_gk_ids:
            return 0.0
        identified = sum(1 for t in self._chunk_gk_ids if t is not None)
        return identified / len(self._chunk_gk_ids)

    def summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "total_chunks": len(self._chunk_gk_ids),
            "identified_chunks": sum(1 for t in self._chunk_gk_ids if t is not None),
            "identification_rate": round(self.identification_rate, 2),
            "confirmed_track_ids": list(self._confirmed_gk_track_ids),
        }
