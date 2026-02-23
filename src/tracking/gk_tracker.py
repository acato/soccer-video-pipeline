"""
GK track persistence across chunk boundaries.

The PlayerTracker resets per-chunk. This module maintains GK identity
across the full match by correlating identified GK tracks between chunks.

Supports two keepers: keeper_a (left half) and keeper_b (right half).
"""
from __future__ import annotations

from typing import Optional

import structlog

log = structlog.get_logger(__name__)


class _SingleKeeperState:
    """Tracks one keeper's identity across chunks."""

    def __init__(self, role: str):
        self.role = role
        self._confirmed_track_ids: set[int] = set()
        self._jersey_hsv: Optional[tuple[float, float, float]] = None
        self._chunk_ids: list[Optional[int]] = []

    def register(self, chunk_idx: int, track_id: Optional[int]) -> None:
        self._chunk_ids.append(track_id)
        if track_id is not None:
            self._confirmed_track_ids.add(track_id)

    @property
    def identified(self) -> bool:
        return len(self._confirmed_track_ids) > 0

    @property
    def identification_rate(self) -> float:
        if not self._chunk_ids:
            return 0.0
        return sum(1 for t in self._chunk_ids if t is not None) / len(self._chunk_ids)

    def summary(self) -> dict:
        return {
            "role": self.role,
            "total_chunks": len(self._chunk_ids),
            "identified_chunks": sum(1 for t in self._chunk_ids if t is not None),
            "identification_rate": round(self.identification_rate, 2),
            "confirmed_track_ids": list(self._confirmed_track_ids),
        }


class MatchDualGoalkeeperTracker:
    """
    Maintains two goalkeeper identities (keeper_a and keeper_b) across
    all chunks of a match.

    keeper_a = left half of the pitch (mean_x < 0.5)
    keeper_b = right half of the pitch (mean_x >= 0.5)
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._keepers = {
            "keeper_a": _SingleKeeperState("keeper_a"),
            "keeper_b": _SingleKeeperState("keeper_b"),
        }

    def register_chunk_gks(
        self, chunk_idx: int, gk_ids: dict[str, Optional[int]]
    ) -> None:
        """Record which track_ids were keepers in a given chunk."""
        for role in ("keeper_a", "keeper_b"):
            track_id = gk_ids.get(role)
            self._keepers[role].register(chunk_idx, track_id)
            if track_id is not None:
                log.debug(
                    "gk_tracker.chunk_registered",
                    chunk=chunk_idx,
                    role=role,
                    gk_track_id=track_id,
                    job_id=self.job_id,
                )

    def summary(self) -> dict:
        keeper_summaries = {
            role: state.summary() for role, state in self._keepers.items()
        }
        total_chunks = max(
            (s["total_chunks"] for s in keeper_summaries.values()), default=0
        )
        identified_either = 0
        for i in range(total_chunks):
            if any(
                i < len(self._keepers[r]._chunk_ids)
                and self._keepers[r]._chunk_ids[i] is not None
                for r in ("keeper_a", "keeper_b")
            ):
                identified_either += 1
        return {
            "job_id": self.job_id,
            "total_chunks": total_chunks,
            "identification_rate": round(
                identified_either / total_chunks if total_chunks else 0.0, 2
            ),
            "keepers": keeper_summaries,
        }


# Backward-compatible alias
class MatchGoalkeeperTracker:
    """
    Deprecated: use MatchDualGoalkeeperTracker instead.

    Wraps MatchDualGoalkeeperTracker, exposing the old single-keeper API.
    """

    def __init__(self, job_id: str):
        self._dual = MatchDualGoalkeeperTracker(job_id)
        self.job_id = job_id

    def register_chunk_gk(self, chunk_idx: int, gk_track_id: Optional[int]) -> None:
        self._dual.register_chunk_gks(
            chunk_idx, {"keeper_a": gk_track_id, "keeper_b": None}
        )

    def set_jersey_color(self, hsv: tuple[float, float, float]) -> None:
        pass  # No-op for backward compat

    @property
    def identified(self) -> bool:
        return self._dual._keepers["keeper_a"].identified

    @property
    def identification_rate(self) -> float:
        return self._dual._keepers["keeper_a"].identification_rate

    def summary(self) -> dict:
        s = self._dual.summary()
        # Flatten to old format
        ka = s["keepers"]["keeper_a"]
        return {
            "job_id": self.job_id,
            "total_chunks": ka["total_chunks"],
            "identified_chunks": ka["identified_chunks"],
            "identification_rate": ka["identification_rate"],
            "confirmed_track_ids": ka["confirmed_track_ids"],
        }
