"""
Base detector abstract class. All event detectors implement this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np

from src.detection.models import Detection, Event, Track


class BaseDetector(ABC):
    """
    Abstract base for all event detectors.

    Detection pipeline per chunk:
      1. detect_frame() called on each decoded frame
      2. finalize_chunk() called at end of chunk with all accumulated tracks
      3. Returns list of Events found in that chunk
    """

    def __init__(self, job_id: str, source_file: str):
        self.job_id = job_id
        self.source_file = source_file

    @abstractmethod
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> list[Detection]:
        """
        Run detection on a single decoded frame.
        Returns detections (untracked) for this frame.
        Frame is BGR (OpenCV convention), HxWxC uint8.
        """
        ...

    @abstractmethod
    def finalize_chunk(self, tracks: list[Track]) -> list[Event]:
        """
        Given all tracks across the current chunk, classify events.
        Called once per chunk after all frames have been processed.
        """
        ...

    @property
    @abstractmethod
    def reel_targets(self) -> list[str]:
        """Which output reels this detector contributes to."""
        ...

    def detect_chunk(
        self,
        source_path: str,
        chunk_start_sec: float,
        chunk_duration_sec: float,
        source_fps: float,
    ) -> list:
        """
        Run detection on one chunk of video. Default implementation calls detect_frame
        per frame — subclasses may override for batch/GPU efficiency.
        Returns list of Detection objects.
        """
        return []

    def reset_chunk(self) -> None:
        """Reset per-chunk state. Called before processing each new chunk."""
        pass


class NullDetector(BaseDetector):
    """No-op detector for testing and dry-run mode."""

    @property
    def reel_targets(self) -> list[str]:
        return []

    def detect_frame(self, frame, frame_number, timestamp) -> list[Detection]:
        return []

    def finalize_chunk(self, tracks) -> list[Event]:
        return []
