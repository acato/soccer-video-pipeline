"""
VLM-first event detector: orchestrates the two-pass VLM detection pipeline.

Pass 1: Coarse scan — sample frames at fixed intervals, classify game state.
Pass 2: Boundary refinement — dense frames around each detected event region.

Binary inclusion: if the VLM confirms an event, it's included (confidence=1.0).
No heuristic pre-filtering, no confidence thresholds.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

import structlog

from src.detection.frame_sampler import FrameSampler
from src.detection.models import (
    Event,
    EventBoundary,
    EventType,
    GameState,
    SceneLabel,
)
from src.detection.scene_analyzer import SceneAnalyzer

log = structlog.get_logger(__name__)

# Map GameState values to EventType values for event creation
_STATE_TO_EVENT_TYPE: dict[GameState, EventType] = {
    GameState.CORNER_KICK: EventType.CORNER_KICK,
    GameState.GOAL_KICK: EventType.GOAL_KICK,
}

# Minimum consecutive frames (at coarse interval) to consider an event region
_MIN_REGION_FRAMES = 1


class VLMEventDetector:
    """Orchestrates the two-pass VLM detection pipeline."""

    def __init__(
        self,
        api_key: str,
        model: str,
        source_file: str,
        video_duration: float,
        job_id: str,
        event_types: list[str],
        frame_interval: float = 3.0,
        frame_width: int = 960,
    ):
        self._api_key = api_key
        self._model = model
        self._source_file = source_file
        self._video_duration = video_duration
        self._job_id = job_id
        self._event_types = event_types
        self._frame_interval = frame_interval
        self._frame_width = frame_width

        self._sampler = FrameSampler(source_file, frame_width=frame_width)
        self._analyzer = SceneAnalyzer(
            api_key=api_key,
            model=model,
            source_file=source_file,
            event_types=event_types,
        )

    def detect(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[Event]:
        """Run the full two-pass detection pipeline.

        1. Sample frames at fixed intervals
        2. Coarse scan -> identify event regions
        3. For each region, run boundary refinement
        4. Convert to Event objects

        Args:
            progress_callback: Called with progress fraction (0.0 to 1.0).

        Returns:
            List of Event objects with binary inclusion (confidence=1.0).
        """
        # -- Pass 1: sample + coarse scan --
        if progress_callback:
            progress_callback(0.0)

        frames = self._sampler.sample(
            duration_sec=self._video_duration,
            interval_sec=self._frame_interval,
        )
        log.info("vlm_detector.frames_sampled", count=len(frames))

        if progress_callback:
            progress_callback(0.1)

        labels = self._analyzer.scan(frames)

        if progress_callback:
            progress_callback(0.5)

        # -- Identify event regions --
        regions = self._identify_event_regions(labels)
        log.info("vlm_detector.regions_found", count=len(regions))

        if not regions:
            if progress_callback:
                progress_callback(1.0)
            return []

        # -- Pass 2: boundary refinement --
        events: list[Event] = []
        for i, region in enumerate(regions):
            center_time = region["center_sec"]
            event_type_str = region["event_type"]

            # Sample dense frames around the region
            region_frames = self._sampler.sample_range(
                center_sec=center_time,
                window_sec=15.0,
                interval_sec=1.0,
                duration_sec=self._video_duration,
            )

            boundary = self._analyzer.refine_event(
                region_frames=region_frames,
                event_type=event_type_str,
                center_time=center_time,
            )

            if boundary is not None:
                event = self._boundary_to_event(boundary)
                if event is not None:
                    events.append(event)

            if progress_callback:
                progress_callback(0.5 + 0.5 * (i + 1) / len(regions))

        log.info("vlm_detector.detection_complete", events=len(events))
        return events

    def _identify_event_regions(
        self, labels: list[SceneLabel]
    ) -> list[dict]:
        """Find contiguous regions of event-producing game states.

        Returns a list of dicts with keys: event_type, center_sec, start_sec, end_sec.
        """
        # Filter to labels whose game_state maps to an event type we care about
        target_states = set()
        for et in self._event_types:
            for gs, evt in _STATE_TO_EVENT_TYPE.items():
                if evt.value == et:
                    target_states.add(gs)

        regions: list[dict] = []
        current_region: list[SceneLabel] | None = None
        current_state: GameState | None = None

        for label in labels:
            if label.game_state in target_states:
                if current_region is not None and label.game_state == current_state:
                    current_region.append(label)
                else:
                    # Flush previous region if any
                    if current_region:
                        regions.append(self._region_from_labels(current_region, current_state))
                    current_region = [label]
                    current_state = label.game_state
            else:
                if current_region:
                    regions.append(self._region_from_labels(current_region, current_state))
                    current_region = None
                    current_state = None

        # Flush last region
        if current_region:
            regions.append(self._region_from_labels(current_region, current_state))

        return regions

    def _region_from_labels(
        self, labels: list[SceneLabel], state: GameState
    ) -> dict:
        event_type = _STATE_TO_EVENT_TYPE[state].value
        timestamps = [l.timestamp_sec for l in labels]
        return {
            "event_type": event_type,
            "center_sec": (min(timestamps) + max(timestamps)) / 2,
            "start_sec": min(timestamps),
            "end_sec": max(timestamps),
        }

    def _boundary_to_event(self, boundary: EventBoundary) -> Optional[Event]:
        """Convert an EventBoundary to an Event if confirmed."""
        if not boundary.confirmed:
            log.info("vlm_detector.event_not_confirmed",
                     event_type=boundary.event_type, reasoning=boundary.reasoning)
            return None

        # Clamp to video bounds
        start = max(0.0, boundary.clip_start_sec)
        end = min(self._video_duration, boundary.clip_end_sec)
        if end <= start:
            return None

        try:
            event_type = EventType(boundary.event_type)
        except ValueError:
            log.warning("vlm_detector.unknown_event_type", event_type=boundary.event_type)
            return None

        # Binary inclusion: VLM confirmed = confidence 1.0
        return Event(
            event_id=str(uuid.uuid4()),
            job_id=self._job_id,
            source_file=self._source_file,
            event_type=event_type,
            timestamp_start=start,
            timestamp_end=end,
            confidence=1.0,
            reel_targets=[boundary.event_type],
            is_goalkeeper_event=False,
            frame_start=int(start * 30),   # approximate at 30fps
            frame_end=int(end * 30),
            reviewed=False,
            review_override=None,
            metadata={
                "vlm_confirmed": True,
                "vlm_reasoning": boundary.reasoning,
                "vlm_model": self._model,
                "detection_method": "vlm_two_pass",
            },
            created_at=datetime.now(timezone.utc).isoformat(),
        )
