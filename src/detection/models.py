"""
Detection and event data models.
These types flow through the entire detection → segmentation → assembly pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # ── Goalkeeper reel events ──────────────────────────────────────────────
    SHOT_STOP_DIVING    = "shot_stop_diving"
    SHOT_STOP_STANDING  = "shot_stop_standing"
    PUNCH               = "punch"
    CATCH               = "catch"
    GOAL_KICK           = "goal_kick"
    DISTRIBUTION_SHORT  = "distribution_short"
    DISTRIBUTION_LONG   = "distribution_long"
    ONE_ON_ONE          = "one_on_one"
    CORNER_KICK         = "corner_kick"

    # ── Highlights reel events ──────────────────────────────────────────────
    SHOT_ON_TARGET      = "shot_on_target"
    SHOT_OFF_TARGET     = "shot_off_target"
    GOAL                = "goal"
    NEAR_MISS           = "near_miss"
    DRIBBLE_SEQUENCE    = "dribble_sequence"
    TACKLE              = "tackle"
    PENALTY             = "penalty"
    FREE_KICK_SHOT      = "free_kick_shot"


# Map each event type to which reels it contributes to.
# GK event types have empty defaults — reel_targets are assigned dynamically
# at event creation time based on which keeper (keeper_a / keeper_b) is involved.
EVENT_REEL_MAP: dict[EventType, list[str]] = {
    EventType.SHOT_STOP_DIVING:   [],
    EventType.SHOT_STOP_STANDING: [],
    EventType.PUNCH:              [],
    EventType.CATCH:              [],
    EventType.GOAL_KICK:          [],
    EventType.DISTRIBUTION_SHORT: [],
    EventType.DISTRIBUTION_LONG:  [],
    EventType.ONE_ON_ONE:         [],  # keeper role + "highlights" assigned at creation
    EventType.CORNER_KICK:        [],  # reel_targets assigned dynamically
    EventType.SHOT_ON_TARGET:     ["highlights"],
    EventType.SHOT_OFF_TARGET:    ["highlights"],
    EventType.GOAL:               ["highlights"],
    EventType.NEAR_MISS:          ["highlights"],
    EventType.DRIBBLE_SEQUENCE:   [],
    EventType.TACKLE:             [],
    EventType.PENALTY:            ["highlights", "keeper"],
    EventType.FREE_KICK_SHOT:     ["highlights"],
}

# Valid keeper reel type
GK_REEL_TYPES = ("keeper",)

# GK-specific event types (reel_targets assigned dynamically)
_GK_EVENT_TYPES = frozenset({
    EventType.SHOT_STOP_DIVING,
    EventType.SHOT_STOP_STANDING,
    EventType.PUNCH,
    EventType.CATCH,
    EventType.GOAL_KICK,
    EventType.DISTRIBUTION_SHORT,
    EventType.DISTRIBUTION_LONG,
    EventType.ONE_ON_ONE,
    EventType.CORNER_KICK,
})


def is_gk_event_type(event_type: EventType) -> bool:
    """Return True if the event type is a goalkeeper-specific event."""
    return event_type in _GK_EVENT_TYPES

# Per-event minimum confidence thresholds (override MIN_EVENT_CONFIDENCE global)
EVENT_CONFIDENCE_THRESHOLDS: dict[EventType, float] = {
    EventType.GOAL:               0.85,   # High precision — false goals are very bad UX
    EventType.SHOT_STOP_DIVING:   0.75,
    EventType.SHOT_STOP_STANDING: 0.70,
    EventType.ONE_ON_ONE:         0.75,
    EventType.PUNCH:              0.65,
    EventType.CATCH:              0.70,
    EventType.GOAL_KICK:          0.65,
    EventType.DISTRIBUTION_SHORT: 0.65,
    EventType.DISTRIBUTION_LONG:  0.68,
    EventType.SHOT_ON_TARGET:     0.70,
    EventType.SHOT_OFF_TARGET:    0.65,
    EventType.NEAR_MISS:          0.70,
    EventType.DRIBBLE_SEQUENCE:   0.65,
    EventType.TACKLE:             0.65,
    EventType.PENALTY:            0.60,
    EventType.FREE_KICK_SHOT:     0.65,
    EventType.CORNER_KICK:        0.65,
}


# ---------------------------------------------------------------------------
# EventTypeConfig — single source of truth per event type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventTypeConfig:
    """Per-event-type configuration for clip cutting and UI display."""
    label: str            # Human-readable, e.g. "Diving Save"
    category: str         # "goalkeeper" | "highlights"
    pre_pad_sec: float    # Clip padding before event
    post_pad_sec: float   # Clip padding after event
    max_clip_sec: float   # Max single clip duration
    min_confidence: float # Detection threshold
    is_gk_event: bool     # Needs jersey color classification


EVENT_TYPE_CONFIG: dict[EventType, EventTypeConfig] = {
    # ── Goalkeeper events ─────────────────────────────────────────────────
    EventType.SHOT_STOP_DIVING: EventTypeConfig(
        label="Diving Save", category="goalkeeper",
        pre_pad_sec=8.0, post_pad_sec=2.0, max_clip_sec=25.0,
        min_confidence=0.75, is_gk_event=True,
    ),
    EventType.SHOT_STOP_STANDING: EventTypeConfig(
        label="Standing Save", category="goalkeeper",
        pre_pad_sec=8.0, post_pad_sec=2.0, max_clip_sec=25.0,
        min_confidence=0.70, is_gk_event=True,
    ),
    EventType.PUNCH: EventTypeConfig(
        label="Punch", category="goalkeeper",
        pre_pad_sec=8.0, post_pad_sec=2.0, max_clip_sec=25.0,
        min_confidence=0.65, is_gk_event=True,
    ),
    EventType.CATCH: EventTypeConfig(
        label="Catch", category="goalkeeper",
        pre_pad_sec=8.0, post_pad_sec=2.0, max_clip_sec=25.0,
        min_confidence=0.70, is_gk_event=True,
    ),
    EventType.GOAL_KICK: EventTypeConfig(
        label="Goal Kick", category="goalkeeper",
        pre_pad_sec=1.0, post_pad_sec=2.0, max_clip_sec=15.0,
        min_confidence=0.65, is_gk_event=True,
    ),
    EventType.DISTRIBUTION_SHORT: EventTypeConfig(
        label="Short Distribution", category="goalkeeper",
        pre_pad_sec=1.0, post_pad_sec=2.0, max_clip_sec=20.0,
        min_confidence=0.65, is_gk_event=True,
    ),
    EventType.DISTRIBUTION_LONG: EventTypeConfig(
        label="Long Distribution", category="goalkeeper",
        pre_pad_sec=1.0, post_pad_sec=2.0, max_clip_sec=20.0,
        min_confidence=0.68, is_gk_event=True,
    ),
    EventType.ONE_ON_ONE: EventTypeConfig(
        label="1-on-1", category="goalkeeper",
        pre_pad_sec=3.0, post_pad_sec=2.0, max_clip_sec=30.0,
        min_confidence=0.75, is_gk_event=True,
    ),
    EventType.CORNER_KICK: EventTypeConfig(
        label="Corner Kick", category="goalkeeper",
        pre_pad_sec=3.0, post_pad_sec=2.0, max_clip_sec=25.0,
        min_confidence=0.65, is_gk_event=True,
    ),
    EventType.PENALTY: EventTypeConfig(
        label="Penalty", category="goalkeeper",
        pre_pad_sec=8.0, post_pad_sec=2.0, max_clip_sec=25.0,
        min_confidence=0.60, is_gk_event=True,
    ),
    # ── Highlights events ─────────────────────────────────────────────────
    EventType.SHOT_ON_TARGET: EventTypeConfig(
        label="Shot on Target", category="highlights",
        pre_pad_sec=3.0, post_pad_sec=5.0, max_clip_sec=30.0,
        min_confidence=0.70, is_gk_event=False,
    ),
    EventType.SHOT_OFF_TARGET: EventTypeConfig(
        label="Shot off Target", category="highlights",
        pre_pad_sec=3.0, post_pad_sec=5.0, max_clip_sec=30.0,
        min_confidence=0.65, is_gk_event=False,
    ),
    EventType.GOAL: EventTypeConfig(
        label="Goal", category="highlights",
        pre_pad_sec=5.0, post_pad_sec=8.0, max_clip_sec=60.0,
        min_confidence=0.85, is_gk_event=False,
    ),
    EventType.NEAR_MISS: EventTypeConfig(
        label="Near Miss", category="highlights",
        pre_pad_sec=3.0, post_pad_sec=5.0, max_clip_sec=30.0,
        min_confidence=0.70, is_gk_event=False,
    ),
    EventType.DRIBBLE_SEQUENCE: EventTypeConfig(
        label="Dribble Sequence", category="highlights",
        pre_pad_sec=3.0, post_pad_sec=5.0, max_clip_sec=30.0,
        min_confidence=0.65, is_gk_event=False,
    ),
    EventType.TACKLE: EventTypeConfig(
        label="Tackle", category="highlights",
        pre_pad_sec=3.0, post_pad_sec=5.0, max_clip_sec=30.0,
        min_confidence=0.65, is_gk_event=False,
    ),
    EventType.FREE_KICK_SHOT: EventTypeConfig(
        label="Free Kick Shot", category="highlights",
        pre_pad_sec=3.0, post_pad_sec=5.0, max_clip_sec=30.0,
        min_confidence=0.65, is_gk_event=False,
    ),
}


class BoundingBox(BaseModel):
    """Normalized bounding box (0.0–1.0 relative to frame dimensions)."""
    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height


class FieldPosition(BaseModel):
    """Position on the pitch in meters (requires homography calibration)."""
    x_meters: float   # 0 = left goal line, ~105 = right goal line
    y_meters: float   # 0 = bottom touchline, ~68 = top touchline


class Detection(BaseModel):
    """Single object detection in one frame."""
    frame_number: int
    timestamp: float            # Seconds from video start
    class_name: str             # "player", "ball", "goalkeeper", "referee"
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None  # Assigned by tracker
    metadata: dict = Field(default_factory=dict)


class Track(BaseModel):
    """Persistent object track across multiple frames."""
    track_id: int
    detections: list[Detection] = Field(default_factory=list)
    is_goalkeeper: bool = False
    jersey_color_hsv: Optional[tuple[float, float, float]] = None  # Dominant HSV

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_number if self.detections else 0

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_number if self.detections else 0

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


class Event(BaseModel):
    """A detected soccer event that may contribute to one or more reels."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    source_file: str            # Relative path from NAS mount
    event_type: EventType
    timestamp_start: float      # Seconds from video start
    timestamp_end: float
    confidence: float
    reel_targets: list[str]     # ["goalkeeper"], ["highlights"], or both
    player_track_id: Optional[int] = None
    is_goalkeeper_event: bool = False
    frame_start: int
    frame_end: int
    bounding_box: Optional[BoundingBox] = None
    field_position: Optional[FieldPosition] = None
    reviewed: bool = False
    review_override: Optional[bool] = None  # None=auto, True=force-include, False=force-exclude
    metadata: dict = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def duration_sec(self) -> float:
        return self.timestamp_end - self.timestamp_start

    def should_include(self, global_min_confidence: float = 0.65) -> bool:
        """Return True if this event should be included in output reels."""
        if self.review_override is not None:
            return self.review_override
        threshold = EVENT_CONFIDENCE_THRESHOLDS.get(self.event_type, global_min_confidence)
        return self.confidence >= threshold
