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

    # ── Structural / restart events ──────────────────────────────────────
    KICKOFF             = "kickoff"
    THROW_IN            = "throw_in"


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
    EventType.KICKOFF:            [],
    EventType.THROW_IN:           [],
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
    EventType.GOAL:               0.50,   # Lowered — kickoff detection is the quality gate
    EventType.SHOT_STOP_DIVING:   0.55,   # Lowered — sideline camera makes saves ambiguous
    EventType.SHOT_STOP_STANDING: 0.55,   # Lowered — same reason
    EventType.ONE_ON_ONE:         0.65,
    EventType.PUNCH:              0.60,
    EventType.CATCH:              0.60,
    EventType.GOAL_KICK:          0.50,   # Lowered — visually distinctive, high recall needed
    EventType.DISTRIBUTION_SHORT: 0.60,
    EventType.DISTRIBUTION_LONG:  0.60,
    EventType.SHOT_ON_TARGET:     0.55,   # Lowered — shot detection feeds save/goal inference
    EventType.SHOT_OFF_TARGET:    0.55,
    EventType.NEAR_MISS:          0.60,
    EventType.DRIBBLE_SEQUENCE:   0.65,
    EventType.TACKLE:             0.65,
    EventType.PENALTY:            0.50,
    EventType.FREE_KICK_SHOT:     0.55,
    EventType.CORNER_KICK:        0.50,   # Lowered — distinctive but VLM uncertain at distance
    EventType.KICKOFF:            0.50,
    EventType.THROW_IN:           0.50,
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
        pre_pad_sec=10.0, post_pad_sec=15.0, max_clip_sec=45.0,
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
        pre_pad_sec=5.0, post_pad_sec=12.0, max_clip_sec=35.0,
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
        pre_pad_sec=10.0, post_pad_sec=15.0, max_clip_sec=60.0,
        min_confidence=0.50, is_gk_event=False,
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
    # ── Structural / restart events ────────────────────────────────────
    EventType.KICKOFF: EventTypeConfig(
        label="Kickoff", category="highlights",
        pre_pad_sec=2.0, post_pad_sec=2.0, max_clip_sec=10.0,
        min_confidence=0.50, is_gk_event=False,
    ),
    EventType.THROW_IN: EventTypeConfig(
        label="Throw-in", category="highlights",
        pre_pad_sec=1.0, post_pad_sec=2.0, max_clip_sec=10.0,
        min_confidence=0.50, is_gk_event=False,
    ),
}


# ---------------------------------------------------------------------------
# Jersey color palette — used by UI and match config
# ---------------------------------------------------------------------------

JERSEY_COLOR_PALETTE: dict[str, tuple[float, float, float]] = {
    # Achromatic — HSV (H: 0-180, S: 0-1, V: 0-1)
    "white":        (0.0,   0.05, 0.95),
    "silver":       (0.0,   0.05, 0.65),
    "gray":         (0.0,   0.06, 0.45),
    "black":        (0.0,   0.08, 0.10),
    # Reds
    "red":          (0.0,   0.85, 0.70),
    "dark_red":     (0.0,   0.85, 0.40),
    "maroon":       (0.0,   0.80, 0.28),
    "burgundy":     (170.0, 0.72, 0.30),
    # Oranges / yellows
    "orange":       (12.0,  0.90, 0.85),
    "neon_orange":  (10.0,  0.95, 0.95),
    "yellow":       (28.0,  0.85, 0.90),
    "neon_yellow":  (35.0,  0.95, 0.95),
    # Greens
    "green":        (60.0,  0.80, 0.55),
    "dark_green":   (60.0,  0.85, 0.28),
    "neon_green":   (55.0,  0.95, 0.95),
    "teal":         (88.0,  0.80, 0.55),
    # Blues
    "sky_blue":     (103.0, 0.48, 0.85),
    "light_blue":   (107.0, 0.58, 0.82),
    "blue":         (112.0, 0.82, 0.65),
    "dark_blue":    (115.0, 0.90, 0.32),
    "navy":         (116.0, 0.92, 0.20),
    # Other
    "purple":       (135.0, 0.65, 0.50),
    "pink":         (157.0, 0.45, 0.80),
    "hot_pink":     (153.0, 0.80, 0.82),
    "neon_pink":    (153.0, 0.90, 0.95),
}


def resolve_jersey_color(name: str) -> tuple[float, float, float]:
    """Return the HSV tuple for a named jersey color."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in JERSEY_COLOR_PALETTE:
        valid = ", ".join(sorted(JERSEY_COLOR_PALETTE))
        raise ValueError(f"Unknown jersey color {name!r}. Valid options: {valid}")
    return JERSEY_COLOR_PALETTE[key]


# ---------------------------------------------------------------------------
# VLM detection pipeline data types
# ---------------------------------------------------------------------------

class GameState(str, Enum):
    """Game state classification from VLM scene analysis."""
    ACTIVE_PLAY = "active_play"
    CORNER_KICK = "corner_kick"
    GOAL_KICK   = "goal_kick"
    STOPPAGE    = "stoppage"
    REPLAY      = "replay"
    OTHER       = "other"


@dataclass
class SceneLabel:
    """VLM classification result for a single frame."""
    timestamp_sec: float
    game_state: GameState


@dataclass
class EventBoundary:
    """Precise event boundaries from VLM refinement pass."""
    event_type: str
    clip_start_sec: float
    clip_end_sec: float
    confirmed: bool
    reasoning: str = ""


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
