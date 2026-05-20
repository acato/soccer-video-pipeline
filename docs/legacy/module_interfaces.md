# Module Interface Contracts

All public interfaces are binding. Changes require an ADR and version bump.
All timestamps are in seconds (float) relative to video start.

---

## src/ingestion/models.py

```python
import re
from enum import Enum
from pydantic import BaseModel

class JobStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    DETECTING = "detecting"
    SEGMENTING = "segmenting"
    ASSEMBLING = "assembling"
    COMPLETE = "complete"
    FAILED = "failed"

class VideoFile(BaseModel):
    path: str                    # absolute path on NAS mount
    filename: str
    duration_sec: float
    fps: float
    width: int
    height: int
    codec: str                   # h264 or hevc
    size_bytes: int
    sha256: str                  # for idempotency check

class KitConfig(BaseModel):
    """Per-match kit configuration. Colors vary by fixture (home/away)."""
    team_name: str               # e.g. "Home FC"
    outfield_color: str          # named color from JERSEY_COLOR_PALETTE
    gk_color: str                # named color from JERSEY_COLOR_PALETTE (required)

    @property
    def team_slug(self) -> str:
        """URL-safe identifier derived from team_name, e.g. "home_fc_gk"."""
        slug = re.sub(r"[^a-z0-9]+", "_", self.team_name.lower()).strip("_")
        return f"{slug}_gk"

class MatchConfig(BaseModel):
    """
    team    = the team whose GK reel is being produced
    opponent = needed for color-based identification only
    """
    team: KitConfig
    opponent: KitConfig

class Job(BaseModel):
    job_id: str                  # UUID
    video_file: VideoFile
    match_config: MatchConfig    # required — specifies team and kit colors
    status: JobStatus
    created_at: str              # ISO datetime
    updated_at: str
    reel_types: list[str]        # default: ["keeper", "highlights"]
    output_paths: dict[str, str] # reel_type -> output MP4 path
    error: str | None = None
    progress_pct: float = 0.0
```

### Valid reel types

| Value        | Description                                      |
|--------------|--------------------------------------------------|
| `"keeper"`   | Goalkeeper reel for the team specified in `match_config.team` |
| `"highlights"` | Match highlights reel                          |
| `"player"`   | Individual outfield player reel                  |

---

## src/detection/models.py

```python
from pydantic import BaseModel
from enum import Enum

class EventType(str, Enum):
    # Goalkeeper events
    SHOT_STOP_DIVING = "shot_stop_diving"
    SHOT_STOP_STANDING = "shot_stop_standing"
    PUNCH = "punch"
    CATCH = "catch"
    GOAL_KICK = "goal_kick"
    DISTRIBUTION_SHORT = "distribution_short"
    DISTRIBUTION_LONG = "distribution_long"
    ONE_ON_ONE = "one_on_one"
    # Highlights events
    SHOT_ON_TARGET = "shot_on_target"
    SHOT_OFF_TARGET = "shot_off_target"
    GOAL = "goal"
    NEAR_MISS = "near_miss"
    DRIBBLE_SEQUENCE = "dribble_sequence"
    TACKLE = "tackle"
    PENALTY = "penalty"
    FREE_KICK_SHOT = "free_kick_shot"

# GK event types that route to the "keeper" reel
GK_REEL_TYPES: tuple[str, ...] = ("keeper",)

class Detection(BaseModel):
    frame_number: int
    timestamp: float
    class_name: str              # "player", "ball", "goalkeeper"
    confidence: float
    bbox: tuple[float, float, float, float]  # x, y, w, h normalized

class Track(BaseModel):
    track_id: int
    detections: list[Detection]
    is_goalkeeper: bool = False

class Event(BaseModel):
    event_id: str                # UUID
    job_id: str
    source_file: str
    event_type: EventType
    timestamp_start: float
    timestamp_end: float
    confidence: float
    reel_targets: list[str]      # e.g. ["keeper"] or ["highlights"]
    player_track_id: int | None = None
    is_goalkeeper_event: bool = False
    frame_start: int
    frame_end: int
    reviewed: bool = False
    review_override: bool | None = None
    metadata: dict = {}
```

---

## src/detection/jersey_classifier.py

```python
# 29 named colors mapping to HSV tuples (H: 0–180, S: 0–1, V: 0–1)
JERSEY_COLOR_PALETTE: dict[str, tuple[float, float, float]]

def resolve_jersey_color(name: str) -> tuple[float, float, float]:
    """
    Convert a human-readable color name to an HSV tuple.
    Normalises case, spaces and hyphens (e.g. "Dark Blue" == "dark_blue").
    Raises ValueError for unknown color names.
    """

def identify_gk_by_known_colors(
    track_colors: dict[int, tuple],
    track_positions: dict[int, float],
    home_gk_hsv: tuple[float, float, float],
    away_gk_hsv: tuple[float, float, float],
    min_similarity: float = 0.40,
) -> dict[str, int | None]:
    """
    Supervised GK identification using known jersey colors.
    Returns {"keeper_a": track_id | None, "keeper_b": track_id | None}
    where keeper_a is the GK on the left half of the pitch.
    """
```

---

## src/detection/base.py

```python
from abc import ABC, abstractmethod
from .models import Detection, Event
import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def detect_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> list[Detection]:
        """Run detection on a single decoded frame."""
        ...

    @abstractmethod
    def classify_events(self, tracks: list[Track], job_id: str, source_file: str) -> list[Event]:
        """Given tracked detections across a chunk, classify discrete events."""
        ...

    @property
    @abstractmethod
    def reel_targets(self) -> list[str]:
        """Which reels this detector contributes to."""
        ...
```

---

## src/segmentation/clipper.py

```python
class ClipBoundary(BaseModel):
    source_file: str
    start_sec: float             # with pre-event padding applied
    end_sec: float               # with post-event padding applied
    events: list[str]            # event_ids included in this clip
    reel_type: str

def compute_clips(
    events: list[Event],
    video_duration: float,
    reel_type: str,
    pre_pad: float,
    post_pad: float,
) -> list[ClipBoundary]:
    """Convert event list to padded, bounded clip windows. No overlaps."""
    ...
```

---

## src/assembly/encoder.py

```python
def extract_clip(
    source_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    use_stream_copy: bool = True,
) -> bool:
    """Extract a clip from source video using FFmpeg. Returns True on success."""
    ...

def concat_clips(
    clip_paths: list[str],
    output_path: str,
    add_timestamps: bool = True,
) -> bool:
    """Concatenate ordered clips into a single MP4 reel."""
    ...
```

---

## src/api/routes/jobs.py

```
POST   /jobs              — Submit new job
                          Body: {
                            "nas_path": str,
                            "match_config": {
                              "team": {
                                "team_name": str,
                                "outfield_color": str,   # named color
                                "gk_color": str          # named color
                              },
                              "opponent": {
                                "team_name": str,
                                "outfield_color": str,
                                "gk_color": str
                              }
                            },
                            "reel_types": list[str]      # default: ["keeper", "highlights"]
                          }
                          Returns: Job

GET    /jobs/{job_id}     — Get full job record
                          Returns: Job

GET    /jobs/{job_id}/status
                          Returns: { "status": JobStatus, "progress_pct": float }

POST   /jobs/{job_id}/retry
                          Returns: Job (reset to PENDING)

GET    /reels/{job_id}/{reel_type}
                          Returns: { "download_url": str } or 404
```
