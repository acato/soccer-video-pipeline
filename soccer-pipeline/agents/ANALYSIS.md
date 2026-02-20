# Agent: ANALYSIS

## Role
You own the computer vision layer: frame sampling, object detection, player tracking,
pose estimation, and pitch homography. You produce structured per-frame annotations
that feed the Event Detection agent.

## Responsibilities

- **Frame sampling**: 2 fps initial pass; 15–30 fps for flagged windows
- **Object detection**: players, ball, goalposts (YOLOv8)
- **Multi-object tracking**: ByteTrack; maintain consistent `track_id` per player
- **Pose estimation**: YOLOv8-pose on GK and nearby players (flagged windows only)
- **Pitch homography**: map pixel coordinates → pitch coordinates (meters)
- **GK identification**: identify which track_id is the goalkeeper each half
- Write per-frame annotations to the event store

## Frame Sampling Strategy

```
┌─────────────────────────────────────────────────────────┐
│  Pass 1: 2 fps full video                                │
│  → Detect: players, ball                                │
│  → Flag windows where ball is in GK zone OR             │
│     rapid player convergence detected                    │
│                                                          │
│  Pass 2: Full rate (30/60 fps) on flagged windows ±5s   │
│  → Full detection + tracking + pose                      │
│  → Write dense annotations to event store               │
└─────────────────────────────────────────────────────────┘
```

## Pitch Homography

Critical for GK zone definition and spatial event detection.

```python
class PitchHomography:
    """
    Compute H (3x3) mapping image coords → pitch coords (0,0)=center.
    FIFA standard pitch: 105m x 68m.
    
    Strategy:
    1. Detect pitch lines via Hough transform or line detector
    2. Match to known pitch template (center circle, penalty box corners)
    3. Compute H via cv2.findHomography (RANSAC)
    4. Recompute every 300 frames or on camera cut detection
    """
    
    def fit(self, frame: np.ndarray) -> bool: ...
    def to_pitch_coords(self, pixel_xy: tuple) -> tuple[float, float]: ...
    def gk_zone(self, side: Literal["left", "right"]) -> Polygon: ...
```

## GK Identification

```python
class GoalkeeperIdentifier:
    """
    Heuristic: The GK is the player who:
    1. Spends >60% of time within 20m of goal line
    2. Wears a distinct jersey color from outfield players
    
    Per half: re-identify at kick-off (camera pulls back, all players visible).
    Output: {half: 1|2, track_id: int, side: "left"|"right"}
    """
```

## YOLOv8 Configuration

```python
DETECTION_CONFIG = {
    "model": "yolov8x.pt",          # largest for accuracy; swap to yolov8m for speed
    "pose_model": "yolov8x-pose.pt",
    "confidence": 0.35,
    "iou": 0.5,
    "classes": [0],                 # person only for tracking; add 32 (ball) separately
    "device": "cuda:0",
    "half": True,                   # FP16 for speed
    "batch": 4,                     # auto-tune based on VRAM
    "imgsz": 1280,                  # 4K downscaled to 1280 wide for detection
}
```

## Frame Annotation Schema

```python
class FrameAnnotation(BaseModel):
    job_id: UUID
    frame_number: int
    timestamp_s: float
    detections: list[Detection]     # per-object bbox + class + conf
    tracks: list[Track]             # track_id + bbox + pitch_coords
    gk_track_id: int | None
    ball_track_id: int | None
    homography_valid: bool
    pose_keypoints: dict[int, Keypoints] | None  # track_id → keypoints
```

## Performance Budget

| Pass | Target throughput | GPU |
|---|---|---|
| Pass 1 (2fps, 1280px) | ≥ 30 fps processing | 8GB VRAM |
| Pass 2 (full rate, flagged) | ≥ real-time (30 fps) | 8GB VRAM |

If throughput falls below target: reduce `imgsz` to 960, then 640.

## Output
- `FrameAnnotation` records batch-inserted to SQLite
- Emit `analysis_complete` Celery task → Event Detection queue
- Structured log per pass with frame count, flagged windows, GPU utilization
