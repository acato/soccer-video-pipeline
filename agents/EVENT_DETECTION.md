# Agent: EVENT_DETECTION

## Role
You consume frame annotations from the Analysis agent and produce typed, timestamped
`VideoEvent` records. You are the intelligence layer that transforms raw CV data into
soccer-semantic events.

## Event Taxonomy

```
EventType (enum)
├── GK Events
│   ├── GK_SAVE_REFLEX       – reaction save, <0.5s from shot to contact
│   ├── GK_SAVE_DIVING       – GK leaves feet
│   ├── GK_CATCH             – clean hands catch
│   ├── GK_PUNCH             – fist clearance
│   ├── GK_DISTRIBUTION_THROW
│   ├── GK_DISTRIBUTION_KICK – drop kick, goal kick
│   └── GK_DISTRIBUTION_PUNT
├── Field Events
│   ├── SHOT_ON_TARGET
│   ├── SHOT_OFF_TARGET
│   ├── GOAL
│   ├── CLEARANCE_GOAL_LINE
│   └── CORNER_KICK
└── Meta
    ├── HALF_START
    ├── HALF_END
    └── CAMERA_CUT
```

## Detection Logic

### GK Save Detection
```
Trigger: ball trajectory enters GK zone (from homography)
         AND ball velocity vector points toward goal
         AND GK track_id is within 3m of ball (pitch coords)

Classify save type:
  - GK_SAVE_DIVING: GK keypoints show hip height < 0.5m at ball contact
  - GK_SAVE_REFLEX: Δt(shot_detected → contact) < 0.4s
  - GK_CATCH: ball not detected for 3+ frames after GK contact (in hands)
  - GK_PUNCH: ball continues trajectory after GK contact, deflected

Window: start_ts = trigger - 4s, end_ts = trigger + 3s
```

### GK Distribution Detection
```
Trigger: GK track_id has ball (last seen in GK bbox)
         AND ball then appears >15m from GK (next 60 frames)

Classify:
  - THROW: GK arm keypoints — elbow above shoulder during ball release
  - PUNT/DROP_KICK: GK leg keypoints — foot contact with ball at low height
  - GOAL_KICK: ball starts at penalty area edge, GK in proximity

Window: start_ts = trigger - 2s, end_ts = trigger + 4s
```

### Shot Detection
```
Trigger: Player (non-GK) leg keypoint makes contact with ball
         AND ball velocity increases significantly (Δv > threshold)
         AND ball trajectory toward goal half

For GOAL: ball passes goal line plane (homography), 
          no GK contact in final 0.2s

Window: start_ts = trigger - 3s, end_ts = trigger + 5s
```

## Confidence Scoring

Each event gets a `confidence: float` from 0.0–1.0 based on:
- Homography validity at trigger frame
- Tracking continuity (no gaps in GK track within window)
- Pose keypoint completeness (for classify-by-pose events)
- Ball detection quality

Thresholds (from CLAUDE.md):
- `< 0.4` → discarded
- `0.4–0.6` → `NEEDS_REVIEW` status
- `> 0.6` → `ACCEPTED`

## Deduplication

Events within 8 seconds of each other of the same type are merged
(take highest-confidence, expand window to cover both).

## Output

```python
# Batch insert to events table
events: list[VideoEvent]

# Emit to downstream queues
celery.send_task("gk_reel.process_events", args=[job_id])
celery.send_task("highlights.process_events", args=[job_id])
```

## Manual Review API

```
GET  /api/v1/review/pending          → list events needing review
POST /api/v1/review/{event_id}       → {"verdict": "accept|reject", "corrected_type": ...}
```
