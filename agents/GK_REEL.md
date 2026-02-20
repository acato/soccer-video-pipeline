# Agent: GK_REEL

## Role
You own the goalkeeper reel product. Given accepted GK events from the event store,
you select, rank, trim, and sequence clips into a final GK reel video.

## Responsibilities

- Query event store for all GK_* events for a given job_id
- Filter to accepted + needs_review events (let operator confirm review items)
- Score and rank clips by visual impact
- Define per-clip in/out points with appropriate pre/post roll
- Generate clip manifest
- Invoke Render agent for final encoding

## Clip Selection Rules

### Inclusion (all must pass)
- Event type in: GK_SAVE_*, GK_CATCH, GK_PUNCH, GK_DISTRIBUTION_*
- Confidence ≥ 0.4 (0.4–0.6 flagged for review before inclusion)
- Clip duration: 4–12 seconds (trim if longer, skip if shorter than 3s)

### Ranking (descending)
1. GK_SAVE_DIVING — highest visual impact
2. GK_SAVE_REFLEX
3. GK_CATCH (ball under pressure)
4. GK_PUNCH
5. GK_SAVE_* (other)
6. GK_DISTRIBUTION_PUNT / THROW (long range)
7. GK_DISTRIBUTION_KICK
8. GK_DISTRIBUTION_THROW (short)

Score formula:
```python
score = (
    TYPE_WEIGHT[event.event_type] * 0.5 +
    event.confidence * 0.3 +
    distribution_bonus(event) * 0.2   # variety: penalize >3 consecutive same type
)
```

## Per-Clip Timing

```python
PRE_ROLL = {
    "GK_SAVE_*": 4.0,        # show the shot developing
    "GK_DISTRIBUTION_*": 2.0,
    "default": 3.0,
}
POST_ROLL = {
    "GK_SAVE_*": 3.0,        # show the reaction
    "GK_DISTRIBUTION_*": 4.0, # show the ball landing
    "default": 2.5,
}
```

## Reel Structure

```
[Optional 3s title card: "ELENA — GOALKEEPER HIGHLIGHTS"]
[Clip 1: best save]
[0.5s black transition]
[Clip 2–N: ranked sequence]
[Final clip: best distribution or memorable save]
```

Maximum reel length: configurable (default: 8 minutes).
If total clip time exceeds max, drop lowest-ranked clips.

## Output

```python
class GKReelManifest(BaseModel):
    job_id: UUID
    reel_type: Literal["GK_REEL"]
    title: str                    # e.g. "Elena — GK Reel — 2025-03-15"
    clips: list[ClipSpec]         # ordered
    total_duration_s: float
    output_filename: str          # e.g. "gk_reel_20250315.mp4"

class ClipSpec(BaseModel):
    source_file: Path
    start_ts: float
    end_ts: float
    event_id: UUID
    event_type: EventType
    rank: int
    title_overlay: str | None    # optional burn-in: "Save #3"
```

Emit `render.encode_reel` Celery task with manifest.
