# Agent: HIGHLIGHTS

## Role
You own the match highlights reel. You select the most impactful moments from the
full event set and assemble a compelling, well-paced highlights video.

## Event Priority for Highlights

```python
HIGHLIGHT_PRIORITY = {
    EventType.GOAL:                10,
    EventType.GK_SAVE_DIVING:       9,
    EventType.GK_SAVE_REFLEX:       8,
    EventType.CLEARANCE_GOAL_LINE:  8,
    EventType.SHOT_ON_TARGET:       6,
    EventType.GK_CATCH:             5,
    EventType.GK_PUNCH:             5,
    EventType.SHOT_OFF_TARGET:      3,
    EventType.GK_DISTRIBUTION_PUNT: 3,
    EventType.CORNER_KICK:          2,
}
```

## Temporal Deduplication

- Events within 15 seconds of a higher-priority event of the same phase of play:
  merge into one extended clip rather than two separate clips.
- Goals: always include 8s pre-roll (build-up) + 10s post-roll (celebration).

## Pacing Engine

```python
class PacingEngine:
    """
    Ensure the reel has dynamic pacing — not a monotone sequence.
    
    Rules:
    1. Open with the best goal or best save (hook)
    2. Never more than 3 consecutive clips of the same type
    3. Alternate between attack (shots/goals) and defense (saves/clearances)
    4. Target reel duration: 3–5 minutes for a 90-min match
    5. End with a goal if available, else best save
    """
    
    def sequence(self, clips: list[ClipSpec]) -> list[ClipSpec]: ...
```

## Clip Timing

```python
HIGHLIGHT_TIMING = {
    EventType.GOAL:             {"pre": 8.0, "post": 10.0},
    EventType.GK_SAVE_DIVING:   {"pre": 4.0, "post":  3.0},
    EventType.SHOT_ON_TARGET:   {"pre": 3.0, "post":  4.0},
    "default":                  {"pre": 3.0, "post":  3.0},
}
```

## Output

```python
class HighlightsManifest(BaseModel):
    job_id: UUID
    reel_type: Literal["HIGHLIGHTS"]
    title: str           # "Match Highlights — Seattle Reign Academy — 2025-03-15"
    clips: list[ClipSpec]
    total_duration_s: float
    output_filename: str
    scoreline: str | None   # "2-1" if goals detected
```

Emit `render.encode_reel` Celery task with manifest.

## Future: Audio Energy Signal (Phase 3+)

When crowd/sideline audio is present, the highlights scorer can optionally
incorporate an audio energy signal (RMS loudness) to boost clips where
crowd noise spikes. This is additive to the priority score, not a replacement.
