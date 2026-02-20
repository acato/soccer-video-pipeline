# Agent: PLAYER_REEL

## Role (Phase 2 — Not Yet Active)
On-demand personalized reel for any field player. You depend on player re-ID
across clips, which requires appearance-based re-identification beyond basic tracking.

## Prerequisites (must be complete before Phase 2 begins)
- [ ] `global_player_id` field populated in `Track` schema (Architect)
- [ ] Appearance-based re-ID model integrated in Analysis agent (e.g. OSNet/StrongSORT)
- [ ] Player jersey number OCR (optional — improves re-ID accuracy)
- [ ] Phase 1 (GK Reel) in production and stable

## Planned Responsibilities

- Accept on-demand requests: `POST /api/v1/reels/player {"job_id": ..., "player_id": ...}`
- Query event store for all frames where `player_id` has high-activity events
- Activity scoring for field players:
  - Ball touches (possession events)
  - Shots
  - Key passes / through balls
  - Defensive actions (tackles, interceptions, clearances)
  - Dribbles (rapid direction changes with ball)
- Clip selection: similar to GK reel but player-centric
- Reel assembly via Render agent

## Re-ID Strategy (to be detailed in Phase 2 ADR)

```
Option A: OSNet (lightweight, fast) — appearance embedding per crop
Option B: StrongSORT (combines motion + appearance) — better across camera cuts
Option C: Jersey number OCR (pytesseract / PaddleOCR) — precise but slow

Recommendation: StrongSORT for tracking + OCR for disambiguation on high-confidence crops.
ADR required before implementation.
```

## API Contract (planned)

```
POST /api/v1/reels/player
{
  "job_ids": ["uuid1", "uuid2"],   // can span multiple matches
  "player_name": "Elena",          // display name only
  "player_id": 7,                  // jersey number OR track_id if known
  "deliverable": "PLAYER_REEL"
}

→ 202 Accepted
{
  "reel_job_id": "uuid",
  "status_url": "/api/v1/jobs/uuid/status"
}
```
