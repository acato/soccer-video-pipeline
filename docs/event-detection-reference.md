# Event Detection Reference

How each of the 17 event types is currently detected, including thresholds, confidence values, and reel assignments.

## Composable Reels Architecture

Reels are now user-composed lists of event types via `ReelSpec`:

```python
class ReelSpec(BaseModel):
    name: str                    # user-chosen, e.g. "deflections"
    event_types: list[str]       # EventType values to include
    max_reel_duration_sec: float = 1200.0
```

Two presets are available: `"keeper"` (all GK event types) and `"highlights"` (shot/goal types). Users can also build custom reels by selecting individual event types.

Each event type has a per-type configuration in `EVENT_TYPE_CONFIG` (`src/detection/models.py`):

| Event Type | Category | Pre-pad | Post-pad | Max clip | Min confidence | GK event? |
|-----------|----------|---------|----------|----------|---------------|-----------|
| SHOT_STOP_DIVING | goalkeeper | 8.0s | 2.0s | 25s | 0.75 | Yes |
| SHOT_STOP_STANDING | goalkeeper | 8.0s | 2.0s | 25s | 0.70 | Yes |
| PUNCH | goalkeeper | 8.0s | 2.0s | 25s | 0.65 | Yes |
| CATCH | goalkeeper | 8.0s | 2.0s | 25s | 0.70 | Yes |
| GOAL_KICK | goalkeeper | 1.0s | 2.0s | 15s | 0.65 | Yes |
| DISTRIBUTION_SHORT | goalkeeper | 1.0s | 2.0s | 20s | 0.65 | Yes |
| DISTRIBUTION_LONG | goalkeeper | 1.0s | 2.0s | 20s | 0.68 | Yes |
| ONE_ON_ONE | goalkeeper | 3.0s | 2.0s | 30s | 0.75 | Yes |
| CORNER_KICK | goalkeeper | 3.0s | 2.0s | 25s | 0.65 | Yes |
| PENALTY | goalkeeper | 3.0s | 2.0s | 25s | 0.60 | Yes |
| SHOT_ON_TARGET | highlights | 3.0s | 5.0s | 30s | 0.70 | No |
| SHOT_OFF_TARGET | highlights | 3.0s | 5.0s | 30s | 0.65 | No |
| GOAL | highlights | 5.0s | 8.0s | 60s | 0.85 | No |
| NEAR_MISS | highlights | 3.0s | 5.0s | 30s | 0.70 | No |
| FREE_KICK_SHOT | highlights | 3.0s | 5.0s | 30s | 0.65 | No |
| DRIBBLE_SEQUENCE | highlights | 3.0s | 5.0s | 30s | 0.65 | No |
| TACKLE | highlights | 3.0s | 5.0s | 30s | 0.65 | No |

The worker pipeline applies quality filters once (spatial filter + sim gate for GK events), then loops over each `ReelSpec`, filtering events by the spec's event types, computing clips with per-type padding from `EVENT_TYPE_CONFIG`, and assembling each reel.

### API Submission

```json
{
  "nas_path": "match.mp4",
  "match_config": {
    "team": {"team_name": "Rush", "outfield_color": "white", "gk_color": "teal"},
    "opponent": {"team_name": "GA 2008", "outfield_color": "blue", "gk_color": "purple"}
  },
  "reels": [
    {"name": "deflections", "event_types": ["shot_stop_diving", "shot_stop_standing"]},
    {"name": "all_gk", "event_types": ["shot_stop_diving", "shot_stop_standing", "punch", "catch", "goal_kick", "distribution_short", "distribution_long", "one_on_one", "corner_kick", "penalty"]}
  ]
}
```

Legacy `reel_types: ["keeper", "highlights"]` is still accepted and auto-converted to the corresponding `ReelSpec` presets.

## Event Detection Summary

| # | Event Type | Detector | Key Conditions | Confidence | Category | Threshold |
|---|-----------|----------|---------------|------------|----------|-----------|
| 1 | SHOT_STOP_DIVING | GK Detector / Ball Touch | GK vertical velocity >2.5 body-heights/s (GK-first path) OR ball touch + GK vertical velocity >2.0 body-heights/s (ball-first path) | 0.50-0.90 | goalkeeper | 0.75 |
| 2 | SHOT_STOP_STANDING | GK Detector / Ball Touch | GK vertical velocity 1.0-2.5 body-heights/s + ball proximity <0.15 OR ball speed drop >60% near GK | 0.50-0.80 | goalkeeper | 0.70 |
| 3 | PUNCH | ActionClassifier (ML) | VideoMAE temporal action recognition only. No heuristic fallback. | model-dependent | goalkeeper | 0.65 |
| 4 | CATCH | Ball Touch Detector | Ball trajectory gap (disappears for >=5 frames) with incoming speed >0.30 OR end-of-trajectory catch | 0.60-0.90 | goalkeeper | 0.70 |
| 5 | GOAL_KICK | Ball Touch (reclassify) | Dead-ball reclassification: save/catch where ball was stationary (speed <0.05) for >=1s, then kicked (speed spike). GK center_y >0.75. | inherited | goalkeeper | 0.65 |
| 6 | DISTRIBUTION_SHORT | GK Detector | GK stationary (vel <1.0 body-widths/s) then moves (vel >1.5). GK center_y 0.55-0.75. | 0.55-0.90 | goalkeeper | 0.65 |
| 7 | DISTRIBUTION_LONG | GK Detector | Same as SHORT but GK center_y <=0.55 (further from goal line). | 0.55-0.90 | goalkeeper | 0.68 |
| 8 | ONE_ON_ONE | GK Detector | GK deviation from goal line >0.3 body-heights for >=3 consecutive frames. | 0.65-0.88 | goalkeeper | 0.75 |
| 9 | CORNER_KICK | Ball Touch Detector | Two methods: (1) Spatial: ball near frame edge (x<0.12/x>0.88 AND y<0.15/y>0.85, or strong edge x<0.08/x>0.92/y<0.10/y>0.90), stationary >=0.5s, kicked, >=3-5 players. (2) Post-save: save event → ball disappears (gap >=1s) → ball reappears stationary → kicked with >=4 players. | 0.70 | goalkeeper | 0.65 |
| 10 | SHOT_ON_TARGET | Event Classifier | Ball speed >1.0 (any position) OR >0.50 near goal (y<0.25/y>0.75) or sides (x<0.15/x>0.85). Near-goal required for ON_TARGET. | 0.55-0.90 | highlights | 0.70 |
| 11 | SHOT_OFF_TARGET | Event Classifier | Same velocity conditions as ON_TARGET but ball NOT in goal-scoring area. | 0.55-0.90 | highlights | 0.65 |
| 12 | GOAL | ActionClassifier (ML) / Ball-in-Net | VideoMAE only, OR ball at x<0.03/x>0.97 moving fast (speed>0.30) toward that side then disappears (gap>=3 frames). | model-dependent / 0.85 | highlights | 0.85 |
| 13 | NEAR_MISS | ActionClassifier (ML) | VideoMAE only. No heuristic fallback. | model-dependent | highlights | 0.70 |
| 14 | PENALTY | ActionClassifier (ML) | VideoMAE only. No heuristic fallback. | model-dependent | goalkeeper + highlights | 0.60 |
| 15 | FREE_KICK_SHOT | ActionClassifier (ML) | VideoMAE only. No heuristic fallback. | model-dependent | highlights | 0.65 |
| 16 | DRIBBLE_SEQUENCE | Event Classifier | Player track >=10 detections + speed >0.08 body-widths/s sustained >=1.5s. | 0.67 | highlights | 0.65 |
| 17 | TACKLE | Event Classifier | Two player tracks within distance <0.05 normalized units in same frame. | 0.65 | highlights | 0.65 |

## Detailed Breakdown

### Goalkeeper Events (Keeper Reel)

#### SHOT_STOP_DIVING
- **Detectors:** `GoalkeeperDetector._detect_saves()`, `BallTouchDetector._classify_touch()`
- **Conditions (GK-first path):**
  - GK vertical velocity >2.5 body-heights per second (unmistakable dive), OR
  - Ball within ARM_REACH=0.06 of GK bbox edge + trajectory change (direction >45deg or speed drop >60% or ball caught)
- **Conditions (Ball-first path):**
  - Ball trajectory change (speed_drop or direction_change) classified to nearest GK player
  - GK vertical velocity >2.0 body-heights/s (computed from `_compute_player_vertical_velocity()` over ±10 frames)
  - If GK vertical velocity <=2.0, classified as SHOT_STOP_STANDING instead
- **Event window:** +/-0.5s (+ keeper reel padding -8.0/+2.0s)
- **Confidence:** Save method: base 0.50 + vertical_velocity, capped 0.80-0.90 with ball proximity bonus. Ball contact: 0.85.

#### SHOT_STOP_STANDING
- **Detectors:** `GoalkeeperDetector._detect_saves()`, `BallTouchDetector`
- **Conditions:**
  - GK vertical velocity 1.0-2.5 body-heights/s with ball proximity (distance <0.15), OR
  - Ball speed drop >60% or direction change >45deg near GK (ARM_REACH=0.06)
- **Event window:** +/-0.5s
- **Confidence:** 0.50-0.80 (base + vertical velocity, ball proximity bonus +0.10)

#### CATCH
- **Detector:** `BallTouchDetector._find_touch_frames()`
- **Conditions:**
  - Ball trajectory gap: ball detected, disappears for >=5 frames, with fast incoming speed >0.30, OR
  - End-of-trajectory catch: ball moving fast then detection ends within chunk
- **Confidence:** 0.60 + sim_team_gk * 0.3 (range 0.60-0.90), requires 0.10 margin over other jerseys

#### GOAL_KICK
- **Detector:** `BallTouchDetector._reclassify_dead_ball_collections()`
- **Conditions:**
  - Post-filter reclassification of save/catch events
  - Ball was stationary (speed <0.05) for >=1s after GK touched it
  - Then ball kicked (speed spike)
  - GK bbox center_y >0.75 (goal line area)
  - Pre-filter rejects slow pickups (incoming speed <0.30)
- **Event window:** -1.5s / +2.0s (expanded from -0.5s to capture ball placement)

#### DISTRIBUTION_SHORT
- **Detector:** `GoalkeeperDetector._detect_distribution()`
- **Conditions:**
  - GK pre-velocity <1.0 body-widths/s (stationary)
  - GK post-velocity >1.5 body-widths/s (threshold)
  - GK center_y between 0.55 and 0.75 (mid-goal area)
- **Event window:** -0.5s / +2.0s

#### DISTRIBUTION_LONG
- **Detector:** `GoalkeeperDetector._detect_distribution()`
- **Conditions:**
  - Same velocity conditions as DISTRIBUTION_SHORT
  - GK center_y <=0.55 (further from goal line)
- **Event window:** -0.5s / +2.0s

#### ONE_ON_ONE
- **Detector:** `GoalkeeperDetector._detect_one_on_ones()`
- **Conditions:**
  - GK moves significantly from goal line (deviation from baseline >0.3 body-heights)
  - Sustained for >=3 consecutive frames
- **Event window:** -0.5s / +3.0s
- **Note:** Dual reel — appears in both keeper AND highlights reels

#### CORNER_KICK
- **Detector:** `BallTouchDetector._detect_corner_kicks()` — two methods
- **Method 1 — Spatial (relaxed for auto-panning cameras):**
  - Ball near frame edge: corner zone (x<0.12 AND (y<0.15 OR y>0.85)) or strong edge (x<0.08 OR x>0.92 OR y<0.10 OR y>0.90)
  - Ball stationary/slow (speed <0.08) for >=0.5s
  - Then kicked (speed increase >0.12)
  - Corner zone requires >=3 players in same half; strong edge requires >=5
- **Method 2 — Post-save restart inference:**
  - A save event (SHOT_STOP_STANDING/DIVING, CATCH) detected in same chunk
  - Ball trajectory gap >=1.0s starting within ±1s of the save (ball went out of play)
  - After gap, ball reappears, becomes stationary, then kicked
  - >=4 players in same half of frame at kick time
  - Reliably detects corner kicks after GK deflections regardless of camera type
- **Event window:** -1.0s / +4.0s
- **Confidence:** 0.70

### Highlights Events

#### SHOT_ON_TARGET / SHOT_OFF_TARGET
- **Detector:** `event_classifier._detect_shots_from_all()`
- **Conditions:**
  - Scans ALL tracks for ball detections (ByteTrack is class-agnostic)
  - Ball speed >1.0 = guaranteed shot at any position
  - Ball speed >0.50 = shot if near goal (y<0.25 or y>0.75) or near sides (x<0.15 or x>0.85)
  - Inter-frame gap must be <=2.0s (skip different plays)
  - ON_TARGET: ball in near-goal zone (y<0.25 or y>0.75)
  - OFF_TARGET: everything else that passes speed threshold
- **Event window:** -1.0s before ball start, +3.0s after ball peak
- **Merge logic:** Shots <1.0s apart merged
- **Confidence:** 0.55 + speed * 0.5, capped at 0.90
- **Metadata:** `ball_speed`, `ball_vx` (signed), `ball_origin_x`
- **Known issue:** No spatial filter — fires on clearances, long passes, goal kicks anywhere on pitch

#### DRIBBLE_SEQUENCE *(excluded from reels)*
- **Detector:** `event_classifier._detect_dribbles()`
- **Conditions:**
  - Player track with >=10 detections
  - Sustained speed >0.08 body-widths/s
  - Duration >=1.5 seconds
- **Confidence:** 0.67 (fixed)
- **Note:** Detected but excluded from default reel presets to prevent flooding. Can be included in custom ReelSpecs.

#### TACKLE *(excluded from default reels)*
- **Detector:** `event_classifier._detect_tackles()`
- **Conditions:**
  - Two player tracks in same frame
  - Distance <0.05 normalized units (very close proximity)
- **Merge logic:** Tackles <3.0s apart merged
- **Confidence:** 0.65 (fixed)
- **Note:** Detected but excluded from default reel presets. Can be included in custom ReelSpecs.

### ML-Only Events (require VideoMAE ActionClassifier)

These events have NO heuristic fallback (except GOAL which also has a ball-in-net heuristic):

| Event | VideoMAE Label | Notes |
|-------|---------------|-------|
| GOAL | goal_celebration | Highest threshold (0.85). Also detected by ball-in-net heuristic. |
| NEAR_MISS | near_miss | Shot hits post/bar or goes just wide |
| PENALTY | penalty_kick | Most lenient threshold (0.60) |
| FREE_KICK_SHOT | free_kick | Direct free kick attempts |
| PUNCH | punch | GK punches ball away |

## Detection Paths

### GK-First (Default: `USE_BALL_TOUCH_DETECTOR=false`)
1. Identify GK by jersey color uniqueness + edge heuristic
2. Look for ball contacts near identified GK
3. Classify by GK motion (diving vs standing) and ball trajectory

### Ball-First (Optional: `USE_BALL_TOUCH_DETECTOR=true`)
1. Build ball trajectory from all tracks (dedup + median smoothing)
2. Find touch moments: direction changes, speed changes, disappearances (zone-aware thresholds near goal)
3. Attribute each touch to nearest player
4. Classify by jersey color → keeper event or skip (requires 0.15 color margin over all other jersey classes + midfield position gate rejects x=0.30–0.70)
5. Compute GK vertical velocity → upgrade SHOT_STOP_STANDING to SHOT_STOP_DIVING if >2.0 body-heights/s
6. Smart endpoint extension: extend clips to next touch / out-of-bounds / ball disappearance
7. Ball-in-net detection: fast ball at x<0.03 or x>0.97 → GOAL event
8. Corner kick detection: (a) spatial — ball near frame edge + players, (b) post-save restart — save → gap → stationary → kicked with players
9. Post-filter: reclassify dead-ball collections as GOAL_KICKs

## Clip Parameters

Clip padding is now per-event-type via `EVENT_TYPE_CONFIG` (see table at top). The old plugin system (`src/reel_plugins/`) is still present for backward compatibility but is no longer used by the worker pipeline. The worker uses `compute_clips_v2()` which looks up padding from `EVENT_TYPE_CONFIG` for each event.

### GK Classification Guards

The ball touch detector applies four guards to prevent false GK identification:

1. **Color margin (0.15)** — `sim_team_gk` must exceed every other jersey similarity (opponent GK, team outfield, opponent outfield) by at least 0.15. Prevents blue/teal confusion where blue opponent outfield (H≈120) matches teal GK (H≈80) at sim≈0.73.
2. **Midfield position gate (0.30–0.70)** — any player in the midfield band is rejected as GK regardless of jersey color match. Goalkeepers don't operate at midfield.
3. **Minimum similarity (0.60)** — `gk_color_min_similarity` default raised from 0.55 to 0.60 for baseline defense.
4. **Goal kick reclassification guard (0.78)** — dead-ball→goal_kick reclassification requires `sim_team_gk >= 0.78`. The dead-ball pattern (stationary ball then kicked) can happen anywhere on the pitch, so a stricter threshold prevents FPs from defenders collecting clearances.

All keeper plugins apply a three-stage quality filter:
1. **Reel-level sim gate (0.75)** — events with sim_team_gk < 0.75 are excluded before spatial filtering. This catches pre-game FPs and weak jersey matches where the color margin check passed but absolute similarity is too low. Penalty events (ML-detected, no jersey color) are exempt.
2. **Midfield gate** — events with bounding_box center_x in the middle band (0.35–0.65) are rejected outright; GKs don't operate in midfield.
3. **Majority-vote side filter** — among the remaining outer-third events, only events with confidence >= 0.75 AND sim_team_gk >= 0.77 (strong jersey color match) participate as voters. Goal kicks are excluded from voting (they inherit bounding_box from the original save/catch). This prevents low-confidence or weak-color-match false positives from outnumbering real events and flipping the side determination. If one side dominates (>=55% of >=2 qualified voters), events on the opposite side are removed.

Clip deduplication uses temporal IoU threshold of 0.3 (previously 0.5) to catch near-duplicate clips from chunk boundaries and multiple plugins with different padding.

## Zone-Aware Detection Thresholds

The ball touch detector uses relaxed thresholds near the goal (x < 0.15 or x > 0.85) to catch deflection saves and goals that would be missed with standard thresholds:

| Parameter | Standard (midfield) | Near-goal |
|-----------|-------------------|-----------|
| Direction change | 40° | 25° |
| Speed change ratio | 0.50 (50% drop) | 0.30 (30% drop) |
| Min touch speed | 0.20 | 0.10 (0.5x) |

This is critical for detecting glancing deflections, fingertip saves, and slow-ball saves near the goal line.

## Smart Clip Endpoints

Clips are extended beyond the raw event window to capture the full play outcome. The extension scans ball trajectory forward for one of these triggers:

- **Out of bounds**: ball at x<0.01 or x>0.99 or y<0.01 or y>0.99
- **Next touch**: direction change >20° or speed drop >30% (relaxed from 30°/40%)
- **Ball disappears**: gap >=5 frames
- **Max extension reached**: per-type cap

| Event Type | Max Extension |
|-----------|--------------|
| DISTRIBUTION_SHORT/LONG | 10s |
| GOAL_KICK | 12s |
| SHOT_STOP_*, CATCH, PUNCH | 5s |
| CORNER_KICK | 8s |
| ONE_ON_ONE | 8s |

Events not in this list (e.g. SHOT_ON_TARGET) are not extended. Extended events have `metadata.endpoint_extended=true` and `metadata.endpoint_reason`.

## Known Issues

1. **Shot detection has no spatial awareness** — fires on clearances, long passes, and goal kicks because it only checks ball velocity, not position relative to opposing goal or direction toward goal.
2. **Keeper distribution/buildout not captured** — GK passes and throws after collecting the ball are classified as DISTRIBUTION_SHORT/LONG by the GK Detector's velocity method, but this is unreliable. The BallTouchDetector has distribution detection disabled by default (`detect_distributions=false`).
