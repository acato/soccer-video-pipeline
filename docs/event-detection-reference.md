# Event Detection Reference

How each of the 16 event types is currently detected, including thresholds, confidence values, and reel assignments.

## Event Detection Summary

| # | Event Type | Detector | Key Conditions | Confidence | Reel | Threshold |
|---|-----------|----------|---------------|------------|------|-----------|
| 1 | SHOT_STOP_DIVING | GK Detector / Ball Touch | GK vertical velocity >2.5 body-heights/s OR ball direction change >45deg near GK (ARM_REACH=0.06) | 0.50-0.90 | keeper | 0.75 |
| 2 | SHOT_STOP_STANDING | GK Detector / Ball Touch | GK vertical velocity 1.0-2.5 body-heights/s + ball proximity <0.15 OR ball speed drop >60% near GK | 0.50-0.80 | keeper | 0.70 |
| 3 | PUNCH | ActionClassifier (ML) | VideoMAE temporal action recognition only. No heuristic fallback. | model-dependent | keeper | 0.65 |
| 4 | CATCH | Ball Touch Detector | Ball trajectory gap (disappears for >=5 frames) with incoming speed >0.30 OR end-of-trajectory catch | 0.60-0.90 | keeper | 0.70 |
| 5 | GOAL_KICK | Ball Touch (reclassify) | Dead-ball reclassification: save/catch where ball was stationary (speed <0.05) for >=1s, then kicked (speed spike). GK center_y >0.75. | inherited | keeper | 0.65 |
| 6 | DISTRIBUTION_SHORT | GK Detector | GK stationary (vel <1.0 body-widths/s) then moves (vel >1.5). GK center_y 0.55-0.75. | 0.55-0.90 | keeper | 0.65 |
| 7 | DISTRIBUTION_LONG | GK Detector | Same as SHORT but GK center_y <=0.55 (further from goal line). | 0.55-0.90 | keeper | 0.68 |
| 8 | ONE_ON_ONE | GK Detector | GK deviation from goal line >0.3 body-heights for >=3 consecutive frames. | 0.65-0.88 | keeper + highlights | 0.75 |
| 9 | SHOT_ON_TARGET | Event Classifier | Ball speed >1.0 (any position) OR >0.50 near goal (y<0.25/y>0.75) or sides (x<0.15/x>0.85). Near-goal required for ON_TARGET. | 0.55-0.90 | highlights | 0.70 |
| 10 | SHOT_OFF_TARGET | Event Classifier | Same velocity conditions as ON_TARGET but ball NOT in goal-scoring area. | 0.55-0.90 | highlights | 0.65 |
| 11 | GOAL | ActionClassifier (ML) | VideoMAE only. No heuristic fallback. | model-dependent | highlights | 0.85 |
| 12 | NEAR_MISS | ActionClassifier (ML) | VideoMAE only. No heuristic fallback. | model-dependent | highlights | 0.70 |
| 13 | PENALTY | ActionClassifier (ML) | VideoMAE only. No heuristic fallback. | model-dependent | highlights + keeper | 0.60 |
| 14 | FREE_KICK_SHOT | ActionClassifier (ML) | VideoMAE only. No heuristic fallback. | model-dependent | highlights | 0.65 |
| 15 | DRIBBLE_SEQUENCE | Event Classifier | Player track >=10 detections + speed >0.08 body-widths/s sustained >=1.5s. | 0.67 | *(excluded)* | 0.65 |
| 16 | TACKLE | Event Classifier | Two player tracks within distance <0.05 normalized units in same frame. | 0.65 | *(excluded)* | 0.65 |

## Detailed Breakdown

### Goalkeeper Events (Keeper Reel)

#### SHOT_STOP_DIVING
- **Detectors:** `GoalkeeperDetector._detect_saves()`, `BallTouchDetector`
- **Conditions:**
  - GK vertical velocity >2.5 body-heights per second (unmistakable dive), OR
  - Ball within ARM_REACH=0.06 of GK bbox edge + trajectory change (direction >45deg or speed drop >60% or ball caught)
- **Event window:** +/-0.5s (+ keeper reel padding +/-1.5s = +/-2s clips)
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
- **Confidence:** 0.60 + sim_team_gk * 0.3 (range 0.60-0.90)

#### GOAL_KICK
- **Detector:** `BallTouchDetector._reclassify_dead_ball_collections()`
- **Conditions:**
  - Post-filter reclassification of save/catch events
  - Ball was stationary (speed <0.05) for >=1s after GK touched it
  - Then ball kicked (speed spike)
  - GK bbox center_y >0.75 (goal line area)
  - Pre-filter rejects slow pickups (incoming speed <0.30)

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
- **Note:** Detected but excluded from all reels (EVENT_REEL_MAP set to `[]`) to prevent flooding

#### TACKLE *(excluded from reels)*
- **Detector:** `event_classifier._detect_tackles()`
- **Conditions:**
  - Two player tracks in same frame
  - Distance <0.05 normalized units (very close proximity)
- **Merge logic:** Tackles <3.0s apart merged
- **Confidence:** 0.65 (fixed)
- **Note:** Detected but excluded from all reels (EVENT_REEL_MAP set to `[]`)

### ML-Only Events (require VideoMAE ActionClassifier)

These events have NO heuristic fallback and will only be detected if the VideoMAE model is loaded:

| Event | VideoMAE Label | Notes |
|-------|---------------|-------|
| GOAL | goal_celebration | Highest threshold (0.85) |
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
2. Find touch moments: direction changes, speed changes, disappearances
3. Attribute each touch to nearest player
4. Classify by jersey color -> keeper event or skip
5. Post-filter: reclassify dead-ball collections as GOAL_KICKs

## Reel Plugin Clip Parameters

| Plugin | Events | Pre-pad | Post-pad | Max clip |
|--------|--------|---------|----------|----------|
| KeeperSavesPlugin | SHOT_STOP_*, PUNCH, CATCH, ONE_ON_ONE, PENALTY | 2.0s | 1.5s | 15s |
| KeeperDistributionPlugin | GOAL_KICK, DISTRIBUTION_* | 1.0s | 3.0s | 10s |
| HighlightsShotsPlugin | SHOT_ON/OFF_TARGET, GOAL, NEAR_MISS, PENALTY, FREE_KICK_SHOT | 3.0s | 5.0s | 90s |

## Known Issues

1. **Shot detection has no spatial awareness** — fires on clearances, long passes, and goal kicks because it only checks ball velocity, not position relative to opposing goal or direction toward goal.
2. **Keeper distribution/buildout not captured** — GK passes and throws after collecting the ball are classified as DISTRIBUTION_SHORT/LONG by the GK Detector's velocity method, but this is unreliable. The BallTouchDetector has distribution detection disabled by default (`detect_distributions=false`).
