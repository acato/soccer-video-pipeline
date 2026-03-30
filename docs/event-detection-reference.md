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

Five presets are available: `"keeper"` (all GK event types + goals), `"highlights"` (shot/goal types), `"goal_kicks"` (goal kicks only), `"corner_kicks"` (corner kicks only), and `"goals"` (goals only). Users can also build custom reels by selecting individual event types.

Each event type has a per-type configuration in `EVENT_TYPE_CONFIG` (`src/detection/models.py`):

| Event Type | Category | Pre-pad | Post-pad | Max clip | Min confidence | GK event? |
|-----------|----------|---------|----------|----------|---------------|-----------|
| SHOT_STOP_DIVING | goalkeeper | 8.0s | 2.0s | 25s | 0.75 | Yes |
| SHOT_STOP_STANDING | goalkeeper | 8.0s | 2.0s | 25s | 0.70 | Yes |
| PUNCH | goalkeeper | 8.0s | 2.0s | 25s | 0.65 | Yes |
| CATCH | goalkeeper | 8.0s | 2.0s | 25s | 0.70 | Yes |
| GOAL_KICK | goalkeeper | 10.0s | 15.0s | 45s | 0.65 | Yes |
| DISTRIBUTION_SHORT | goalkeeper | 1.0s | 2.0s | 20s | 0.65 | Yes |
| DISTRIBUTION_LONG | goalkeeper | 1.0s | 2.0s | 20s | 0.68 | Yes |
| ONE_ON_ONE | goalkeeper | 3.0s | 2.0s | 30s | 0.75 | Yes |
| CORNER_KICK | goalkeeper | 5.0s | 12.0s | 35s | 0.65 | Yes |
| PENALTY | goalkeeper | 8.0s | 2.0s | 25s | 0.60 | Yes |
| SHOT_ON_TARGET | highlights+keeper | 3.0s | 5.0s | 30s | 0.70 | Yes |
| SHOT_OFF_TARGET | highlights | 3.0s | 5.0s | 30s | 0.65 | No |
| GOAL | highlights+keeper | 10.0s | 15.0s | 60s | 0.50 | Opponent goals |
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
1. **Reel-level sim gate** — non-save events require sim_team_gk >= 0.75; save events (SHOT_STOP_DIVING/STANDING, PUNCH, CATCH, PENALTY) require sim_team_gk >= 0.60 (lowered from 0.70 since speed gates now filter routine play). Penalty and corner kick events are exempt. This catches pre-game FPs and weak jersey matches.
2. **Midfield gate** — events with bounding_box center_x in the middle band (0.35–0.65) are rejected outright; GKs don't operate in midfield.
3. **Majority-vote side filter** — among the remaining outer-third events, only events with confidence >= 0.75 AND sim_team_gk >= 0.77 (strong jersey color match) participate as voters. Goal kicks are excluded from voting (they inherit bounding_box from the original save/catch). This prevents low-confidence or weak-color-match false positives from outnumbering real events and flipping the side determination. If one side dominates (>=55% of >=2 qualified voters), events on the opposite side are removed. **Halftime switch enforcement**: teams switch sides at halftime, so if both halves vote the same side, the half with fewer voters is flipped to the opposite. If only one half has a clear majority, the other half is inferred as the opposite side.

Clip deduplication uses temporal IoU threshold of 0.3 (previously 0.5) to catch near-duplicate clips from chunk boundaries and multiple plugins with different padding.

## Zone-Aware Detection Thresholds

The ball touch detector uses relaxed thresholds near the goal (x < 0.15 or x > 0.85) to catch deflection saves and goals that would be missed with standard thresholds:

| Parameter | Standard (midfield) | Near-goal |
|-----------|-------------------|-----------|
| Direction change | 40° | 25° |
| Speed change ratio | 0.50 (50% drop) | 0.30 (30% drop) |
| Min touch speed | 0.20 | 0.10 (0.5x) |

This is critical for detecting glancing deflections, fingertip saves, and slow-ball saves near the goal line.

## Save Speed Gates

Save-like ball touches (speed_drop, direction_change, ball_caught) are subject to additional speed gates to reduce false positives from routine play:

| Gate | Threshold | Purpose |
|------|-----------|---------|
| Pre-speed minimum | ball_pre_speed >= 0.40/s | Saves stop SHOTS, which are fast. Rejects slow-ball FPs (collections, traps, back-passes) |
| Speed-drop ratio | post/pre < 0.40 | For `speed_drop` reason only. Requires dramatic deceleration (60%+ loss), not gentle slowdowns |

Speed metadata (`ball_pre_speed`, `ball_post_speed`, `speed_ratio`) is stored in event metadata for post-hoc analysis.

## Goal-Area GK Override

When the nearest player is an attacker (e.g. during a parry), the detector overrides attribution to a GK-colored player in the goal area. Constraints:
- Player must be in outer goal area (px < 0.22 or px > 0.78)
- GK candidate must be on same side as ball (both left quarter or both right quarter)
- Minimum sim_team_gk >= 0.70
- GK sim must exceed nearest player's sim by >= 0.03

The previous goal-area **fallback** in `_classify_touch` (which bypassed all color margins at frame edges x<0.15/x>0.85) has been removed — it was a major source of false positives with panning cameras.

## Save-Context Fallback

When the normal color margin check fails (e.g., teal GK jersey looks blue under certain lighting), a save-context fallback can still classify the touch as a GK event if ALL of these conditions are met:

1. **Save reason** — touch reason is `speed_drop`, `direction_change`, or `ball_caught`
2. **Fast ball** — `ball_pre_speed >= 0.40` (passed save speed gate, confirming a shot)
3. **Goal area** — player position x < 0.22 or x > 0.78
4. **Minimum similarity** — sim_team_gk >= `gk_color_min_similarity` (0.60)
5. **Better than opponent GK** — sim_team_gk > sim_opp_gk

This recovers saves where the GK's jersey color is ambiguous (e.g., teal vs blue under stadium lighting) but positional and ball-speed evidence strongly confirm a save. The speed gate provides sufficient FP protection — only actual shots trigger this path.

## Smart Clip Endpoints

Clips are extended beyond the raw event window to capture the full play outcome. The extension scans ball trajectory forward for one of these triggers:

- **Out of bounds**: ball at x<0.01 or x>0.99 or y<0.01 or y>0.99
- **Next touch**: direction change >20° or speed drop >30% (relaxed from 30°/40%)
- **Ball disappears**: gap >=5 frames
- **Max extension reached**: per-type cap

| Event Type | Max Extension |
|-----------|--------------|
| DISTRIBUTION_SHORT/LONG | 10s |
| GOAL_KICK | 15s |
| SHOT_STOP_*, CATCH, PUNCH | 5s |
| CORNER_KICK | 8s |
| ONE_ON_ONE | 8s |

Events not in this list (e.g. SHOT_ON_TARGET) are not extended. Extended events have `metadata.endpoint_extended=true` and `metadata.endpoint_reason`.

## VLM (Vision Language Model) Classifier

Optional post-detection verification using Claude Sonnet to replace heuristic jersey-color classification for GK save events. Enabled via `VLM_ENABLED=true`.

### Why

The heuristic jersey-color approach (HSV comparison) is unreliable when team colors are spectrally close (e.g., teal GK vs blue opponents). After multiple rounds of threshold tuning, it still produces ~50% false positives and misses ~30% of real saves. The VLM can visually identify the goalkeeper and the save action directly from video frames.

### Architecture

```
BallTouchDetector → candidate events (relaxed thresholds when VLM_RELAX_COLOR_MARGINS=true)
     ↓
VLMClassifier.filter_events(candidates)
     ↓  FFmpeg extracts 3 frames per event (-0.5s, center, +0.5s)
     ↓  Claude Sonnet analyzes each frame set
     ↓  Returns keep/reject + confidence + reasoning
     ↓
Segmentation → Assembly
```

Runs **post-detection, pre-segmentation** in the worker pipeline. When enabled, it **replaces** both the spatial filter and the sim gate for GK events.

### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `VLM_ENABLED` | `false` | Toggle VLM classification on/off |
| `ANTHROPIC_API_KEY` | (empty) | Required when VLM_ENABLED=true |
| `VLM_MODEL` | `claude-sonnet-4-20250514` | Claude model for classification |
| `VLM_FRAME_WIDTH` | `640` | Resize width for extracted frames |
| `VLM_MIN_CONFIDENCE` | `0.6` | Minimum VLM confidence to keep event |
| `VLM_RELAX_COLOR_MARGINS` | `false` | Relax BallTouchDetector thresholds |

### Relaxed Detection Thresholds

When `VLM_RELAX_COLOR_MARGINS=true`, the BallTouchDetector lowers thresholds to produce more candidates (higher recall, lower precision) since the VLM is the quality gate:

| Parameter | Normal | VLM-relaxed |
|-----------|--------|-------------|
| `_SAVE_PRE_SPEED_MIN` | 0.40 | 0.25 |
| `_COLOR_MARGIN_OF` | 0.12 | 0.05 |

### Prompt Design

The VLM receives:
- 3 frames as JPEG images (640px wide, ~50-100KB each)
- Match context: team names, jersey colors for all 4 roles (team GK, opponent GK, team outfield, opponent outfield)
- Event metadata: ball position, ball speed (pre/post), event type
- Structured criteria: is the GK visible, in goal area, performing a save action

The VLM returns structured JSON: `{is_gk_save: bool, confidence: 0-1, reasoning: str}`.

### Error Handling

- **Fail-open**: if frame extraction or API call fails, the event is kept (not dropped)
- **Parse errors**: if the JSON response can't be parsed, event is kept with confidence=0.0
- VLM verdict + reasoning stored in `event.metadata` (vlm_is_save, vlm_confidence, vlm_reasoning, vlm_model)

### Cost Estimate

For a typical match (~17 candidate save events):
- ~17 API calls × 3 frames × ~80KB/frame = ~4MB total image data
- ~17 × 800 input tokens + 50 output tokens per call
- Processing time: ~51s sequential

### Source

`src/detection/vlm_classifier.py` — `VLMClassifier` class.

## vLLM Classifier (Dead-Ball Restarts)

Optional post-detection classification using a vLLM-hosted vision model (Qwen3-VL) for dead-ball restart events (goal kicks, corner kicks). Enabled via `VLLM_ENABLED=true`.

### Why

Ball-direction heuristics are too brittle for classifying restart types — they break across camera angles, zoom levels, and field positions. A vision-language model with native video understanding can see ball motion, player positions, and scene context (GK placing ball on 6-yard box, corner flag, players in wall, etc.) across time and classify accurately.

### Architecture

```
BallTouchDetector._find_trajectory_gaps()
     ↓  Ball disappears ≥1.5s → GapCandidate list
     ↓
RestartClassifier.classify_gaps(candidates, fps)
     ↓  FFmpeg extracts 12-15s video clip per gap (scaled to 640px)
     ↓  vLLM classifies via OpenAI-compatible API: goal_kick / corner_kick / throw_in / free_kick / goal / other
     ↓  Returns structured Event list
     ↓
Worker extends event list → Segmentation → Assembly
```

Runs **post-detection, pre-segmentation** in the worker pipeline. Classified events are appended to the event log alongside events from the existing BallTouchDetector.

### Two-Stage Pipeline

**Stage 1 — Candidate Generation:** Scan ball trajectory for gaps where the ball disappears for ≥1.5s and reappears within 30s. These gaps reliably indicate dead-ball situations. No classification at this stage.

**Stage 2 — vLLM Classification:** Extract a video clip around each gap (5s before + gap + 8s after). Send to vLLM as a base64 `video_url` via the OpenAI-compatible `/v1/chat/completions` endpoint with match context (jersey colors, ball positions). Parse structured JSON response.

### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `VLLM_ENABLED` | `false` | Toggle vLLM classification on/off |
| `VLLM_URL` | `http://10.10.2.222:8000` | vLLM server URL (OpenAI-compatible) |
| `VLLM_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | Model name as registered in vLLM |
| `VLLM_CLIP_PRE_SEC` | `5.0` | Seconds before gap in clip |
| `VLLM_CLIP_POST_SEC` | `8.0` | Seconds after gap in clip |
| `VLLM_MIN_CONFIDENCE` | `0.5` | Minimum confidence to keep event |

### Sim Gate Exemption

GOAL_KICK events (from vLLM or heuristic) are exempt from the jersey color sim gate in `spatial_filter.py`, alongside PENALTY and CORNER_KICK. Model classification is the quality gate — no jersey color check needed.

### Goal Kick Clip Padding

GOAL_KICK events use extended padding to capture the full sequence: cause (shot going wide / deflection) → dead ball → GK placement → kick → first touch.

- Pre-pad: **10.0s** (captures the shot/cause before ball went out)
- Post-pad: **3.0s** (captures kick trajectory + first touch)
- Max clip: **35.0s** (full sequence)
- Smart endpoint extension: **15.0s** (extends to first touch after kick)

### Error Handling

- **Fail-open**: if clip extraction or API call fails, the gap is skipped (not crashed)
- **Parse errors**: malformed JSON responses are logged and skipped
- Classification metadata stored in `event.metadata` (vllm_event_type, vllm_confidence, vllm_reasoning, vllm_model)

### Cost

Self-hosted via vLLM — no per-request API cost. Requires GPU server running the model.

### Source

`src/detection/restart_classifier.py` — `RestartClassifier` class.
`src/detection/ball_touch_detector.py` — `_find_trajectory_gaps()` method.

## Chunk Tagger (vLLM Vision Model)

When `VLLM_ENABLED=true` and `USE_NULL_DETECTOR=true`, the worker uses `ChunkTagger` (`src/detection/chunk_tagger.py`) instead of YOLO-based detection. The tagger splits the video into overlapping chunks (default 45s at 4 FPS, 15s overlap), sends each to Qwen3-VL via vLLM, and the model tags events using negative-prompt inversion (must list reasons each candidate is NOT a given event type before tagging).

### Tagged Event Types

| Model Tag | Pipeline EventType | GK Event? | Observation Chain |
|-----------|-------------------|-----------|-------------------|
| `goal` | GOAL | If opponent scores | Shot → net → celebration → teams to halves → kickoff. Clip: shot to celebration. |
| `penalty` | PENALTY | If opponent takes it | Box clears (only ref+GK+shooter) → ball set → shot. |
| `free_kick` | FREE_KICK_SHOT | No | Ball placed down → kicked with no one approaching. |
| `shot` | SHOT_ON_TARGET | Yes | Ball struck toward goal → goes out past back line (miss). Only when no GK touch or goal. |
| `corner_kick` | CORNER_KICK | Yes | Ball placed on corner arc → kicked into penalty area. |
| `goal_kick` | GOAL_KICK | Yes | Ball placed on ground in six-yard box → kicked by GK/defender → no opponents in box. NOT a throw-in. |
| `catch` | CATCH | Yes | GK grabs ball and holds it. |
| `save` | SHOT_STOP_DIVING | Yes | Shot → keeper touch/deflection → ball out for corner. |

### Key Rules in Prompt

- All prompts use **negative-prompt inversion**: for each candidate event, the model lists reasons it is NOT that event type, then tags only if no disqualifying reason applies
- GOAL requires at least TWO of: (a) ball visibly in net, (b) celebration, (c) center-circle setup. "Ball disappears toward goal" alone is never enough.
- SAVE always ends with a corner kick; if keeper holds the ball = CATCH
- SHOT is only tagged when the outcome is a miss (not goal/save/catch)
- Throw-ins are NOT goal kicks — a throw-in is a player holding the ball overhead at the sideline
- KICKOFF only happens from the exact center circle after a goal or at halftime — not after any other stoppage
- "team" field = team performing the action (scoring team, GK's team, kicking team)
- Special fog/low-visibility rule: ball disappearing near the goal does NOT equal a goal; saves, misses, and goals all look identical in poor visibility

### Kickoff-Based Goal Recovery (Two-Pass)

Fast goals (breakouts, quick shots) can be invisible even at 4 FPS. The tagger detects these via kickoffs:

1. **Pass 1 (4 FPS, 45s chunks)**: Tags all events including `kickoff` (center-circle restart). All prompts use negative-prompt inversion. Goals-only mode uses a focused prompt that requires at least TWO of: ball in net, celebration, center-circle setup.
2. **Orphan detection**: A kickoff not preceded by a detected goal within 90s, and not halftime (>120s event gap), implies a missed goal
3. **Pass 2 (8 FPS)**: Extracts the 60s before the orphan kickoff in 15s chunks at 8 FPS, sends to vLLM with a focused "find the goal" prompt that also uses disqualification reasoning
4. Goal events from the rescan are added to the event list
5. **Goal inference**: If the rescan still finds nothing, a synthetic goal is created anchored to the last shot within 120s, or falling back to ko_t-30s

Config: `VLLM_CHUNK_DURATION_SEC` (default 45), `VLLM_CHUNK_FPS` (default 4), `VLLM_CHUNK_OVERLAP_SEC` (default 15), `VLLM_RESCAN_FPS` (default 8), `VLLM_RESCAN_PRE_SEC` (default 60).

### Deduplication

Events from overlapping chunks are deduplicated: same event type within 10s proximity → keep higher confidence. Cross-type dedup then removes shots within 15s of a goal (the shot was superseded by the goal).

## Known Issues

1. **Shot detection has no spatial awareness** — fires on clearances, long passes, and goal kicks because it only checks ball velocity, not position relative to opposing goal or direction toward goal.
2. **Keeper distribution/buildout not captured** — GK passes and throws after collecting the ball are classified as DISTRIBUTION_SHORT/LONG by the GK Detector's velocity method, but this is unreliable. The BallTouchDetector has distribution detection disabled by default (`detect_distributions=false`).
