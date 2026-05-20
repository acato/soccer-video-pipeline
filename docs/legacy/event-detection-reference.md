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

## Current Detection Architecture: Motion-First Pipeline

**Status**: Active (replaced audio-first and YOLO-heuristic pipelines)

The pipeline uses a three-phase motion-first architecture optimised for VEO-style sideline cameras where audio is unreliable and COCO YOLO cannot detect the ball at 50m+ distance.

### Phase 1: Motion Scan (Primary Trigger)

Dense frame-differencing at 0.5s intervals with **adaptive sliding-window threshold**.

- **Sample interval**: 0.5s (every 15 frames at 30 FPS)
- **Resolution**: 320×180 grayscale (downscaled for speed)
- **Threshold**: 10-minute sliding window, local mean + 0.7σ, floor at 80% of global mean
- **Spike detection**: Contiguous above-threshold segments (using per-sample adaptive threshold)
- **Merge window**: 8s (spikes within 8s merged, keep highest-confidence peak)
- **Confidence**: Log-scale `0.3 + 0.35 × log₂(peak/threshold) + duration_bonus` where duration_bonus = min(spike_dur/10, 0.15), capped at 0.85 before bonus.
- **Clip window**: 5s pre / 15s post (VLM classification uses 3s pre / 3s post centred on event)

Source: `src/detection/visual_candidate.py` — `VisualCandidateGenerator.motion_scan()`

### Phase 2: Audio Boost (Supplementary)

Audio detection runs independently, then boosts co-located motion candidates. Audio never gates candidates — it only increases confidence.

- **Co-location window**: ±5s
- **Whistle boost**: +0.15 confidence
- **Energy surge boost**: +0.10 confidence
- **Whistle detection**: bandpass 2-4 kHz, min 0.2s duration
- **Surge detection**: RMS energy > mean + 3.5σ above rolling mean

Source: `src/detection/visual_candidate.py` — `VisualCandidateGenerator.boost_with_audio()`

### Phase 2b: Match Structure Detection

Detects halftime break (largest gap >5 min between candidates) and estimates match boundaries. Filters out pre-game, halftime, and post-match noise candidates.

- **Halftime gap minimum**: 5 minutes
- **Match end estimate**: halftime_end + 55 min (or game_start + 100 min fallback)

Source: `src/detection/pipeline.py` — `DetectionPipeline._detect_match_structure()`

### Phase 2c: Audio Gap Fill

Promotes orphan audio cues (whistles not near any motion candidate) to standalone candidates. Fills gaps where motion scan missed events but audio detected whistles.

- **Orphan window**: ±8s (audio cue with no motion candidate within 8s)
- **Only whistles**: energy surges excluded (too noisy standalone)
- **Confidence**: 0.45 (lower than motion — needs VLM to confirm)

Source: `src/detection/pipeline.py` — `DetectionPipeline._audio_gap_fill()`

### Phase 2d: Spot-Check Probes

Inserts VLM probes in temporal gaps >3 min between candidates. Insurance against motion+audio blind spots.

- **Gap threshold**: 3 minutes
- **Max probes**: 15 per match
- **Halftime excluded**

Source: `src/detection/pipeline.py` — `DetectionPipeline._spot_check_probes()`

### Phase 3: VLM Classification (Single-Pass)

Each candidate is classified by a vision-language model using a single-pass prompt with images.

The prompt is calibrated for sideline camera limitations:
- Explicitly notes the camera cannot see GK hand contact at 50m distance
- Uses post-event restart patterns as a signal (kickoff after = goal, goal_kick after = shot or save)
- Save (catch) detection: looks for the **result state** — GK holding/carrying the ball after a shot — rather than the action itself. Catches are static poses visible even at distance.
- Save (parry) detection is handled structurally by Phase 3c.5 (shot + corner = parry), not by the VLM
- Provides visual cues for each event type

**Key parameters**:
- Frame extraction: 2 FPS, 768px width, JPEG quality 5
- Max frames: 24 (12s window); audio-aware centering shifts window before whistle
- Claude goal verification: saves/free kicks with kickoff evidence re-checked via Claude API
- Max tokens: 500, Temperature: 0 (deterministic)
- VLM candidate cap: 120 (time-distributed sampling with overflow fill)
- Min confidence: 0.5 to confirm

**Strict goal rules**: Classify as "goal" ONLY if celebration visible OR kickoff restart at center circle. "Ball near goal" alone is never enough.

Source: `src/detection/vlm_verifier.py` — `VLMVerifier`

### Phase 3a.5: Save Reclassification (Post-VLM)

Recovers saves that the VLM classified as "goal_kick". The VLM sometimes sees a shot → save → goal kick sequence and classifies by the visible restart rather than the meaningful action.

**Rule**: If a VLM verdict is `goal_kick` AND the reasoning mentions save-related keywords (save, stopped, blocked, parried, caught, pushed away, tipped, denied, kept out), reclassify to `SHOT_STOP_DIVING`.

**Note**: With the updated prompt (which requires visible GK action for "save"), this phase now mainly catches edge cases where the VLM describes a save in reasoning but classifies the restart instead.

Source: `src/detection/pipeline.py` — `DetectionPipeline._save_reclassification()`

### Phase 3a.6: Shot Reclassification (Post-VLM)

The VLM often *sees* a shot (describes it in reasoning text) but classifies the overall sequence as "none" because no clear restart follows. This phase recovers those shots.

**Rule**: Scan rejected verdicts for positive shot keywords (e.g. "shot toward goal", "fired at goal", "went wide", "over the bar", "blocked", "deflected"). Exclude false positives where the reasoning just says "no shot" or "no clear shot". Reclassify matching verdicts as `SHOT_ON_TARGET` or `SHOT_OFF_TARGET` (based on keywords like "wide", "over the bar", "blocked", "deflected").

**Confidence**: Reclassified shots get `original_confidence × 0.7` to reflect lower certainty.

Source: `src/detection/pipeline.py` — `DetectionPipeline._shot_reclassification()`

### Phase 3b: Goal Inference (Post-VLM)

After the main VLM classification, a goal inference pass uses temporal kickoff patterns to both **confirm real goals** and **filter false positive goals**. This phase has four safeguards against false positives:

**Step 0 — Pre-game exclusion**: Identify the first confirmed kickoff in the match (the opening whistle). Any shot/save that occurs BEFORE this kickoff is excluded from goal inference entirely — it's a pre-game event and the opening kickoff is NOT a post-goal restart.

**Step 1 — Kickoff rescan**: For each confirmed shot/save AND each VLM-classified goal, look for unclassified motion candidates 20-90s later. If no motion candidate exists in the window, extract frames directly at dense offsets (+30s, +45s, +60s, +75s, +90s). Send to VLM with a focused kickoff-specific prompt. The opening kickoff is excluded from matching — only kickoffs detected AFTER the shot count.

**Step 2 — Shot→goal upgrade**: If a non-opening kickoff is confirmed after a shot/save → provisionally upgrade to GOAL.

**Step 3 — Celebration probe**: Each provisional goal is verified with a secondary VLM probe: "is there a celebration, ball in net, or dejected opponents?" using frames from shot+2s to shot+12s. Goals that fail the celebration probe are downgraded back to `SHOT_ON_TARGET` with 0.7× confidence. This catches false positives where a shot coincidentally precedes a false-positive kickoff.

**Step 4 — VLM goal→shot downgrade**: VLM-classified goals WITHOUT a confirmed kickoff are downgraded to `SHOT_ON_TARGET` with 0.85× confidence penalty.

**Step 5 — Goal dedup**: Merge goals within 240s (4 min), keeping higher confidence.

- **Kickoff prompt**: Binary question ("is this a center-circle kickoff?")
- **Celebration prompt**: Binary question ("is there a goal celebration / ball in net?")
- **Direct probes**: Dense frame extraction at +30/45/60/75/90s after shot
- **Parameters**: `_KICKOFF_RESCAN_MIN_GAP=20s`, `_KICKOFF_RESCAN_MAX_GAP=90s`, `_GOAL_DEDUP_WINDOW=240s`
- **Diagnostics**: `kickoff_rescan.jsonl`, `goal_celebration_probe.jsonl`

Source: `src/detection/pipeline.py` — `DetectionPipeline._goal_inference()`

### Phase 3c: Set-Piece Inference (Post-VLM)

After shots/saves, rescan for restarts (corners, goal kicks, throw-ins) that the main VLM pass missed.

**Trigger**: Each confirmed shot or save (SHOT_ON_TARGET, SHOT_STOP_DIVING, SHOT_OFF_TARGET).

**Step 1 — Candidate search**: For each shot/save, find the first unsampled motion candidate 10-90s later. If none found, extract frames directly at +20s, +35s, +50s.

**Step 2 — Focused prompt**: Send a 4-way classification prompt: "corner_kick / goal_kick / throw_in / none". Faster and more focused than the general two-pass prompt.

**Step 3 — Confirm**: Confirmed set pieces are added to the verdict list as new events.

- **Parameters**: `_SET_PIECE_MIN_GAP=10s`, `_SET_PIECE_MAX_GAP=90s`, `_SET_PIECE_PROBES=[20, 35, 50]`

Source: `src/detection/pipeline.py` — `DetectionPipeline._set_piece_inference()`

### Phase 3c.5: Shot→Restart Reclassification (Post-VLM)

Reclassifies shots based on the restart that follows — the strongest signal for what happened after a shot, since the VLM cannot reliably see GK save actions at distance.

**Rules** (for each confirmed `SHOT_ON_TARGET` with no existing save within ±15s):
- **shot + corner_kick** (within 90s) → `SHOT_STOP_DIVING` (parry inferred: GK deflected ball over goal line)
- **shot + goal_kick with short gap** (<25s) → `SHOT_OFF_TARGET` (ball went out quickly = miss)
- **shot + goal_kick with long gap** (≥25s) → left as `SHOT_ON_TARGET` for catch scan (Phase 3g)
- Corner is checked first — if both a corner and goal kick follow, the corner takes priority.

**Gap discrimination rationale**: A quick goal kick after a shot means the ball went wide/over. A long delay suggests the GK caught the ball, held it, then distributed — the goal kick is a consequence of catch-then-play, not a direct miss. Phase 3g (catch scan) handles these ambiguous cases with structural inference and VLM probing.

**Zero VLM calls** — purely post-processing on existing verdicts.

- **Parameters**: `_SHOT_RESTART_WINDOW=90s`, `_MISS_GAP_MAX=25s`

Source: `src/detection/pipeline.py` — `DetectionPipeline._shot_restart_reclassify()`

### Phase 3d: Independent Corner Scan (Post-VLM)

Corners are visually distinctive but the general VLM prompt often misses corner-specific cues (player at corner arc, ball at corner flag). This phase independently scans for corners at candidates that the main VLM pass classified as "none" (no event detected).

**Why separate from Phase 3c**: The set-piece rescan only triggers after detected shots/saves. Many corners don't follow a detected shot — they follow unclassified events, clearances, or deflections. This phase catches those.

**Step 1 — Candidate selection**: Take all VLM "none" verdicts. For each, also gather nearby unsampled motion candidates within ±20s. Skip candidates within 45s of an already-detected corner.

**Step 2 — Focused prompt**: Send a binary "is this a corner kick?" prompt with detailed corner visual cues (player at corner arc, players in penalty area, high cross into box).

**Step 3 — Confirm**: Confirmed corners are added to the verdict list.

- **Parameters**: `_CORNER_SCAN_WINDOW=20s`, `_CORNER_DEDUP_WINDOW=45s`

Source: `src/detection/pipeline.py` — `DetectionPipeline._corner_scan()`

### Phase 3e: Reverse Restart Inference (Post-VLM)

Works backwards from detected restarts (goal kicks, corners) to find shots that the VLM missed. If a confirmed `goal_kick` or `corner_kick` has no associated save or shot within the preceding 30s, the nearest rejected candidate before it is reclassified as a shot.

**Logic**:
- `goal_kick` with no preceding save → rejected candidate 3-30s before = `SHOT_OFF_TARGET` (missed → goal kick)
- `corner_kick` with no preceding save → rejected candidate 3-30s before = `SHOT_OFF_TARGET` (blocked/deflected → corner)

**Confidence**: Fixed at 0.6 for reverse-inferred shots.

Source: `src/detection/pipeline.py` — `DetectionPipeline._reverse_restart_inference()`

### Phase 3f: Binary Shot Scan (Post-VLM)

Re-probes remaining rejected candidates with a focused binary "was a shot taken?" prompt. Binary questions perform better than multi-class for VLMs on less common categories.

**Candidate selection**: Rejected verdicts that are >10s from any confirmed event. Capped at 30 VLM calls, distributed evenly across time.

**Clip window**: Tighter than the main prompt — 3s pre / 5s post — to focus on the shot moment rather than the aftermath.

**Prompt**: Binary yes/no with `shot_type` sub-classification (`on_target`, `off_target`, `blocked`).

Source: `src/detection/vlm_verifier.py` — `VLMVerifier.verify_shot()`, `src/detection/pipeline.py` — `DetectionPipeline._shot_scan()`

### Phase 3g: Catch Scan (Post-VLM)

Detects goalkeeper catches — the hardest save type because catches produce no motion spike (GK just holds the ball) and are difficult for VLMs to see at 50m distance.

**Two strategies**:

1. **Structural inference** (no VLM cost): For each remaining `SHOT_ON_TARGET` with no restart (corner, goal kick, or kickoff) within 60s → infer `CATCH`. After a catch, the GK distributes in open play without a dead-ball restart. Confidence: original × 0.85.

2. **VLM probe** (targeted): For shots followed by a goal kick with a long gap (≥25s), extract frames at shot+3s to shot+8s and send a focused binary prompt: "Is the goalkeeper HOLDING the ball?" This shifted window captures the result state, not the shot itself.

**Why this works**: Catches are defined by what does NOT happen afterward (no restart), not by what the VLM sees. Parries produce corners, misses produce goal kicks, goals produce kickoffs — but catches produce nothing. The absence of a restart is itself the signal.

**Edge cases**: Post/crossbar hits where play continues look identical structurally. The 0.85 confidence penalty accounts for this ambiguity. The VLM probe path provides additional confirmation for the goal_kick-gap cases.

**Fallback**: If a VLM catch probe is rejected (GK not holding ball), the shot is downgraded to `SHOT_OFF_TARGET` — the catch hypothesis was tested and rejected, so the ball went out without a save.

- **Parameters**: `_CATCH_NO_RESTART_WINDOW=60s`, `_CATCH_SCAN_MAX=25` VLM calls
- **Confidence**: structural 0.85×, VLM probe uses VLM confidence directly

Source: `src/detection/pipeline.py` — `DetectionPipeline._catch_scan()`, `src/detection/vlm_verifier.py` — `VLMVerifier.verify_catch()`

### Diagnostic Dumps

Each phase writes JSONL diagnostics to `{working_dir}/diagnostics/`:
- `motion_candidates.jsonl` — all motion spikes (Phase 1)
- `audio_candidates.jsonl` — all audio cues (Phase 2)
- `final_candidates.jsonl` — after audio boost (Phase 2)
- `filtered_candidates.jsonl` — after match structure filter + audio gap fill + spot-checks (Phase 2b-2d)
- `vlm_verdicts.jsonl` — VLM classification results (Phase 3)
- `kickoff_rescan.jsonl` — goal inference kickoff rescan results (Phase 3b)
- `goal_celebration_probe.jsonl` — goal celebration/scoring VLM probe results (Phase 3b)
- `set_piece_rescan.jsonl` — set-piece inference results (Phase 3c)
- `corner_scan.jsonl` — independent corner scan results (Phase 3d)
- `shot_scan.jsonl` — binary shot scan results (Phase 3f)
- `catch_scan.jsonl` — catch scan VLM probe results (Phase 3g)

### Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `VLLM_ENABLED` | `false` | Enable vLLM backend for VLM classification |
| `VLLM_URL` | `http://10.10.2.222:8000` | vLLM server URL |
| `VLLM_MODEL` | `Qwen/Qwen3-VL-32B-Instruct-FP8` | Model name |
| `VLLM_MIN_CONFIDENCE` | `0.5` | Minimum VLM confidence |
| `AUDIO_ENABLED` | `true` | Enable audio boost phase |
| `AUDIO_SURGE_STDDEV` | `3.5` | Audio surge threshold (σ) |
| `MIN_EVENT_CONFIDENCE` | `0.50` | Global minimum event confidence |

---

## Legacy: YOLO-Heuristic Detection (Archived)

> The following section documents the original YOLO+ByteTrack heuristic detection system. It is **no longer active** — the motion-first pipeline above replaced it. Retained for reference.

### Event Detection Summary (Legacy)

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

### Dual-Emit: Shots from Saves

Every save (SHOT_STOP_DIVING, SHOT_STOP_STANDING, CATCH, PUNCH) implies a shot on target. The pipeline emits a companion SHOT_ON_TARGET event for each save verdict so saves appear in the GK reel and the implied shots appear in the highlights reel. The companion shot has 0.95× the save's confidence.

### Deduplication

Same-type dedup: events of the same type within 15s → keep higher confidence.

Cross-type dedup: events in the same **related group** within 15s → keep higher-priority type. Groups are separated so saves and shots can coexist:

1. **GK save events**: SHOT_STOP_DIVING, SHOT_STOP_STANDING, CATCH, PUNCH, ONE_ON_ONE
2. **Highlights shot events**: GOAL, SHOT_ON_TARGET, SHOT_OFF_TARGET, NEAR_MISS, PENALTY, FREE_KICK_SHOT
3. **Set pieces**: CORNER_KICK, THROW_IN, GOAL_KICK

This ensures a save and its companion shot both survive dedup (different groups, different reels). Within each group, the highest-priority event wins (e.g., goal beats shot_on_target).

## Known Issues

1. **Shot detection has no spatial awareness** — fires on clearances, long passes, and goal kicks because it only checks ball velocity, not position relative to opposing goal or direction toward goal.
2. **Keeper distribution/buildout not captured** — GK passes and throws after collecting the ball are classified as DISTRIBUTION_SHORT/LONG by the GK Detector's velocity method, but this is unreliable. The BallTouchDetector has distribution detection disabled by default (`detect_distributions=false`).
