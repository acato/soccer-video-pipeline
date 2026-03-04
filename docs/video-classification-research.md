# Video Classification Research for Soccer Event Detection

Research findings for classifying dead-ball restart events (goal kicks, corner kicks, throw-ins, free kicks) in youth soccer match recordings.

## Problem Statement

Ball-direction heuristics (speed thresholds, stationary detection, pixel-based zone checks) are too brittle for classifying restart types. They break across camera angles, zoom levels, and field positions. Jersey color matching for GK identification has a ~94% false positive rate with spectrally close colors (e.g., teal GK vs blue opponents).

## Approach: Two-Stage Pipeline

### Stage 1 — Candidate Generation

Scan ball trajectory for gaps where the ball disappears for ≥1.5s. These gaps reliably indicate dead-ball situations regardless of camera position or zoom. No classification at this stage — just "something interrupted play."

Gap detection is implemented in `BallTouchDetector._find_trajectory_gaps()`. It uses the existing `BallTrajectory.find_gaps()` method, filtering by:
- Minimum gap duration: 1.5s (below this, likely tracking noise or brief occlusion)
- Maximum gap duration: 30s (above this, likely halftime or extended stoppage)

### Stage 2 — Gemini 2.5 Flash Classification

Extract a 12-15s video clip around each gap. Send to Gemini 2.5 Flash for structured classification. Gemini provides native video understanding — temporal continuity, player movement patterns, referee signals, and (optionally) audio cues like whistles.

## Model Comparison

| Metric | Gemini 2.5 Flash | Claude VLM (current) | Trajectory heuristics |
|--------|------------------|---------------------|-----------------------|
| Cost/event | ~$0.001 | ~$0.013 | $0 |
| Cost/game | ~$0.05-0.20 | ~$2.60 | $0 |
| Video understanding | Native temporal | 3 static frames | Speed/direction thresholds |
| Audio cues | Yes (whistle) | No | No |
| Accuracy (est.) | 75-85% | 50-65% | 60-75% (brittle) |

### Token Math

- 12s clip at 1 FPS, low resolution: 12 frames × 66 tokens = 792 visual tokens
- Plus ~500 tokens for prompt + ~200 for response
- Total per clip: ~1,500 tokens input + ~200 output
- At $0.15/1M input, $0.60/1M output: **$0.000345 per clip**
- 50 dead-ball gaps per game × $0.000345 = **$0.017 per game**

Even at default resolution (258 tokens/frame): $0.001 per clip, $0.05 per game.

## Alternatives Considered

### Pitch Calibration / Homography

Map pixel coordinates to real-world pitch coordinates using a homography transformation. Would enable precise zone detection (6-yard box, corner arc, touchline) regardless of camera angle.

**Pros:** Enables precise spatial reasoning, works with any downstream classifier.
**Cons:** Requires manual calibration per camera position. Youth soccer uses varied venues with no consistent camera setup. Semi-automatic methods (line detection) exist but require visible pitch markings.

**Verdict:** Too much per-game manual effort for a youth soccer pipeline. Revisit if consistent camera positions become available.

### Video Transformers (VideoMAE, T-DEED, Hiera)

Fine-tune a video understanding model on labeled soccer restart clips.

**Pros:** Fast inference (local, no API cost), can run on GPU alongside YOLO.
**Cons:** Requires 200+ labeled clips per class for fine-tuning, significant training infrastructure. Zero-shot video transformers exist but accuracy is lower than Gemini for this task.

**Verdict:** Future enhancement once we have enough labeled data from Gemini classifications. Current labeled dataset is empty.

### Commercial Sports Analytics APIs

Several companies offer soccer event detection APIs (StatsBomb, Opta, Hudl).

**Pros:** Production-grade accuracy, real-time capable.
**Cons:** Expensive ($1000+/mo), designed for professional leagues, may not work well on youth soccer with single sideline camera.

**Verdict:** Overkill for a personal keeper reel tool. Revisit if scaling to a service.

### Frame-by-Frame Image Classification

Send individual frames to a vision model (GPT-4V, Claude) for classification.

**Pros:** Simpler than video, many model options.
**Cons:** Loses temporal context (can't see ball trajectory, player movement). A single frame of a goal kick looks similar to a free kick. Much more expensive (3+ frames per event vs 1 video clip).

**Verdict:** Already tried with Claude VLM for saves — 3 static frames lack the temporal signal needed for restart classification. Video-native models are clearly better for motion-defined events.

## Implementation

See `src/detection/gemini_classifier.py` for the `GeminiClassifier` class and `src/detection/ball_touch_detector.py` for `_find_trajectory_gaps()`.

Configuration:
- `GEMINI_ENABLED=true` — enable the pipeline
- `GEMINI_API_KEY` — Google AI API key
- `GEMINI_MODEL=gemini-2.5-flash` — model to use
- `GEMINI_CLIP_PRE_SEC=5.0` — seconds before gap in clip
- `GEMINI_CLIP_POST_SEC=8.0` — seconds after gap in clip
- `GEMINI_MIN_CONFIDENCE=0.5` — minimum confidence threshold

## Future Work

1. **Throw-in / free kick reels** — Gemini already classifies these; add event types + reel presets
2. **Fine-tuned local model** — Once 200+ labeled clips are available from Gemini output, train a VideoMAE or T-DEED model for local inference
3. **Pitch calibration** — Semi-manual homography for precise zone detection
4. **Full-game Gemini scan** — Process entire game segments instead of just gap candidates, enabling detection of events that don't cause ball disappearance
5. **Auto-kickoff detection** — Use Gemini to identify game start, eliminating need for manual `game_start_sec`
