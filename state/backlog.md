# Backlog

Items deferred from active work. Reference from `state/current_run.json` notes when a run picks one up. Shipped items are removed; git history carries the record.

## YOLO grounding — spatial gates

### B — Relax goal_kick vertical band
Drop or widen the `_GOAL_LINE_VERT_MIN=0.25` / `_GOAL_LINE_VERT_MAX=0.75` gate in `src/detection/yolo_grounding.py`.

**Why**: Rejected goal_kicks at y=0.24, y=0.14 look like plausible goal-kick geometry under perspective (elevated/corner camera). The middle-band assumption comes from a strict sideline POV that doesn't hold for our footage.

**When to pick up**: After Run #35 lands. Revisit only if goal_kick still has TP loss attributable to vertical-band rejections in Run #35 diagnostics.

**Expected impact**: Recover ~2–5 goal_kick TPs. Low risk.

## Goalkeeper-action gates (new layer — Run #35 enables)

Run #35 is the data-collection run for these. It ships (1) the soccer-tuned
YOLO weight (uisikdag) with native goalkeeper class, and (2) GK-position
diagnostics in `yolo_grounding.jsonl`. Both Run #36 and Run #37 depend on
ball-detection rate improving from 52% (COCO) to ~75%+ in Run #35 — if
that doesn't materialize, these runs are premature.

### Run #36 — GK-proximity gate (SHIPPED; threshold calibration pending)
Extends `_GATED_TYPES` to include `CATCH`, `SHOT_STOP_DIVING`, `PUNCH`,
`SHOT_STOP_STANDING`. For those events, requires ball and GK to be
co-visible in ≥1 sampled frame with normalized Euclidean distance
≤ `gk_proximity_threshold` (default 0.20).

**Why this scope**: Run #35 showed the gate wasn't running on GK events
at all — no ball or GK data was being collected for `catch`/`shot_stop`.
Run #36 does both the FP cut AND the diagnostic collection in one step.

**Threshold is provisional at 0.20** (≈ 20% of frame diagonal). This
was picked without TP-set calibration because Run #35's gate didn't
cover GK events. **Tune for Run #36b** from Run #36's diag:
measure ball-GK distances on the 4 shot_stop_diving TPs + 5 catch TPs.

**Expected impact**: Cut `shot_stop_diving` FPs by 50–70% (13–18 FPs)
and `catch` FPs by 30–50% (5–8 FPs). Net F1 gain ~+0.03 to +0.05.

**Fail-open policy**: keep when ball or GK missing (preserves recall);
only reject on active evidence they were co-observed but never close.

### Run #37 — Parry signature (additive on top of Run #36)
For GK events that survive Run #36's proximity gate, classify the
ball's post-contact trajectory signature:

- **Ball reverses direction** (angle change >~90° from pre-contact vector
  to post-contact vector): parry signature → keep, tag as parry-quality
  for reels.
- **Ball trajectory continues past GK toward goal**: probable goal/
  near-miss; the "save" is a false positive — drop.
- **Ball stays near GK for ≥N frames post-contact**: catch signature —
  only valid for `catch`-typed VLM events.

**Corroborative signal** (when corner_kick detection is fixed): a
`corner_kick` VLM event landing within 10–15s after a save event boosts
parry-signature confidence (ball deflected out of bounds → corner).

**Why (from user)**: Parries are distinct, highly watchable events with
a temporal signature — GK activity + ball reversal + optional corner
follow-up. They're currently buried in `shot_stop_diving` (mixed with
catches and FPs). Surfacing them adds reel quality independent of F1.

**Dependencies**:
1. Ball-detection density sufficient for trajectory estimation — needs
   n_frames ≥ 8 over ±3s and detection rate ≥ 75%. Gate this run on
   Run #35 confirming the detection rate.
2. For the corner corroboration: `corner_kick` VLM emission must be
   fixed first (currently 0 detected). Not a blocker — trajectory alone
   is viable — but the signal is stronger with both.
3. Ground-truth calibration: eyeball the 5 GT `shot_stop_diving` events
   after Run #35 to categorize parries vs catches; sets realistic
   expectations for Run #37.

**Expected impact**: 
- F1: modest (~+0.01–0.02) since `shot_stop_diving` GT is only 5 events.
- Reel quality: significant — parries get dedicated tagging, high
  confidence on `keeper` reel inclusion.

**Fold-in point**: Once Run #35 finishes, review its GK-position and
ball-trajectory diagnostics before finalizing Run #37's thresholds —
measure real trajectory reversals for TP saves, set angle threshold
empirically rather than at a guessed 90°.

## Corner-kick emission (not yet scheduled)
Unrelated to the YOLO grounding work but blocking parry corroboration
and `corner_kick` F1 altogether. VLM has emitted zero corners since
Run #32's prompt revert. Separate problem, separate owner.
