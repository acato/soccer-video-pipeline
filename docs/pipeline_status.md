# Pipeline Status

Snapshot of the current production pipeline's measured performance and known
limits, as of 2026-05-20.

For the system design that produced these numbers see
[architecture.md](architecture.md). For setup/run instructions see
[install.md](install.md).

---

## Detection performance — 4-game eval set

The pipeline was tuned on four games against full event-stream ground truth
provided by a third-party stats data feed.

| Game        | GT goals | GT saves | GT shots |
|-------------|---------:|---------:|---------:|
| game_20     |        9 |       12 |       37 |
| game_22     |        5 |       17 |       48 |
| game_21     |        2 |       11 |       19 |
| rush        |        4 |       17 |       47 |
| **Total**   |   **20** |   **57** |  **151** |

All numbers below use a **±60s coverage metric**: a GT event is matched if any
detector event lies within ±60s. This is the right metric for reel
construction (overlapping clips share coverage; greedy 1:1 matching
under-counts and isn't relevant for the end use case).

### Goals

| Tier      | TP | FP  | FN | Recall | Precision |
|-----------|---:|----:|---:|-------:|----------:|
| confirmed |  7 |   1 | 13 |  0.35  |  0.88     |
| candidate | 13 | 96  |  0 |  0.65  |  0.12     |
| **union** | **20** | **97** | **0** | **1.00** | **0.17** |

100% recall on all 20 GT goals across 4 games. Confirmed-tier alone is the
high-precision dual_pass detection; candidate-tier is the kickoff ensemble
(formation_base + pattern_v11) that recovers the missing 65% recall.

### Saves

| Tier      | TP | FP  | FN | Recall | Precision |
|-----------|---:|----:|---:|-------:|----------:|
| confirmed | 24 |  27 | 33 |  0.42  |  0.47     |
| candidate | 44 | 143 | 13 |  0.77  |  0.24     |
| inferred  | 23 |  52 | 34 |  0.40  |  0.31     |
| **union** | **55** | **222** | **2** | **0.96** | **0.20** |

- confirmed: catch + shot_stop_diving/standing
- candidate: shot_on_target + free_kick_shot (most shots-on-goal end in a save)
- inferred: throw_in (with ±180s shot-proximity filter) + corner_kick

Only 2 silent FNs at ±60s — both game_20 parries with no detector event of
any type nearby. At ±90s only 1 FN remains.

### Shots

| Tier      | TP  | FP  | FN | Recall | Precision |
|-----------|----:|----:|---:|-------:|----------:|
| confirmed | 120 |  79 | 31 |  0.79  |  0.60     |
| candidate |  91 |  25 | 60 |  0.60  |  0.78     |
| **union** | **129** | **104** | **22** | **0.85** | **0.55** |

- confirmed: goal + shot_on_target + free_kick_shot
- candidate: saves + corners + goal_kicks (necessary-not-sufficient signals)

Notably, candidate-tier shot precision (0.78) is *higher* than confirmed —
saves and corner/goal kicks are sparser but stronger shot-evidence signals
than the bare `shot_on_target` detector firing.

Per-outcome coverage at ±60s union:

| GT outcome        | Covered | Total | Coverage |
|-------------------|--------:|------:|---------:|
| Blocked Shots     |      21 |    23 |   0.91   |
| Shots Off Target  |      44 |    51 |   0.86   |
| Goals             |      17 |    20 |   0.85   |
| Shots On Target   |      44 |    53 |   0.83   |

---

## Reel output (final keeper + highlights reels)

Reel-builder selection rules apply on top of the tier tagging:
- **keeper**: events with `goal_tier` OR `save_tier`, excluding `event_type ∈ {throw_in, free_kick_shot}` (sideline restarts and defender clearances respectively)
- **highlights**: events with `shot_tier`

Per-game reel durations (after exclusions):

| Game       | Keeper duration | Highlights duration |
|------------|----------------:|--------------------:|
| game_20    |    15 min       |     12 min          |
| game_21    |    13 min       |     15 min          |
| game_22    |    21 min       |     22 min          |
| rush       |    20 min       |     25 min          |

---

## What's known to fail

- **Cameras that crop out the goal mouths.** Empirically confirmed on
  game_18 (Westside Metros 2026-03-14): the camera framing excluded both
  goal areas, so the detector saw zero catches, no real shots-on-target,
  and no celebration cuts. The pipeline assumes both goals are in the
  camera's coverage. If they aren't, keeper-reel output collapses to
  ~all FPs from midfield activity.
- **game_20 has 2 truly silent parries** (no detector event of any type
  within ±60s). They appear in neither tier. Hard ceiling unless a new
  visual save-pattern detector is built.
- **Single-camera new-venue games** may underperform the four trained-on
  games. The detection chain was optimized for sideline-camera amateur
  footage at ~50m. Behind-goal, drone, or significantly different framings
  are out of distribution.
- **Subtype labels are unreliable.** catch vs shot_stop_diving vs
  shot_stop_standing — the detector mixes these. The pipeline collapses
  them into a single "save" union; subtype-level scoring is not meaningful.
- **Free-kick shots include defender clearances.** The detector tags any
  free-kick motion (attacking or defensive) as `free_kick_shot`. The
  keeper reel excludes this type for that reason; the highlights reel
  keeps it (FP-tolerant by design).

---

## Methodology notes

- **Coverage metric, not greedy 1:1** — for each GT event, we ask "does any
  detector event lie within ±60s?" not "is there a unique detector event
  matched to it?" The latter under-counts when GT events cluster.
- **±60s tolerance** — wide enough to accommodate ~30s of pre-event camera
  framing and ~30s of post-event play. Tighter (±15-30s) is too sensitive
  to detector timestamp jitter; wider (±90s) starts double-counting.
- **Ground truth source** — third-party stats data with timestamps in
  game-clock seconds. The pipeline converts to video time using 1H and
  2H offsets calibrated per game via the wide-shot density anchor.
