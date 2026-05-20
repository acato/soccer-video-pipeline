# Soccer Video Pipeline — Architecture

This document describes the production detection pipeline as it stands today:
a multi-stage chain that converts a raw soccer match video into per-game
**keeper** and **highlights** reels with high recall on the events that matter.

If you're trying to install and run the system, see [INSTALL.md](INSTALL.md)
instead.

---

## 1. End-to-end picture

```
┌──────────────────┐
│ raw video (mp4)  │  e.g. ~60-90 min, 1080p, sideline camera at ~50m
└─────────┬────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────────┐
│                      Stage 1 — Per-frame signals                    │
│                                                                     │
│  FrameSampler  →  YOLO grounding (player + ball)                    │
│                                                                     │
│  Output:  /tmp/kickoff_<game>_frames.jsonl                          │
│  Schema:  {t, ball:[x,y,conf]|null, p_left, p_right, total_field,  │
│            in_circle, wide_shot, ball_at_center, kickoff_setup,    │
│            kickoff_setup_strong, kickoff_scene}                     │
└─────────┬───────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────────┐
│                  Stage 2 — Dual-pass VLM event detector             │
│                                                                     │
│  Sliding-window frames  →  Qwen3-VL-32B-FP8 (v11 LoRA, vLLM 0.19.1)│
│                                                                     │
│  Output:  /tmp/soccer-pipeline/<job_id>/events.jsonl                │
│  Events:  goal, shot_on_target, free_kick_shot, catch,              │
│           shot_stop_diving, shot_stop_standing, corner_kick,        │
│           throw_in, goal_kick                                        │
└─────────┬───────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────────┐
│            Stage 3 — Kickoff ensemble (two parallel paths)          │
│                                                                     │
│  3a. pattern_v11 — heuristic on cached YOLO frames                  │
│      detectors:  celebration_cut, ball_traversal                    │
│      output:     kickoff_<game>_pattern_v11_0191.jsonl              │
│                                                                     │
│  3b. formation_base — wide-shot density anchor → formation          │
│      candidates → base Qwen3-VL-32B-FP8 (NO LoRA) verifier         │
│      output:     kickoff_<game>_formation_v2_base.jsonl             │
└─────────┬───────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────────┐
│            Stage 4 — Merge + tier tagging                           │
│            (scripts/merge_ensemble_into_events.py)                  │
│                                                                     │
│  Per event, tags three orthogonal metadata fields:                  │
│   • goal_tier   ∈ {confirmed, candidate}                            │
│   • save_tier   ∈ {confirmed, candidate, inferred}                  │
│   • shot_tier   ∈ {confirmed, candidate}                            │
│                                                                     │
│  Applies: relaxed-aggregation rule, negative-evidence filter,       │
│           throw_in shot-proximity filter (±180s)                    │
│                                                                     │
│  Output:  /tmp/kickoff_<game>_tiered_events.jsonl                   │
└─────────┬───────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────────┐
│                 Stage 5 — Reel assembly                             │
│                 (scripts/build_reels.py)                            │
│                                                                     │
│  • Select events by tier flags (per reel type)                      │
│  • Pad ±6s pre / +12s post                                          │
│  • Merge overlapping windows (gap ≤3s, max clip 60s)                │
│  • ffmpeg stream-copy each clip → concat demuxer                    │
│                                                                     │
│  Output:  <out>/<game>_keeper.mp4                                   │
│           <out>/<game>_highlights.mp4                               │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Models used

| Role | Model | Distribution |
|------|-------|--------------|
| Dual-pass VLM (primary) | Qwen3-VL-32B-FP8 + **v11 LoRA** (LoRA-merged FP8) | [acatorcini/qwen3-vl-32b-soccer-v11-fp8](https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8) (HuggingFace, ~34 GB) |
| Formation verifier | `Qwen/Qwen3-VL-32B-Instruct-FP8` (base, no LoRA) | [Qwen/Qwen3-VL-32B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) (HuggingFace, public) |
| YOLO player detector | `yolov8_soccer_uisikdag` | Soccer-specialized YOLOv8 ([github.com/uisikdag](https://github.com/uisikdag/weed_soccer_models) or similar) |
| YOLO ball detector | `v9b_best.pt` (custom YOLOv9 fine-tune) | [acatorcini/yolov9-soccer-ball](https://huggingface.co/acatorcini/yolov9-soccer-ball) (HuggingFace, ~22 MB) |

The LoRA adapter alone (2.27 GB, for custom merges or continued training) is
also published as [acatorcini/qwen3-vl-32b-soccer-v11-lora](https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-lora).

The pipeline runs **both** v11 LoRA and base FP8 on the same Qwen architecture
but at different points: the LoRA-merged checkpoint handles the dual-pass event
classification, and the base checkpoint is loaded separately for formation
verification (the LoRA suppresses the `kickoff_restart` label that the base
emits — see [§5](#5-key-design-decisions)).

A typical production deployment runs vLLM with `--quantization compressed-tensors`
for the LoRA-merged FP8, then later restarts with `--quantization fp8` against
the base FP8 path for the formation verification pass.

---

## 3. Event types, tiers, and reel selection

### Event types emitted by the dual-pass detector

| Type | Meaning |
|------|---------|
| `goal` | Ball entered the net |
| `shot_on_target` | Attempt at goal, on frame |
| `free_kick_shot` | Free-kick motion (any free kick — attacking OR defensive clearance) |
| `catch` | Keeper catches ball |
| `shot_stop_diving` | Keeper makes a diving save |
| `shot_stop_standing` | Keeper makes a standing save |
| `corner_kick` | Corner-kick restart |
| `throw_in` | Throw-in restart (sideline OR keeper distribution) |
| `goal_kick` | Goal-kick restart |

### Tier semantics

Three orthogonal `metadata.*_tier` fields. An event can carry all three.

**`goal_tier`**
- `confirmed` — dual_pass-detected `goal`. Precision ~0.88. ~35% recall on its own.
- `candidate` — kickoff ensemble (pattern_v11 + formation_base) goal candidate, deduped within 60s of any confirmed goal. Precision ~0.15. Brings union recall to 1.00 on our 4-game eval.

**`save_tier`**
- `confirmed` — `catch`, `shot_stop_diving`, `shot_stop_standing`. Precision 0.47.
- `candidate` — `shot_on_target`, `free_kick_shot`. Precision 0.24. Shot detections are a strong save proxy (keeper either caught it or it missed / was blocked).
- `inferred` — `throw_in` (filtered to ones preceded by a shot-like event within ±180s — keeper distributions, not sideline throws) and `corner_kick` (every corner means a defender or keeper deflection). Precision 0.31. Captures saves the catch-detector missed.

**`shot_tier`**
- `confirmed` — `goal`, `shot_on_target`, `free_kick_shot`. Precision 0.60.
- `candidate` — saves + corners + goal_kicks (necessary-not-sufficient — never auto-promote one of these to "shot" alone; use only as union evidence). Precision 0.78 — *higher* than confirmed because these signals are sparse but very specific.

### Reel rules

| Reel | Selection | Excluded types |
|------|-----------|----------------|
| keeper | events with `goal_tier` OR `save_tier` set | `throw_in`, `free_kick_shot` (sideline restarts and defender clearances respectively — high FP, low yield) |
| highlights | events with `shot_tier` set | none (FP-tolerant by design) |

A keeper reel for the 4-game eval averaged 18 minutes; highlights averaged 19
minutes.

---

## 4. Detection performance on the 4-game eval set

The pipeline was tuned on four games (game_20 / game_21 / game_22 / rush)
with full event-stream ground truth from a separate data provider. All
numbers below use a ±60s coverage metric (a GT event is matched if any
detector event lies within ±60s — the right metric for reel construction
where overlapping clips share coverage).

| Event type | Confirmed precision | Confirmed recall | Union recall |
|------------|--------------------:|-----------------:|-------------:|
| Goals      | 0.88                | 0.35             | **1.00**     |
| Saves      | 0.47                | 0.42             | **0.96**     |
| Shots      | 0.60                | 0.79             | **0.85**     |

The recall ceilings have been pushed to the visual-perception limit: only
8 of 151 GT shots, 2 of 57 GT saves, and 0 of 20 GT goals lack any
detector event nearby. The remaining hard FNs are concentrated in a
single game with the most distant camera framing.

---

## 5. Key design decisions

**Why two Qwen checkpoints?**
The v11 LoRA was trained to suppress noisy labels (including
`kickoff_restart`) for cleaner event classification, but that came at the
cost of missing goals where the kickoff restart is the strongest signal.
Running the base FP8 in parallel for formation verification recovers
those goals. Two passes, two checkpoints, union output.

**Why a kickoff ensemble?**
Goals at amateur-soccer camera distances have a small ball (3-5 px) that
ViT-based detectors struggle with directly. The kickoff *restart* after
a goal is a much larger visual signal — players form distinctive
formations at the center circle. We detect THAT pattern instead, then
infer "a goal happened ~60s before."

**Why a coverage metric (not greedy 1:1 matching)?**
For reel construction the relevant question is "does any clip cover this
GT moment?" — not "is there a unique detector event per GT?" When 3 GT
saves cluster in 30 seconds, a single clip covers them all and a coach
watching the reel sees all 3. Greedy 1:1 matching would mark 2 as FN
even though they're in the reel.

**Why does the candidate save tier outrank the inferred tier in precision?**
Empirically, `shot_on_target` (candidate tier) is a *stronger* save proxy
than `throw_in` (inferred tier) because most shots on target end in a save
or block, whereas most throw-ins are sideline restarts. The tier names
reflect *evidence directness*, not necessarily precision.

**Why throw_ins need a shot-proximity filter at ±180s?**
Throw-ins arise from two distinct game states: sideline ball-out (most
common, not save-related) and keeper distribution after a catch (rare,
save-related). The shot-proximity filter at 180s preserves the latter
(keeper distributions follow a shot) while excluding the former. At 180s
the filter is lossless on the 4-game eval — preserves all 23 save TPs
from throw-ins while cutting 9 sideline-throw FPs.

**Why are throw_in + free_kick_shot excluded from the keeper reel?**
User feedback after first build: too many sideline throws and defender
free-kick clearances. The throw-in cut is a hard exclusion (corner_kick
inferred stays in). The free_kick_shot cut targets defender clearances
that the detector tags as free-kick shots — at ~40 FPs cut per TP lost,
it's the highest-yield filter we could apply. Highlights reel keeps both
(FP-tolerant by design).

---

## 6. Storage layout

```
/Users/aless/soccer-working/                Source MP4 files (Mac local)
  ├── 2026-04-18 Celtic - Reign GA 11.mp4
  ├── 2026-04-25 Eastern WA Surf - Reign GA11.mp4
  ├── 2026-04-26 Spokane Shadow - Reign GA11.mp4
  └── 2026-02-07 - Rush - GA2008.mp4

/tmp/soccer-pipeline/<job_id>/             Dual-pass detector workspace
  ├── events.jsonl                          Detector output (used by merger)
  └── diagnostics/

/tmp/kickoff_<game>_frames.jsonl            YOLO per-frame cache
/tmp/kickoff_<game>_pattern_v11_0191.jsonl  Stage 3a output
/tmp/kickoff_<game>_formation_v2_base.jsonl Stage 3b output
/tmp/kickoff_<game>_tiered_events.jsonl     Stage 4 output (merged + tagged)

/Volumes/transit/reels/                     Final reel output
  ├── <game>_keeper.mp4
  └── <game>_highlights.mp4

/mnt/transit/soccer-finetune/checkpoints/   Fine-tune checkpoints (LLM host)
  └── v11-32b/
      ├── fp8-c150/                          LoRA-merged FP8 (vLLM-serves)
      └── v0-.../checkpoint-150/             LoRA adapter (alone, 2.27 GB)
```

The `/Volumes/transit/` mount is a shared NAS volume. On the LLM host
the same volume is accessible as `/mnt/transit/`.

---

## 7. Two pipelines in this repo

The repo contains code for **two different pipelines**:

1. **The legacy API/worker pipeline** (`src/api/`, `src/detection/`, Celery
   worker, FastAPI). Submit a video via `POST /jobs`, get reels via a job ID.
   This is the path described in the original README and `docs/architecture.md`.

2. **The current production scripts pipeline** (`scripts/`). The path
   documented above. Run via shell + Python scripts; no API, no Celery.
   This is what produced the 4-game eval results.

The dual-pass detector (stage 2 in the diagram) is shared between the two
pipelines — both invoke the same `src/detection/dual_pass_detector.py` and
write to `/tmp/soccer-pipeline/<job_id>/events.jsonl`. After that, the
pipelines diverge: legacy goes to `src/segmentation/clipper.py` and
`src/assembly/composer.py`, while the current path uses the `scripts/`
chain described above.

Long-term, the kickoff ensemble + tiering + reel builder will be folded
back into the API/worker path so jobs submitted via the REST API benefit
from the same recall/precision pipeline. For now, use the scripts path
for any new work.
