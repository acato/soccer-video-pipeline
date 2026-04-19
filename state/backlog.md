# Backlog

Items deferred from active work. Reference from `state/current_run.json` notes when a run picks one up.

## YOLO grounding follow-ups (from Run #33 post-mortem)

### B — Relax goal_kick vertical band
Drop or widen the `_GOAL_LINE_VERT_MIN=0.25` / `_GOAL_LINE_VERT_MAX=0.75` gate in `src/detection/yolo_grounding.py:57-58`.

**Why**: Rejected goal_kicks at y=0.24, y=0.14 look like plausible goal-kick geometry under perspective (elevated/corner camera). The middle-band assumption comes from a strict sideline POV that doesn't hold for our footage.

**When to pick up**: After Option A lands and we still see goal_kick TP loss in diag.

**Expected impact**: Recover ~2–5 goal_kick TPs. Low risk.

### C — Soccer-ball-tuned ball detector
Replace COCO YOLOv8's generic `sports_ball` class with a soccer-specific detector.

**Why**: Current ball-detection rate is 52%; gate only does real work on half the events. Better detection → gate decisions matter more.

**Options (in order of effort)**:
1. Swap in a Roboflow Universe football-ball YOLOv8 weight (5 min test).
2. Adopt `footandball` (Komorowski 2020) — purpose-built for wide-POV soccer, 91%+ ball recall on Issia-CNR.
3. Fine-tune YOLOv8n on ~300 labeled frames from the Rush game (~1h training on Mac).

**When to pick up**: Only after Option A. C doesn't pay off if sampling window is wrong.

**Expected impact**: Lifts ball-detection rate from 52% → 80%+. Gate becomes load-bearing rather than fail-open on half the traffic.

**Cost/risk**: Mac inference stays fast (nano models). For #3, need a labeling pass.
