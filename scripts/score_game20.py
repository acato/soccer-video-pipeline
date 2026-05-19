"""Score game_20 ensemble verifier outputs against GT.

GT (Goals Conceded events in game-clock seconds):
  1H: 1072.2, 1137.0, 1639.2, 2314.2 (4 goals)
  2H: 2442.0, 3106.4, 3596.0, 3976.8, 4065.2 (5 goals)

period_start_time = 2400 for 2H, so 2H event_time is already cumulative.
"""
import json
from pathlib import Path

# Calibrated offsets (sweep_kickoffs.py used offset_1H=124; add kickoff lag 65)
OFFSET_1H = 124 + 65  # = 189
HALFTIME = 100        # rough estimate; tol=90 absorbs small errors
OFFSET_2H_SHIFT = OFFSET_1H + HALFTIME  # = 289

GT_1H_GAME = [1072.2, 1137.0, 1639.2, 2314.2]
GT_2H_GAME = [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]

GT_VIDEO = [g + OFFSET_1H for g in GT_1H_GAME] + \
           [g + OFFSET_2H_SHIFT for g in GT_2H_GAME]
print(f"GT video times: {[f'{g:.0f}' for g in GT_VIDEO]}")

TOL = 90.0


def load_confirmed(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()
            and json.loads(l).get("_vlm_verdict") == "GOAL"]


def score(detections, gt_times, tol=TOL):
    used = set()
    tp = 0
    pairs = []
    for d in sorted(detections, key=lambda x: x["start_sec"]):
        best_i, best_dt = None, float("inf")
        for i, g in enumerate(gt_times):
            if i in used:
                continue
            dt = abs(d["start_sec"] - g)
            if dt <= tol and dt < best_dt:
                best_i, best_dt = i, dt
        if best_i is not None:
            used.add(best_i)
            tp += 1
            pairs.append((d["start_sec"], gt_times[best_i]))
    return tp, len(detections) - tp, len(gt_times) - tp, pairs


# Load formation_base for 1H and 2H separately
form_1h = load_confirmed("/tmp/kickoff_game20_1H_formation_base.jsonl")
form_2h = load_confirmed("/tmp/kickoff_game20_2H_formation_base.jsonl")
# Pattern (v11 — but currently broken on 0.21.0)
# Try to load anyway; will be empty if v11 returned errors
try:
    pat_1h = load_confirmed("/tmp/kickoff_game20_1H_pattern_vlm.jsonl")
    pat_2h = load_confirmed("/tmp/kickoff_game20_2H_pattern_vlm.jsonl")
except Exception:
    pat_1h = []
    pat_2h = []

# Bound check: 1H scan covered 0-3550, 2H scan covered 3554-7229
# Candidates may overlap on the 2H content visible to the 1H scan.
# Just merge and dedup within DEDUP_WINDOW.
DEDUP = 30.0

def dedup(dets):
    out = []
    for d in sorted(dets, key=lambda x: x["start_sec"]):
        if out and (d["start_sec"] - out[-1]["start_sec"]) <= DEDUP:
            continue
        out.append(d)
    return out

formation_combined = dedup(form_1h + form_2h)
pattern_combined = dedup(pat_1h + pat_2h)
best_combined = dedup(formation_combined + pattern_combined)

for label, dets in (("formation_1h", form_1h),
                    ("formation_2h", form_2h),
                    ("formation_total", formation_combined),
                    ("pattern_1h", pat_1h),
                    ("pattern_2h", pat_2h),
                    ("pattern_total", pattern_combined),
                    ("BEST (form+pat)", best_combined)):
    tp, fp, fn, pairs = score(dets, GT_VIDEO)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"\n{label}: kept={len(dets)} TP={tp} FP={fp} FN={fn} prec={prec:.2f} rec={rec:.2f}")
    for d, g in pairs:
        print(f"  TP: det {d:.0f} -> GT {g:.0f}")
