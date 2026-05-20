"""Test multiple filter variants on the dense YOLO per-frame data for game_22.

The strict variant (wide + ball_at_center + 1-3 in_circle) killed all 58
candidates. Try permutations to find one that keeps real-goal TPs but
cuts midfield-play FPs.
"""
import json
from pathlib import Path

CENTER_X_LO, CENTER_X_HI = 0.40, 0.60
CENTER_Y_LO, CENTER_Y_HI = 0.35, 0.50
WIDE_MIN = 18

GT_VIDEO_G22 = [1755, 2390, 4620, 5738, 5952]
TOL = 90.0


def frame_passes(f, *, ball_required=False,
                 in_c_min=0, in_c_max=15,
                 wide_required=True,
                 require_balanced_lr=False, max_lr_diff=5):
    if wide_required and f["total_field"] < WIDE_MIN:
        return False
    inc = f.get("in_circle", 0)
    if not (in_c_min <= inc <= in_c_max):
        return False
    if ball_required:
        ball = f.get("ball")
        if not ball:
            return False
        bx, by = ball[0], ball[1]
        if not (CENTER_X_LO <= bx <= CENTER_X_HI and CENTER_Y_LO <= by <= CENTER_Y_HI):
            return False
    if require_balanced_lr:
        if abs(f.get("p_left", 0) - f.get("p_right", 0)) > max_lr_diff:
            return False
    return True


def candidate_passes(cand, frame_filter):
    return any(frame_filter(f) for f in cand.get("_dense_frames", []))


def score_against_gt(cands, gts, tol=TOL):
    used = set()
    tp = 0
    fp = 0
    for c in sorted(cands, key=lambda x: x["start_sec"]):
        matched = False
        for i, g in enumerate(gts):
            if i in used:
                continue
            if abs(c["start_sec"] - g) <= tol:
                used.add(i)
                tp += 1
                matched = True
                break
        if not matched:
            fp += 1
    return tp, fp, len(gts) - tp


rows = [json.loads(l) for l in Path("/tmp/kickoff_game_22_dense_v2.jsonl").read_text().splitlines() if l.strip()]
print(f"loaded {len(rows)} candidates with dense_frames")
# Check how many dense_frames per candidate
sample_frame_counts = [len(r.get("_dense_frames", [])) for r in rows]
print(f"dense_frames per candidate: min={min(sample_frame_counts)} "
      f"max={max(sample_frame_counts)} mean={sum(sample_frame_counts)/len(sample_frame_counts):.0f}")

# Diagnostic: for TPs (near GT goals), what do dense frames look like?
print(f"\n=== Sample dense frames near GT 1755 (1H#1) ===")
near_gt = [r for r in rows if abs(r["start_sec"] - 1755) <= 30]
for c in near_gt[:1]:
    frames = sorted(c["_dense_frames"], key=lambda x: x["t"])
    print(f"candidate t={c['start_sec']}, {len(frames)} dense frames")
    for f in frames[:21]:  # first 21 = ±10s
        ball = f.get("ball")
        bxy = f"({ball[0]:.2f},{ball[1]:.2f})" if ball else "None"
        bc = (CENTER_X_LO <= ball[0] <= CENTER_X_HI and
              CENTER_Y_LO <= ball[1] <= CENTER_Y_HI) if ball else False
        print(f"  t={f['t']:.1f} wide={int(f['total_field']>=WIDE_MIN)} "
              f"inC={f.get('in_circle', 0):>2} tot={f['total_field']:>2} "
              f"L={f.get('p_left',0):>2} R={f.get('p_right',0):>2} ball={bxy} bc={bc}")

# Filter variants — apply each at candidate level (any dense frame passes)
VARIANTS = [
    ("strict (wide+ball_c+inC1-3)",
     lambda f: frame_passes(f, ball_required=True, in_c_min=1, in_c_max=3)),
    ("wide+ball_c+inC1-5",
     lambda f: frame_passes(f, ball_required=True, in_c_min=1, in_c_max=5)),
    ("wide+ball_c+inC1-8",
     lambda f: frame_passes(f, ball_required=True, in_c_min=1, in_c_max=8)),
    ("wide+ball_c (no inC)",
     lambda f: frame_passes(f, ball_required=True)),
    ("wide+inC1-3 (no ball)",
     lambda f: frame_passes(f, in_c_min=1, in_c_max=3)),
    ("wide+inC1-5 (no ball)",
     lambda f: frame_passes(f, in_c_min=1, in_c_max=5)),
    ("wide+inC1-8 (no ball)",
     lambda f: frame_passes(f, in_c_min=1, in_c_max=8)),
    ("wide+balanced LR (|L-R|<=3)",
     lambda f: frame_passes(f, require_balanced_lr=True, max_lr_diff=3)),
    ("wide+balanced LR(<=3)+inC<=5",
     lambda f: frame_passes(f, in_c_max=5, require_balanced_lr=True, max_lr_diff=3)),
    ("wide+ball_c+balanced LR(<=5)",
     lambda f: frame_passes(f, ball_required=True, require_balanced_lr=True, max_lr_diff=5)),
]

print(f"\n=== Filter variants applied to game_22 dense data ===")
print(f"{'variant':<42} {'kept':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'recall':>6} {'prec':>6}")
print("-" * 80)
for label, filt in VARIANTS:
    kept = [c for c in rows if candidate_passes(c, filt)]
    tp, fp, fn = score_against_gt(kept, GT_VIDEO_G22)
    rec = tp / len(GT_VIDEO_G22)
    prec = tp / max(1, tp + fp)
    print(f"{label:<42} {len(kept):>4} {tp:>3} {fp:>3} {fn:>3} {rec:>6.2f} {prec:>6.2f}")
