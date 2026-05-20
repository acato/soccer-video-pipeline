"""Apply YOLO-based post-goal verification using cached per-frame data.

Mirrors the existing src/detection/kickoff_verifier.py logic but works
offline on the cached /tmp/kickoff_<game>_frames.jsonl files instead of
re-running YOLO. For each ensemble GOAL candidate, probe per-frame data
at +20, +40, +60s and check:
  (a) ball central (x in [0.4, 0.6], y in [0.4, 0.6])
  (b) ≥18 persons visible
  (c) |p_left - p_right| / total ≤ 0.4 (relatively balanced)

KEEP if ANY probe passes all three. This is the existing verifier's
"high-recall" mode. For FP cutting, we may need a stricter variant.
"""
import json
from pathlib import Path

TOL = 90.0
PROBE_OFFSETS = [20, 40, 60]   # seconds after goal time
BALL_X_LO, BALL_X_HI = 0.40, 0.60
BALL_Y_LO, BALL_Y_HI = 0.40, 0.60
MIN_PERSONS = 18
MAX_HALF_IMBALANCE = 0.4   # |L-R| / total


def aggregate_relaxed(labels):
    labs = sorted(labels, key=lambda x: x[0])
    if any(l == "celebration" for _, l in labs):
        return "GOAL"
    for i, (_, l) in enumerate(labs):
        if l == "goal":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL"
    for i, (_, l) in enumerate(labs):
        if l == "kickoff_restart":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL"
            for _, before in labs[:i]:
                if before in ("goal", "celebration", "set_piece"):
                    return "GOAL"
    return "NO"


def load_frames(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            rows.extend(json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())
    rows.sort(key=lambda r: r["t"])
    return rows


def get_frame_near(frames, t, tol=3.0):
    """Find cached frame closest to t within tol seconds."""
    best = None
    for f in frames:
        if abs(f["t"] - t) <= tol:
            if best is None or abs(f["t"] - t) < abs(best["t"] - t):
                best = f
    return best


def probe_passes(frame, ball_required=True, person_min=MIN_PERSONS, balance_max=MAX_HALF_IMBALANCE):
    """Does this single frame satisfy the kickoff-scene conditions?"""
    if frame is None:
        return False
    if frame.get("total_field", 0) < person_min:
        return False
    tot = max(1, frame.get("total_field", 0))
    imb = abs(frame.get("p_left", 0) - frame.get("p_right", 0)) / tot
    if imb > balance_max:
        return False
    ball = frame.get("ball")
    if ball_required:
        if not ball:
            return False
        bx, by = ball[0], ball[1]
        if not (BALL_X_LO <= bx <= BALL_X_HI and BALL_Y_LO <= by <= BALL_Y_HI):
            return False
    return True


def candidate_passes(det, frames, ball_required=True):
    """KEEP if ANY probe at +20/+40/+60s passes all conditions."""
    base = det["start_sec"]
    for off in PROBE_OFFSETS:
        f = get_frame_near(frames, base + off)
        if probe_passes(f, ball_required=ball_required):
            return True
    return False


GAMES = {
    "game_22": {
        "formations": ["/tmp/kickoff_game_22_formation_v2_base.jsonl"],
        "frame_files": ["/tmp/kickoff_game_22_frames.jsonl"],
        "pattern": None,
        "gts": [g + 195 for g in [1559.7, 2195.4]] +
               [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]],
    },
    "game_21": {
        "formations": ["/tmp/kickoff_game_21_formation_v2_base.jsonl"],
        "frame_files": ["/tmp/kickoff_game_21_frames.jsonl"],
        "pattern": None,
        "gts": [g + 65 for g in [1578.6, 2008.6]],
    },
    "game_20": {
        "formations": ["/tmp/kickoff_game20_1H_formation_base.jsonl",
                       "/tmp/kickoff_game20_2H_formation_base.jsonl"],
        "frame_files": ["/tmp/kickoff_game20_1H_frames.jsonl",
                        "/tmp/kickoff_game20_2H_frames.jsonl"],
        "pattern": None,
        "gts": [g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
               [g + 775 for g in [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]],
    },
    "rush": {
        "formations": ["/tmp/kickoff_rush_formation_v2_base.jsonl"],
        "frame_files": ["/tmp/kickoff_rush_frames.jsonl"],
        "pattern": "/tmp/kickoff_rush_pattern_v11_0191.jsonl",
        "gts": [g + 418 for g in [383.1, 647.0]] +
               [g + 418 + 770 for g in [3038.4, 3263.5]],
    },
}


def load_all_goals(cfg):
    out = []
    for fp in cfg["formations"]:
        if not Path(fp).exists():
            continue
        for line in Path(fp).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if aggregate_relaxed(r.get("_vlm_labels", [])) == "GOAL":
                out.append(r)
    if cfg.get("pattern") and Path(cfg["pattern"]).exists():
        for line in Path(cfg["pattern"]).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_vlm_verdict") == "GOAL":
                out.append(r)
    return out


def gt_level_score(dets, gts, tol=TOL):
    tp = 0
    used = set()
    for g in gts:
        for i, d in enumerate(dets):
            if i in used:
                continue
            if abs(d["start_sec"] - g) <= tol:
                tp += 1
                used.add(i)
                break
    fp = len(dets) - len(used)
    return tp, fp


print(f"{'mode':<32} {'TP':>3} {'FP':>4} {'recall':>6} {'prec':>6}")
print("-" * 60)
for label, ball_req, person_min, balance_max in [
    ("none (baseline)", None, None, None),
    ("verifier (ball+persons+bal)", True, 18, 0.4),
    ("verifier no-ball", False, 18, 0.4),
    ("verifier persons-only ≥18", False, 18, 1.0),
    ("verifier balance only ≤0.4", False, 0, 0.4),
    ("verifier strict ball+18+0.3", True, 18, 0.3),
    ("verifier ball OR balance ≤0.3", "OR", 0, 0.3),
]:
    total_tp = 0
    total_fp = 0
    total_gts = 0
    for game, cfg in GAMES.items():
        frames = load_frames(cfg["frame_files"])
        dets = load_all_goals(cfg)
        if ball_req is None:
            kept = dets
        elif ball_req == "OR":
            kept = []
            for d in dets:
                base = d["start_sec"]
                ok = False
                for off in PROBE_OFFSETS:
                    f = get_frame_near(frames, base + off)
                    if f is None:
                        continue
                    # ball central
                    ball = f.get("ball")
                    ball_ok = (ball and BALL_X_LO <= ball[0] <= BALL_X_HI
                               and BALL_Y_LO <= ball[1] <= BALL_Y_HI)
                    tot = max(1, f.get("total_field", 0))
                    imb = abs(f.get("p_left", 0) - f.get("p_right", 0)) / tot
                    bal_ok = imb <= balance_max
                    if ball_ok or bal_ok:
                        ok = True
                        break
                if ok:
                    kept.append(d)
        else:
            kept = []
            for d in dets:
                base = d["start_sec"]
                ok = False
                for off in PROBE_OFFSETS:
                    f = get_frame_near(frames, base + off)
                    if probe_passes(f, ball_required=ball_req,
                                    person_min=person_min, balance_max=balance_max):
                        ok = True
                        break
                if ok:
                    kept.append(d)
        tp, fp = gt_level_score(kept, cfg["gts"])
        total_tp += tp
        total_fp += fp
        total_gts += len(cfg["gts"])
    prec = total_tp / max(1, total_tp + total_fp)
    rec = total_tp / max(1, total_gts)
    print(f"{label:<32} {total_tp:>3} {total_fp:>4} {rec:>6.2f} {prec:>6.2f}")
