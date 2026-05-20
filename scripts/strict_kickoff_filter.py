"""Test: require the STRICT kickoff_setup_strong condition near the candidate.

kickoff_setup_strong = wide_shot AND ball_at_center AND 1 <= in_circle <= 3.
This is the original strict signal the detector started with. The relaxed
formation generator skips it. Applying it as a precision gate on the
relaxed candidates could be the cleanest non-LLM separator.

Also try: motion-based stability filters (player count stable across frames).
"""
import json
from pathlib import Path

TOL = 90.0

CENTER_X_LO, CENTER_X_HI = 0.40, 0.60
CENTER_Y_LO, CENTER_Y_HI = 0.35, 0.50


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


def load_all_goals(formation_paths, pattern_path=None):
    out = []
    for fp in formation_paths:
        if not Path(fp).exists():
            continue
        for line in Path(fp).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if aggregate_relaxed(r.get("_vlm_labels", [])) == "GOAL":
                out.append(r)
    if pattern_path and Path(pattern_path).exists():
        for line in Path(pattern_path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_vlm_verdict") == "GOAL":
                out.append(r)
    return out


def gt_score(dets, gts, tol=TOL):
    tp = 0; used = set()
    for g in gts:
        for i, d in enumerate(dets):
            if i in used:
                continue
            if abs(d["start_sec"] - g) <= tol:
                tp += 1; used.add(i); break
    return tp, len(dets) - len(used)


def frame_is_strict_kickoff(f, in_c_min=1, in_c_max=3):
    """Original kickoff_setup_strong: wide + ball_at_center + 1<=in_circle<=3."""
    if not f.get("wide_shot"):
        return False
    ball = f.get("ball")
    if not ball:
        return False
    bx, by = ball[0], ball[1]
    if not (CENTER_X_LO <= bx <= CENTER_X_HI and CENTER_Y_LO <= by <= CENTER_Y_HI):
        return False
    inc = f.get("in_circle", 0)
    return in_c_min <= inc <= in_c_max


def filt_strict_nearby(det, frames, window_sec=15, in_c_min=1, in_c_max=3):
    """Pass if ANY frame within ±window_sec of candidate is strict kickoff."""
    base = det["start_sec"]
    for f in frames:
        if abs(f["t"] - base) <= window_sec:
            if frame_is_strict_kickoff(f, in_c_min, in_c_max):
                return True
    return False


def filt_player_count_stable(det, frames, window_sec=15, max_std=5.0):
    """Pass if total_field stays roughly constant across frames near the candidate.
    Real kickoff: ~22 players visible, stable. Active play: variable."""
    base = det["start_sec"]
    nearby = [f.get("total_field", 0) for f in frames if abs(f["t"] - base) <= window_sec]
    if len(nearby) < 3:
        return False
    mean = sum(nearby) / len(nearby)
    var = sum((x - mean) ** 2 for x in nearby) / len(nearby)
    std = var ** 0.5
    return std <= max_std and mean >= 18


def filt_ball_at_center_sustained(det, frames, window_sec=10, min_count=2):
    """Ball at center in ≥min_count frames within ±window_sec of candidate."""
    base = det["start_sec"]
    count = 0
    for f in frames:
        if abs(f["t"] - base) <= window_sec:
            ball = f.get("ball")
            if ball:
                bx, by = ball[0], ball[1]
                if CENTER_X_LO <= bx <= CENTER_X_HI and CENTER_Y_LO <= by <= CENTER_Y_HI:
                    count += 1
    return count >= min_count


GAMES = {
    "game_22": (["/tmp/kickoff_game_22_formation_v2_base.jsonl"], None,
                ["/tmp/kickoff_game_22_frames.jsonl"],
                [g + 195 for g in [1559.7, 2195.4]] +
                [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]]),
    "game_21": (["/tmp/kickoff_game_21_formation_v2_base.jsonl"], None,
                ["/tmp/kickoff_game_21_frames.jsonl"],
                [g + 65 for g in [1578.6, 2008.6]]),
    "game_20": (["/tmp/kickoff_game20_1H_formation_base.jsonl",
                 "/tmp/kickoff_game20_2H_formation_base.jsonl"], None,
                ["/tmp/kickoff_game20_1H_frames.jsonl",
                 "/tmp/kickoff_game20_2H_frames.jsonl"],
                [g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
                [g + 775 for g in [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]]),
    "rush": (["/tmp/kickoff_rush_formation_v2_base.jsonl"],
             "/tmp/kickoff_rush_pattern_v11_0191.jsonl",
             ["/tmp/kickoff_rush_frames.jsonl"],
             [g + 418 for g in [383.1, 647.0]] +
             [g + 418 + 770 for g in [3038.4, 3263.5]]),
}

FILTERS = [
    ("none (baseline)", lambda d, f: True),
    ("strict kickoff (wide+ball_c+inC1-3) ±15s", filt_strict_nearby),
    ("strict kickoff (in_circle 1-5) ±15s",
     lambda d, f: filt_strict_nearby(d, f, in_c_max=5)),
    ("strict kickoff ±30s",
     lambda d, f: filt_strict_nearby(d, f, window_sec=30)),
    ("ball at center ≥2 frames ±10s", filt_ball_at_center_sustained),
    ("ball at center ≥2 ±15s",
     lambda d, f: filt_ball_at_center_sustained(d, f, window_sec=15)),
    ("ball at center ≥1 ±20s",
     lambda d, f: filt_ball_at_center_sustained(d, f, window_sec=20, min_count=1)),
    ("player count stable ±15s", filt_player_count_stable),
    ("strict OR ball-center≥1 ±20s",
     lambda d, f: filt_strict_nearby(d, f, in_c_max=5)
                  or filt_ball_at_center_sustained(d, f, window_sec=20, min_count=1)),
]

print(f"{'filter':<48} {'TP':>3} {'FP':>4} {'recall':>6} {'prec':>6}")
print("-" * 70)
for label, fn in FILTERS:
    total_tp = 0; total_fp = 0; total_gts = 0
    for game, (formation_paths, pattern_path, frame_paths, gts) in GAMES.items():
        frames = load_frames(frame_paths)
        dets = [d for d in load_all_goals(formation_paths, pattern_path) if fn(d, frames)]
        tp, fp = gt_score(dets, gts)
        total_tp += tp; total_fp += fp; total_gts += len(gts)
    rec = total_tp / total_gts
    prec = total_tp / max(1, total_tp + total_fp)
    print(f"{label:<48} {total_tp:>3} {total_fp:>4} {rec:>6.2f} {prec:>6.2f}")
