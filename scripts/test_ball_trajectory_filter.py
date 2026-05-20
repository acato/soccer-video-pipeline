"""Test ball-trajectory filter: was the ball at a GOAL AREA in the 30-90s
preceding each candidate? Real goals require ball reaching goal mouth.
"""
import json
from pathlib import Path

TOL = 90.0


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


def ball_at_goal_area(frame, x_extreme=0.15, y_lo=0.30, y_hi=0.70):
    """Ball within x ≤ x_extreme OR x ≥ 1-x_extreme, AND y in [lo, hi]."""
    ball = frame.get("ball")
    if not ball:
        return False
    bx, by = ball[0], ball[1]
    if not (y_lo <= by <= y_hi):
        return False
    return bx <= x_extreme or bx >= 1.0 - x_extreme


def has_ball_at_goal_before(det, frames, lookback_lo=30, lookback_hi=120, x_extreme=0.15):
    """Was the ball at a goal area in the [lookback_lo, lookback_hi] window
    before the candidate's start_sec?"""
    base = det["start_sec"]
    t_start = base - lookback_hi
    t_end = base - lookback_lo
    for f in frames:
        if t_start <= f["t"] <= t_end:
            if ball_at_goal_area(f, x_extreme=x_extreme):
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


print(f"{'filter':<48} {'TP':>3} {'FP':>4} {'recall':>6} {'prec':>6}")
print("-" * 70)
for label, x_extreme, lo, hi in [
    ("none (baseline)", None, None, None),
    ("ball at x<=0.15 OR >=0.85 in -90s..-30s", 0.15, 30, 90),
    ("ball at x<=0.10 OR >=0.90 in -90s..-30s", 0.10, 30, 90),
    ("ball at x<=0.20 OR >=0.80 in -120s..-30s", 0.20, 30, 120),
    ("ball at x<=0.10 OR >=0.90 in -120s..-15s", 0.10, 15, 120),
    ("ball at x<=0.15 OR >=0.85 in -60s..-15s", 0.15, 15, 60),
]:
    total_tp = 0
    total_fp = 0
    total_gts = 0
    for game, cfg in GAMES.items():
        frames = load_frames(cfg["frame_files"])
        dets = load_all_goals(cfg)
        if x_extreme is None:
            kept = dets
        else:
            kept = [d for d in dets if has_ball_at_goal_before(
                d, frames, lookback_lo=lo, lookback_hi=hi, x_extreme=x_extreme)]
        tp, fp = gt_level_score(kept, cfg["gts"])
        total_tp += tp
        total_fp += fp
        total_gts += len(cfg["gts"])
    prec = total_tp / max(1, total_tp + total_fp)
    rec = total_tp / max(1, total_gts)
    print(f"{label:<48} {total_tp:>3} {total_fp:>4} {rec:>6.2f} {prec:>6.2f}")
