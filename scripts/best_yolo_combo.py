"""Best YOLO-only filter combination. Reports the precision/recall frontier."""
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


def load_all(form_paths, pat_path=None):
    out = []
    for fp in form_paths:
        if not Path(fp).exists():
            continue
        for line in Path(fp).read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            if aggregate_relaxed(r.get("_vlm_labels", [])) == "GOAL":
                out.append(r)
    if pat_path and Path(pat_path).exists():
        for line in Path(pat_path).read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("_vlm_verdict") == "GOAL":
                out.append(r)
    return out


def dedup60(dets):
    out = []
    for d in sorted(dets, key=lambda x: x["start_sec"]):
        if out and (d["start_sec"] - out[-1]["start_sec"]) <= 60:
            continue
        out.append(d)
    return out


def gt_score(dets, gts, tol=TOL):
    tp = 0; used = set()
    for g in gts:
        for i, d in enumerate(dets):
            if i in used: continue
            if abs(d["start_sec"] - g) <= tol:
                tp += 1; used.add(i); break
    return tp, len(dets) - len(used)


def sustained_wide_after(d, frames):
    base = d["start_sec"]
    window = [f for f in frames if base + 15 <= f["t"] <= base + 60]
    if len(window) < 3: return False
    return sum(1 for f in window if f.get("wide_shot")) / len(window) >= 0.6


def ball_at_center_ever(d, frames, window=20, min_count=1):
    base = d["start_sec"]
    c = 0
    for f in frames:
        if abs(f["t"] - base) <= window:
            ball = f.get("ball")
            if ball and CENTER_X_LO <= ball[0] <= CENTER_X_HI \
               and CENTER_Y_LO <= ball[1] <= CENTER_Y_HI:
                c += 1
    return c >= min_count


def player_count_stable(d, frames, window=15, max_std=5):
    base = d["start_sec"]
    counts = [f.get("total_field", 0) for f in frames if abs(f["t"] - base) <= window]
    if len(counts) < 3: return False
    mean = sum(counts) / len(counts)
    var = sum((x - mean) ** 2 for x in counts) / len(counts)
    return var ** 0.5 <= max_std and mean >= 18


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
    ("none", lambda d, f: True),
    ("60s dedup only", "DEDUP"),
    ("60s dedup + sustained_wide", "DEDUP_SW"),
    ("60s dedup + sustained_wide + player_count_stable", "DEDUP_SW_PCS"),
    ("60s dedup + ball_center_ever (±20s)", "DEDUP_BC"),
    ("60s dedup + sustained_wide + ball_center", "DEDUP_SW_BC"),
]


print(f"{'pipeline':<55} {'TP':>3} {'FP':>4} {'recall':>6} {'prec':>6} {'F1':>5}")
print("-" * 80)
for label, mode in FILTERS:
    total_tp = 0; total_fp = 0; total_gts = 0
    for game, (form_paths, pat_path, frame_paths, gts) in GAMES.items():
        frames = load_frames(frame_paths)
        dets = load_all(form_paths, pat_path)
        if mode == "DEDUP":
            dets = dedup60(dets)
        elif mode == "DEDUP_SW":
            dets = [d for d in dets if sustained_wide_after(d, frames)]
            dets = dedup60(dets)
        elif mode == "DEDUP_SW_PCS":
            dets = [d for d in dets if sustained_wide_after(d, frames) and player_count_stable(d, frames)]
            dets = dedup60(dets)
        elif mode == "DEDUP_BC":
            dets = [d for d in dets if ball_at_center_ever(d, frames)]
            dets = dedup60(dets)
        elif mode == "DEDUP_SW_BC":
            dets = [d for d in dets if sustained_wide_after(d, frames) and ball_at_center_ever(d, frames)]
            dets = dedup60(dets)
        tp, fp = gt_score(dets, gts)
        total_tp += tp; total_fp += fp; total_gts += len(gts)
    rec = total_tp / total_gts
    prec = total_tp / max(1, total_tp + total_fp)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(f"{label:<55} {total_tp:>3} {total_fp:>4} {rec:>6.2f} {prec:>6.2f} {f1:>5.2f}")
