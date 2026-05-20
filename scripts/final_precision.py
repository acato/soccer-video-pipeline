"""Final precision/recall: ensemble of formation_base (relaxed rule) + pattern_v11.

Uses the corrected game_20 calibration (offset_1H=100, halftime=675).
"""
import json
from pathlib import Path

TOL = 90.0
DEDUP = 30.0


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


def load_confirmed(path, rule="relaxed"):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if rule == "relaxed":
            if aggregate_relaxed(r.get("_vlm_labels", [])) == "GOAL":
                out.append(r)
        else:
            if r.get("_vlm_verdict") == "GOAL":
                out.append(r)
    return out


def dedup(dets):
    out = []
    for d in sorted(dets, key=lambda x: x["start_sec"]):
        if out and (d["start_sec"] - out[-1]["start_sec"]) <= DEDUP:
            continue
        out.append(d)
    return out


def score(dets, gt_times, tol=TOL):
    used = set()
    tp = 0
    for d in sorted(dets, key=lambda x: x["start_sec"]):
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
    return tp, len(dets) - tp, len(gt_times) - tp


GAMES = {
    "game_22": {
        "formation": "/tmp/kickoff_game_22_formation_v2_base.jsonl",
        "pattern": None,  # not yet run on 0.19.1
        "gts": [g + 195 for g in [1559.7, 2195.4]] +
               [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]],
    },
    "game_21": {
        "formation": "/tmp/kickoff_game_21_formation_v2_base.jsonl",
        "pattern": None,
        "gts": [g + 65 for g in [1578.6, 2008.6]],
    },
    "game_20": {
        # 1H + 2H scans combined
        "formation_1h": "/tmp/kickoff_game20_1H_formation_base.jsonl",
        "formation_2h": "/tmp/kickoff_game20_2H_formation_base.jsonl",
        "pattern": None,
        "gts": [g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
               [g + 775 for g in [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]],
    },
    "rush": {
        "formation": "/tmp/kickoff_rush_formation_v2_base.jsonl",
        "pattern": "/tmp/kickoff_rush_pattern_v11_0191.jsonl",
        "gts": [g + 418 for g in [383.1, 647.0]] +
               [g + 418 + 770 for g in [3038.4, 3263.5]],
    },
}

agg = [0, 0, 0]  # tp fp fn
print(f"{'game':<10} {'formation':>10} {'pattern':>9} {'combined':>10} {'TP':>3} {'FP':>3} {'FN':>3}")
for game, cfg in GAMES.items():
    if "formation_1h" in cfg:
        form = load_confirmed(cfg["formation_1h"]) + load_confirmed(cfg["formation_2h"])
    else:
        form = load_confirmed(cfg["formation"])
    pat = load_confirmed(cfg["pattern"]) if cfg.get("pattern") else []
    combined = dedup(form + pat)

    tp, fp, fn = score(combined, cfg["gts"])
    agg[0] += tp; agg[1] += fp; agg[2] += fn
    print(f"{game:<10} {len(form):>10} {len(pat):>9} {len(combined):>10} "
          f"{tp:>3} {fp:>3} {fn:>3}")

tp, fp, fn = agg
prec = tp / max(1, tp + fp)
rec = tp / max(1, tp + fn)
f1 = 2 * prec * rec / max(1e-9, prec + rec)
print(f"\nTOTAL  TP={tp} FP={fp} FN={fn}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
