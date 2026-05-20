"""Test FP-reduction filters at GT-level (do we keep at least 1 det per GT?)."""
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


GAMES = {
    "game_22": {
        "formations": ["/tmp/kickoff_game_22_formation_v2_base.jsonl"],
        "pattern": None,
        "gts": [g + 195 for g in [1559.7, 2195.4]] +
               [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]],
    },
    "game_21": {
        "formations": ["/tmp/kickoff_game_21_formation_v2_base.jsonl"],
        "pattern": None,
        "gts": [g + 65 for g in [1578.6, 2008.6]],
    },
    "game_20": {
        "formations": ["/tmp/kickoff_game20_1H_formation_base.jsonl",
                       "/tmp/kickoff_game20_2H_formation_base.jsonl"],
        "pattern": None,
        "gts": [g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
               [g + 775 for g in [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]],
    },
    "rush": {
        "formations": ["/tmp/kickoff_rush_formation_v2_base.jsonl"],
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
    """A GT is caught if AT LEAST 1 detection is within tol."""
    tp_gts = 0
    matched_dets = set()
    for g in gts:
        for i, d in enumerate(dets):
            if i in matched_dets:
                continue
            if abs(d["start_sec"] - g) <= tol:
                tp_gts += 1
                matched_dets.add(i)
                break
    fp_dets = len(dets) - len(matched_dets)
    return tp_gts, fp_dets


FILTERS = [
    ("none (current)",
     lambda d: True),
    ("kr_count >= 2",
     lambda d: sum(1 for _, l in d.get("_vlm_labels", []) if l == "kickoff_restart") >= 2),
    ("kr_count >= 2 OR goal label",
     lambda d: (sum(1 for _, l in d.get("_vlm_labels", []) if l == "kickoff_restart") >= 2
                or any(l == "goal" for _, l in d.get("_vlm_labels", [])))),
    ("kr cluster <=40s apart",
     lambda d: (lambda kr: len(kr) >= 2 and min(kr[i+1]-kr[i] for i in range(len(kr)-1)) <= 40 if len(kr) >= 2 else False)(sorted(o for o, l in d.get("_vlm_labels", []) if l == "kickoff_restart"))),
    ("kr OR goal at middle (offsets -20..+20)",
     lambda d: any(l in ("kickoff_restart", "goal") and -20 <= o <= 20 for o, l in d.get("_vlm_labels", []))),
    ("kr at start AND middle (kr at -60 or -40 AND kr at 0 or +20 or +40)",
     lambda d: (any(l == "kickoff_restart" and o in (-60, -40) for o, l in d.get("_vlm_labels", []))
                and any(l == "kickoff_restart" and o in (0, 20, 40) for o, l in d.get("_vlm_labels", [])))),
    ("kr in window AND at least 4 active_play",
     lambda d: (any(l == "kickoff_restart" for _, l in d.get("_vlm_labels", []))
                and sum(1 for _, l in d.get("_vlm_labels", []) if l == "active_play") >= 4)),
]

print(f"{'filter':<55} {'TP':>3} {'FP':>4} {'recall':>6} {'prec':>6}")
print("-" * 80)
for label, fn in FILTERS:
    total_tp = 0
    total_fp = 0
    total_gts = 0
    for game, cfg in GAMES.items():
        dets = [d for d in load_all_goals(cfg) if fn(d)]
        tp, fp = gt_level_score(dets, cfg["gts"])
        total_tp += tp
        total_fp += fp
        total_gts += len(cfg["gts"])
    prec = total_tp / max(1, total_tp + total_fp)
    rec = total_tp / max(1, total_gts)
    print(f"{label:<55} {total_tp:>3} {total_fp:>4} {rec:>6.2f} {prec:>6.2f}")
