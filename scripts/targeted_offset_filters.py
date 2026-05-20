"""Try filters based on label position. TPs vs FPs from earlier analysis showed
that kickoff_restart at certain offsets (-40, 0, +20) is more enriched in TPs.
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


GAMES = {
    "game_22": ("/tmp/kickoff_game_22_formation_v2_base.jsonl", None,
                [g + 195 for g in [1559.7, 2195.4]] +
                [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]]),
    "game_21": ("/tmp/kickoff_game_21_formation_v2_base.jsonl", None,
                [g + 65 for g in [1578.6, 2008.6]]),
    "game_20_1h": ("/tmp/kickoff_game20_1H_formation_base.jsonl", None,
                   [g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
                   [g + 775 for g in [2442.0]]),
    "game_20_2h": ("/tmp/kickoff_game20_2H_formation_base.jsonl", None,
                   [g + 775 for g in [3106.4, 3596.0, 3976.8, 4065.2]]),
    "rush_form": ("/tmp/kickoff_rush_formation_v2_base.jsonl", None,
                  [g + 418 for g in [383.1, 647.0]] +
                  [g + 418 + 770 for g in [3038.4, 3263.5]]),
    "rush_pat": (None, "/tmp/kickoff_rush_pattern_v11_0191.jsonl",
                 [g + 418 for g in [383.1, 647.0]] +
                 [g + 418 + 770 for g in [3038.4, 3263.5]]),
}


def load(formation_path, pattern_path):
    out = []
    if formation_path and Path(formation_path).exists():
        for line in Path(formation_path).read_text().splitlines():
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


def score(dets, gts, tol=TOL):
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
    return tp, len(dets) - len(used)


def has_kr_at(d, offsets):
    return any(l == "kickoff_restart" and o in offsets
               for o, l in d.get("_vlm_labels", []))


def has_strong_label_at(d, offsets):
    return any(l in ("kickoff_restart", "goal", "celebration") and o in offsets
               for o, l in d.get("_vlm_labels", []))


FILTERS = [
    ("none (baseline)", lambda d: True),
    ("kr at +20", lambda d: has_kr_at(d, {20})),
    ("kr at -40", lambda d: has_kr_at(d, {-40})),
    ("kr at 0", lambda d: has_kr_at(d, {0})),
    ("kr at -40 OR +20", lambda d: has_kr_at(d, {-40, 20})),
    ("kr at -40 OR 0 OR +20", lambda d: has_kr_at(d, {-40, 0, 20})),
    ("kr at -40 AND +20", lambda d: has_kr_at(d, {-40}) and has_kr_at(d, {20})),
    ("strong label at -40 OR 0 OR +20",
     lambda d: has_strong_label_at(d, {-40, 0, 20})),
    ("≥2 kr in {-40, 0, +20}",
     lambda d: sum(1 for o, l in d.get("_vlm_labels", []) if l == "kickoff_restart" and o in {-40, 0, 20}) >= 2),
]

print(f"{'filter':<42} {'TP':>3} {'FP':>4} {'recall':>6} {'prec':>6}")
print("-" * 65)
for label, fn in FILTERS:
    total_tp = 0
    total_fp = 0
    total_gts = 0
    for game, (fp, pat, gts) in GAMES.items():
        dets = [d for d in load(fp, pat) if fn(d)]
        tp, fpr = score(dets, gts)
        total_tp += tp
        total_fp += fpr
        total_gts += len(gts)
    # GTs are duplicated across rush_form and rush_pat
    unique_gts = 20
    rec = total_tp / max(1, unique_gts) if total_tp <= unique_gts else 1.0
    # Better: dedup-aware recall
    prec = total_tp / max(1, total_tp + total_fp)
    print(f"{label:<42} {total_tp:>3} {total_fp:>4} {rec:>6.2f} {prec:>6.2f}")

# More accurate: combine rush_form + rush_pat, dedup
print("\n=== combined per-game (formation + pattern_v11 + 60s dedup) ===")
print(f"{'filter':<42} {'TP':>3} {'FP':>4} {'recall':>6} {'prec':>6}")
print("-" * 65)
for label, fn in FILTERS:
    total_tp = 0
    total_fp = 0
    for game in ("game_22", "game_21", "game_20_1h", "game_20_2h"):
        fp, pat, gts = GAMES[game]
        dets = [d for d in load(fp, pat) if fn(d)]
        tp, fpr = score(dets, gts)
        total_tp += tp; total_fp += fpr
    # rush combined
    fp, _, gts_r = GAMES["rush_form"]
    _, pat, _ = GAMES["rush_pat"]
    rush_dets = [d for d in load(fp, pat) if fn(d)]
    # dedup within 60s
    rush_dets.sort(key=lambda d: d["start_sec"])
    rush_kept = []
    for d in rush_dets:
        if rush_kept and (d["start_sec"] - rush_kept[-1]["start_sec"]) <= 60:
            continue
        rush_kept.append(d)
    tp, fpr = score(rush_kept, gts_r)
    total_tp += tp; total_fp += fpr
    rec = total_tp / 20
    prec = total_tp / max(1, total_tp + total_fp)
    print(f"{label:<42} {total_tp:>3} {total_fp:>4} {rec:>6.2f} {prec:>6.2f}")
