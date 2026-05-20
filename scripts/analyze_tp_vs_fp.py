"""Analyze label patterns of TPs vs FPs in the 100%-recall ensemble.

Look for signals that distinguish goal-aftermath kickoffs from regular
balanced midfield play. Tests several candidate filtering rules.
"""
import json
from pathlib import Path
from collections import Counter

TOL = 90.0


def aggregate_relaxed(labels):
    labs = sorted(labels, key=lambda x: x[0])
    if any(l == "celebration" for _, l in labs):
        return "GOAL", "celebration"
    for i, (_, l) in enumerate(labs):
        if l == "goal":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL", "goal_then_play"
    for i, (_, l) in enumerate(labs):
        if l == "kickoff_restart":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL", "kickoff_then_play"
            for _, before in labs[:i]:
                if before in ("goal", "celebration", "set_piece"):
                    return "GOAL", "preceded_then_kickoff"
    return "NO", "no_signal"


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
            verdict, reason = aggregate_relaxed(r.get("_vlm_labels", []))
            if verdict == "GOAL":
                r["_aggregation_reason"] = reason
                out.append(r)
    if cfg.get("pattern") and Path(cfg["pattern"]).exists():
        for line in Path(cfg["pattern"]).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_vlm_verdict") == "GOAL":
                r["_aggregation_reason"] = "pattern_v11"
                out.append(r)
    return out


def is_tp(det, gts):
    return any(abs(det["start_sec"] - g) <= TOL for g in gts)


# Collect TPs and FPs
all_tps = []
all_fps = []
for game, cfg in GAMES.items():
    dets = load_all_goals(cfg)
    for d in dets:
        d["_game"] = game
        if is_tp(d, cfg["gts"]):
            all_tps.append(d)
        else:
            all_fps.append(d)

print(f"TPs: {len(all_tps)}  FPs: {len(all_fps)}\n")

# Aggregation reason breakdown
print("=== aggregation reason distribution ===")
tp_reasons = Counter(d["_aggregation_reason"] for d in all_tps)
fp_reasons = Counter(d["_aggregation_reason"] for d in all_fps)
all_reasons = set(tp_reasons) | set(fp_reasons)
for r in sorted(all_reasons):
    tp = tp_reasons[r]; fp = fp_reasons[r]
    rate = tp / max(1, tp + fp)
    print(f"  {r:<25} TP={tp:>2} FP={fp:>3}  precision={rate:.2f}")

# Label statistics
print("\n=== label position analysis (TPs only) ===")
tp_at_each = {off: Counter() for off in [-60, -40, -20, 0, 20, 40, 60]}
for d in all_tps:
    for off, lbl in d.get("_vlm_labels", []):
        if off in tp_at_each:
            tp_at_each[off][lbl] += 1
for off in sorted(tp_at_each):
    print(f"  t{off:+d}: {dict(tp_at_each[off].most_common(5))}")

print("\n=== label position analysis (FPs only) ===")
fp_at_each = {off: Counter() for off in [-60, -40, -20, 0, 20, 40, 60]}
for d in all_fps:
    for off, lbl in d.get("_vlm_labels", []):
        if off in fp_at_each:
            fp_at_each[off][lbl] += 1
for off in sorted(fp_at_each):
    print(f"  t{off:+d}: {dict(fp_at_each[off].most_common(5))}")

# Distinguishing features
print("\n=== test filter: require kickoff_restart frame count >= 2 ===")
def kr_count(d):
    return sum(1 for _, l in d.get("_vlm_labels", []) if l == "kickoff_restart")
tp_kr_pass = sum(1 for d in all_tps if kr_count(d) >= 2)
fp_kr_pass = sum(1 for d in all_fps if kr_count(d) >= 2)
print(f"  TPs passing: {tp_kr_pass}/{len(all_tps)}  FPs passing: {fp_kr_pass}/{len(all_fps)}")

print("\n=== test filter: require kickoff_restart at >= 2 distinct offsets within 40s of each other ===")
def kr_cluster(d):
    kr_offs = [o for o, l in d.get("_vlm_labels", []) if l == "kickoff_restart"]
    if len(kr_offs) < 2:
        return False
    kr_offs.sort()
    for i in range(len(kr_offs) - 1):
        if kr_offs[i + 1] - kr_offs[i] <= 40:
            return True
    return False
tp_pass = sum(1 for d in all_tps if kr_cluster(d))
fp_pass = sum(1 for d in all_fps if kr_cluster(d))
print(f"  TPs passing: {tp_pass}/{len(all_tps)}  FPs passing: {fp_pass}/{len(all_fps)}")

print("\n=== test filter: NO 'set_piece' anywhere in window ===")
def no_set_piece(d):
    return not any(l == "set_piece" for _, l in d.get("_vlm_labels", []))
tp_pass = sum(1 for d in all_tps if no_set_piece(d))
fp_pass = sum(1 for d in all_fps if no_set_piece(d))
print(f"  TPs passing: {tp_pass}/{len(all_tps)}  FPs passing: {fp_pass}/{len(all_fps)}")

print("\n=== combo filter: relaxed AND (kickoff_restart count >= 2 OR celebration/goal label) ===")
def combo_filter(d):
    labels = d.get("_vlm_labels", [])
    if any(l in ("celebration", "goal") for _, l in labels):
        return True
    kr = sum(1 for _, l in labels if l == "kickoff_restart")
    return kr >= 2
tp_pass = sum(1 for d in all_tps if combo_filter(d))
fp_pass = sum(1 for d in all_fps if combo_filter(d))
print(f"  TPs passing: {tp_pass}/{len(all_tps)}  FPs passing: {fp_pass}/{len(all_fps)}")
