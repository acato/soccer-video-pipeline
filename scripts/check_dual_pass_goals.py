"""Check dual_pass goal detections for all 4 games and AND-fuse with ensemble."""
import json
from pathlib import Path

GAMES = {
    "game_20": ("/tmp/soccer-pipeline/a0f8f93c-8611-466b-934c-8cd48a2aee00/events.jsonl",
                [g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
                [g + 775 for g in [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]]),
    "game_22": ("/tmp/soccer-pipeline/031ee71a-7a4f-4605-a642-9bff003e4804/events.jsonl",
                [g + 195 for g in [1559.7, 2195.4]] +
                [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]]),
    "game_21": ("/tmp/soccer-pipeline/60ed91b3-ebcb-4ddb-ac18-768923a17419/events.jsonl",
                [g + 65 for g in [1578.6, 2008.6]]),
    "rush": ("/tmp/soccer-pipeline/47d1358c-268c-4577-b4e1-f9943f51be6a/events.jsonl",
             [g + 418 for g in [383.1, 647.0]] +
             [g + 418 + 770 for g in [3038.4, 3263.5]]),
}

TOL = 90.0

def score(dets, gts, tol=TOL):
    used = set()
    tp = 0
    pairs = []
    for d in sorted(dets):
        best_i, best_dt = None, float("inf")
        for i, g in enumerate(gts):
            if i in used: continue
            dt = abs(d - g)
            if dt <= tol and dt < best_dt:
                best_i, best_dt = i, dt
        if best_i is not None:
            used.add(best_i); tp += 1
            pairs.append((d, gts[best_i], best_dt))
    fp = len(dets) - tp
    fn = len(gts) - tp
    return tp, fp, fn, pairs

print(f"{'game':<10} {'#dets':>6} {'TP':>3} {'FP':>3} {'FN':>3} {'recall':>6} {'prec':>6}")
print("-" * 60)
all_dp = {}
for game, (path, gts) in GAMES.items():
    p = Path(path)
    if not p.exists():
        print(f"{game}: missing {path}")
        continue
    goal_times = []
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("event_type") == "goal":
            goal_times.append(r.get("timestamp_start", r.get("start_sec")))
    tp, fp, fn, pairs = score(sorted(goal_times), gts)
    all_dp[game] = goal_times
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    print(f"{game:<10} {len(goal_times):>6} {tp:>3} {fp:>3} {fn:>3} {rec:>6.2f} {prec:>6.2f}")
    for d, g, diff in pairs:
        print(f"           TP: det {d:.0f} -> GT {g:.0f} (diff {diff:.0f}s)")

# Aggregate
total_tp = 0; total_fp = 0; total_fn = 0
for game, (path, gts) in GAMES.items():
    if game in all_dp:
        tp, fp, fn, _ = score(all_dp[game], gts)
        total_tp += tp; total_fp += fp; total_fn += fn
prec = total_tp / max(1, total_tp + total_fp)
rec = total_tp / max(1, total_tp + total_fn)
f1 = 2*prec*rec/max(1e-9, prec+rec)
print(f"\nTOTAL: TP={total_tp} FP={total_fp} FN={total_fn} prec={prec:.2f} rec={rec:.2f} F1={f1:.2f}")
