"""Find which rush TP was killed by the save filter and tune the window."""
import json
from pathlib import Path

TOL = 90.0

tiered_path = "/tmp/kickoff_rush_tiered_relaxed.jsonl"
events_path = "/tmp/soccer-pipeline/47d1358c-268c-4577-b4e1-f9943f51be6a/events.jsonl"
gts = [g + 418 for g in [383.1, 647.0]] + [g + 418 + 770 for g in [3038.4, 3263.5]]


def _t(e):
    return e.get("start_sec", e.get("timestamp_start"))


tiered = [json.loads(l) for l in Path(tiered_path).read_text().splitlines() if l.strip()]
candidate_goals = [r for r in tiered
                   if r.get("event_type") == "goal"
                   and r.get("metadata", {}).get("goal_tier") == "candidate"]

all_events = [json.loads(l) for l in Path(events_path).read_text().splitlines() if l.strip()]
saves = sorted([(_t(e), e.get("event_type"))
                for e in all_events
                if e.get("event_type") in {"catch", "shot_stop_diving", "shot_stop_standing"}
                and e.get("metadata", {}).get("detection_method", "") in {"dual_pass", "shot_outcome"}])

print(f"rush GT video times: {gts}")
print(f"rush candidates: {len(candidate_goals)}")
print(f"rush save events: {len(saves)}")
print(f"  save times: {[(round(t, 0), kind) for t, kind in saves]}")
print()
print("Each candidate vs GT match + nearest save:")
for c in candidate_goals:
    t = _t(c)
    nearest_gt = min(((abs(t - g), g) for g in gts), default=(99999, None))
    is_tp = nearest_gt[0] <= TOL
    nearest_save = min((abs(t - st), st, kind) for st, kind in saves) if saves else (99999, None, None)
    save_close = nearest_save[0] <= 30
    label = "TP" if is_tp else "FP"
    save_str = f"save at {nearest_save[1]:.0f} ({nearest_save[2]}, {nearest_save[0]:.0f}s)" if nearest_save[1] else "no save"
    block = "BLOCKED" if save_close else "kept"
    print(f"  cand t={t:.0f}  {label}  nearest_gt={nearest_gt[1]} (diff {nearest_gt[0]:.0f}s)  "
          f"nearest_save={save_str}  -> {block}")

# Also try alternative windows
print("\n=== Window sensitivity (rush only) ===")
print(f"{'window':>8} {'kept':>5} {'TP':>3} {'FP':>3}")
for w in [5, 10, 15, 20, 25, 30]:
    kept = []
    for c in candidate_goals:
        t = _t(c)
        if not any(abs(t - st) <= w for st, _ in saves):
            kept.append(t)
    used = set()
    tp = 0
    for d in sorted(kept):
        for i, g in enumerate(gts):
            if i in used: continue
            if abs(d - g) <= TOL:
                used.add(i); tp += 1; break
    fp = len(kept) - tp
    print(f"{w:>8} {len(kept):>5} {tp:>3} {fp:>3}")
