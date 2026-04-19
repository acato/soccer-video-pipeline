"""Analyze YOLO grounding rejections for a run."""
import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
print(f"TOTAL gated events: {len(rows)}")

by_type = defaultdict(lambda: {"kept": 0, "dropped": 0, "reasons": Counter()})
for r in rows:
    t = r["event_type"]
    key = "kept" if r["keep"] else "dropped"
    by_type[t][key] += 1
    by_type[t]["reasons"][r["reason"]] += 1

for t, d in by_type.items():
    kept = d["kept"]
    dropped = d["dropped"]
    print(f"\n{t}: kept={kept} dropped={dropped}")
    for reason, n in d["reasons"].most_common():
        print(f"   {reason}: {n}")

# Ball detection overall
n_ball = sum(1 for r in rows if r["features"]["ball_detected"])
n_no_ball = len(rows) - n_ball
print(f"\nBall detected: {n_ball}/{len(rows)} ({100*n_ball/len(rows):.1f}%)")
print(f"No-ball (fail-open kept): {n_no_ball}")

# For dropped events, show ball positions
print("\n--- Dropped events — ball position distribution ---")
for t in ["throw_in", "corner_kick", "goal_kick"]:
    drops = [r for r in rows if r["event_type"] == t and not r["keep"]
             and r["features"]["ball_detected"]]
    if not drops:
        continue
    print(f"\n{t} drops (n={len(drops)}):")
    for r in drops[:10]:
        f = r["features"]
        print(f"  ts={r['timestamp_start']:.1f}  "
              f"x={f['ball_x_norm']:.2f} y={f['ball_y_norm']:.2f} "
              f"conf={f['ball_confidence']:.2f}")
