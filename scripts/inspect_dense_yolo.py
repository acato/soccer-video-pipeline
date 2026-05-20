"""Inspect dense YOLO output to understand why strict filter dropped everything."""
import json
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/kickoff_game_22_dense.jsonl").read_text().splitlines() if l.strip()]
print(f"loaded {len(rows)} candidates")

# How many candidates have ANY ball-detected frame in their window?
n_with_ball = 0
n_with_ball_central = 0
n_with_wide = 0
n_with_wide_central_inC1to3 = 0
n_with_wide_central_inC1to6 = 0
n_with_wide_central_inC1to10 = 0

# Look at all dense-passing entries
for r in rows:
    base = r["start_sec"]
    pf = r.get("_dense_passing", [])
    # Wait, _dense_passing is only the PASSING frames (those that satisfied strict)
    # If 0/58 passed, no dense_passing entries.
    # I need to look at the raw per_frame_all data. But that's not saved per-candidate.
    # I'll have to re-extract this from the candidate file's _vlm_labels (which is at 10s offsets, not 1s)
    pass

# Better: look at known TP candidates' surroundings.
# game_22 TPs from earlier: candidates 1740, 4625, 5765, 5955, 1740 / 4625 / 5765 / 5955 etc
# We don't have direct per-frame YOLO output... unless dense_yolo_filter saved more

# Let me look at the candidates that fell at GT goal video times
GT_VIDEO = {1755: "1H#1", 2390: "1H#2", 4620: "2H#1", 5738: "2H#2", 5952: "2H#3"}
for gt, name in GT_VIDEO.items():
    near = [r for r in rows if abs(r["start_sec"] - gt) <= 90]
    print(f"\n{name} GT {gt}: {len(near)} candidates within 90s")
    for r in near[:3]:
        passing = r.get("_dense_passing", [])
        print(f"  t={r['start_sec']:.0f}  dense_passing={len(passing)}  _dense_verdict={r.get('_dense_verdict')}")
