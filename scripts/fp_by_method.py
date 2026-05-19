"""Categorize detections by method, surface rush FPs."""
import json
from collections import Counter
from pathlib import Path

GAMES = ["game_22", "game_21", "rush", "game20_1H"]
for game in GAMES:
    f = Path(f"/tmp/kickoff_{game}_v6.jsonl")
    if not f.exists():
        continue
    dets = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    methods = Counter(d["_method"].split("_")[-1] for d in dets)
    print(f"{game}: {len(dets)} total, methods={dict(methods)}")

print("\n=== rush FPs by method (TPs at 990, 4230 excluded) ===")
rush = [json.loads(l) for l in Path("/tmp/kickoff_rush_v6.jsonl").read_text().splitlines() if l.strip()]
tps = {990.0, 4230.0}
fps_by_method = Counter()
for d in rush:
    if d["start_sec"] not in tps:
        method = d["_method"].split("_")[-1]
        fps_by_method[method] += 1
        t = d["start_sec"]
        print(f"  t={t:.0f}  method={method}")
print(f"\n  FP method breakdown: {dict(fps_by_method)}")
