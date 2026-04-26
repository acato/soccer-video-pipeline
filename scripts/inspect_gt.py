"""Inspect structure of a GT JSON (event type distribution + time range)."""
import json
import sys
import os
from collections import Counter

folder = sys.argv[1]
for fn in sorted(os.listdir(folder)):
    if not fn.endswith(".json"):
        continue
    with open(os.path.join(folder, fn)) as f:
        d = json.load(f)["data"]
    print(f"\n=== {fn} ({len(d)} events) ===")
    if not d:
        continue
    first = d[0]
    last = d[-1]
    print(f"First event keys: {list(first.keys())}")
    print(f"First event_time={first.get('event_time')} period_order={first.get('period_order')}")
    print(f"Last event_time={last.get('event_time')} period_order={last.get('period_order')}")
    types = Counter(e.get("event_name") for e in d)
    print("Top 10 event_name:")
    for t, n in types.most_common(10):
        print(f"   {n:4d}  {t}")
