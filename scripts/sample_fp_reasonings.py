"""Sample FP reasonings for a given event type from a Run on Rush.

Identifies which detected events did NOT match a GT event (likely FPs)
and prints their reasoning text so we can see what triggered the
hallucination.
"""
import json
import sys
from pathlib import Path

events_path = sys.argv[1]
event_type = sys.argv[2]
gt_eval_json_path = sys.argv[3] if len(sys.argv) > 3 else None

events = [json.loads(l) for l in open(events_path)]
events = [e for e in events if e["event_type"] == event_type]
print(f"Total {event_type} detected: {len(events)}")

# If we have an eval JSON, we can extract matched indices. Otherwise
# print all reasonings — caller filters mentally.
if gt_eval_json_path and Path(gt_eval_json_path).exists():
    eval_data = json.load(open(gt_eval_json_path))
    # FN events list contains GT events that were never matched. We want
    # the inverse: detected events whose reasoning suggests FP. For now
    # just print all detection reasonings since we have no per-detection
    # match flag in the eval JSON.

print(f"\n=== Sample reasonings for {event_type} ===")
for e in events[:30]:
    r = e.get("reasoning", "")
    t = e.get("start_sec", 0)
    print(f"\nt={t:.1f}s")
    print(f"  {r[:300]}")
