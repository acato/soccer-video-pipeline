"""Inspect game_21 v3 second-pass (kickoff_pattern prompt) results."""
import json
from pathlib import Path
from collections import Counter

rows = [json.loads(l) for l in Path("/tmp/kickoff_game_21_2pass_v3.jsonl").read_text().splitlines() if l.strip()]
verdicts = Counter(r.get("_2pass_verdict") for r in rows)
print(f"Verdicts: {dict(verdicts)} (total {len(rows)})")

print("\n=== GOAL confirmed ===")
for r in rows:
    if r.get("_2pass_verdict") == "GOAL":
        print(f"  t={r['start_sec']:.0f}  reply={r.get('_2pass_reply', '')[:200]}")

print("\n=== Sample NOs ===")
n_count = 0
for r in rows:
    if r.get("_2pass_verdict") == "NO" and n_count < 5:
        print(f"  t={r['start_sec']:.0f}  reply={r.get('_2pass_reply', '')[:200]}")
        n_count += 1

print("\n=== Score vs GT (game_21: 1644, 2074) ===")
for gt in [1644, 2074]:
    near_g = [r for r in rows if r.get("_2pass_verdict") == "GOAL" and abs(r['start_sec'] - gt) <= 90]
    near_n = [r for r in rows if r.get("_2pass_verdict") == "NO" and abs(r['start_sec'] - gt) <= 90]
    near_all = near_g + near_n
    if near_g:
        print(f"  GT {gt}: CAUGHT — {len(near_g)} GOAL of {len(near_all)}")
    else:
        print(f"  GT {gt}: LOST — {len(near_n)} NOs near, all rejected")
        for r in near_n[:2]:
            print(f"    rejected t={r['start_sec']:.0f}  reply={r.get('_2pass_reply','')[:160]}")
