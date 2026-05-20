"""Inspect game_21 second-pass results."""
import json
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/kickoff_game_21_2pass.jsonl").read_text().splitlines() if l.strip()]
g = [r for r in rows if r.get("_2pass_verdict") == "GOAL"]
n = [r for r in rows if r.get("_2pass_verdict") == "NO"]
amb = [r for r in rows if r.get("_2pass_verdict") not in ("GOAL", "NO")]
print(f"GOAL: {len(g)} | NO: {len(n)} | other: {len(amb)} | total: {len(rows)}")
print("\nSample GOAL replies:")
for r in g[:5]:
    print(f"  t={r['start_sec']:.0f} reply={r.get('_2pass_reply', '')[:150]}")
print("\nSample NO replies:")
for r in n[:5]:
    print(f"  t={r['start_sec']:.0f} reply={r.get('_2pass_reply', '')[:150]}")
print("\nAmbiguous (if any):")
for r in amb[:3]:
    print(f"  t={r['start_sec']:.0f} reply={r.get('_2pass_reply', '')[:200]}")

print("\n=== Score against GT (game_21: 1644, 2074) ===")
gts = [1644, 2074]
for gt in gts:
    near_g = [r for r in g if abs(r['start_sec'] - gt) <= 90]
    near_n = [r for r in n if abs(r['start_sec'] - gt) <= 90]
    near_all = near_g + near_n + [r for r in amb if abs(r['start_sec'] - gt) <= 90]
    if near_g:
        print(f"  GT {gt}: CAUGHT — {len(near_g)} GOAL of {len(near_all)} near dets")
    else:
        print(f"  GT {gt}: LOST — {len(near_n)} NO of {len(near_all)} near dets")
        for r in near_n[:2]:
            print(f"    rejected: t={r['start_sec']:.0f} reply={r.get('_2pass_reply','')[:120]}")
