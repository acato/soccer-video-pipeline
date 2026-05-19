"""Inspect game_22 GT 5952 window and simulate relaxed inC threshold."""
import json
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/kickoff_game_22_frames.jsonl").read_text().splitlines() if l.strip()]
nearby = [r for r in rows if abs(r["t"] - 5952) <= 90]
print(f"=== GT 5952 window ({len(nearby)} frames) ===")
for r in nearby:
    pl, pr = r.get("p_left", 0), r.get("p_right", 0)
    diff = abs(pl - pr)
    inc = r.get("in_circle", 0)
    tot = r.get("total_field", 0)
    wide = "Y" if r.get("wide_shot") else "n"
    cur = (r.get("wide_shot") and 18 <= tot <= 30 and diff <= 5 and 0 <= inc <= 5)
    relax15 = (r.get("wide_shot") and 18 <= tot <= 30 and diff <= 5 and 0 <= inc <= 15)
    relax7lr = (r.get("wide_shot") and 18 <= tot <= 30 and diff <= 7 and 0 <= inc <= 15)
    mark = ""
    if cur:
        mark = " <- CUR fires"
    elif relax15:
        mark = " <- relaxed inC<=15 fires"
    elif relax7lr:
        mark = " <- relax inC<=15 + |L-R|<=7 fires"
    print(f"  t={r['t']:.0f} wide={wide} pL={pl:>2} pR={pr:>2} |L-R|={diff:>2} inC={inc:>2} tot={tot:>2}{mark}")
