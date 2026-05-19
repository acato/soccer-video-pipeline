"""Inspect game_20 per-frame data around GT 2731.

GT 2H goal at game-clock 2442 (42s into 2H). 1H ends at game-clock 2400.
Halftime then 2H kickoff. Post-goal kickoff would be ~30-60s after goal.

Look for the half-time boundary (low-activity stretch) + 2H kickoff +
post-goal kickoff to understand why no formation candidate fired.
"""
import json
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/kickoff_game20_1H_frames.jsonl").read_text().splitlines() if l.strip()]
# Window 2400-2900 (around expected halftime + 2H start + GT 2731 + post-goal kickoff)
nearby = [r for r in rows if 2400 <= r["t"] <= 2900]
print(f"=== game_20 frames 2400-2900s ({len(nearby)} samples) ===")
print(f"  {'t':>5} {'wide':>5} {'pL':>3} {'pR':>3} {'|L-R|':>5} {'inC':>3} {'tot':>3} {'ball':>15}")
for r in nearby:
    pl, pr = r.get("p_left", 0), r.get("p_right", 0)
    diff = abs(pl - pr)
    inc = r.get("in_circle", 0)
    tot = r.get("total_field", 0)
    wide = "Y" if r.get("wide_shot") else "n"
    ball = r.get("ball")
    bxy = f"({ball[0]:.2f},{ball[1]:.2f})" if ball else "None"
    # v2 formation: wide + 18 <= tot <= 30 + |L-R| <= 5 + 0 <= inC <= 15
    cur_fire = (r.get("wide_shot") and 18 <= tot <= 30 and diff <= 5 and 0 <= inc <= 15)
    marker = " <- formation" if cur_fire else ""
    # Halftime marker: low activity
    if not r.get("wide_shot") and not ball:
        marker += " // non-game"
    print(f"  {r['t']:>5.0f} {wide:>5} {pl:>3} {pr:>3} {diff:>5} {inc:>3} {tot:>3} {bxy:>15}{marker}")
