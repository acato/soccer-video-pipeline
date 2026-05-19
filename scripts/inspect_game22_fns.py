"""Inspect game_22 per-frame data near the 3 FN GT goals.

GT video times (offset_1H=195, halftime=690):
  2390 (game-clock 2195.4, 1H goal 2)
  4620 (game-clock 3734.7, 2H goal 1)
  5952 (game-clock 5066.6, 2H goal 3)

Each FN missed because no formation candidate exists within ±90s. Check
what total_field / p_left / p_right / in_circle / wide_shot / ball_at_center
look like in a wider window around each FN.
"""
import json
from pathlib import Path

FRAMES = "/tmp/kickoff_game_22_frames.jsonl"
FNS = [2390, 4620, 5952]

rows = [json.loads(l) for l in Path(FRAMES).read_text().splitlines() if l.strip()]

for fn_t in FNS:
    print(f"\n=== GT at video {fn_t} (±90s window) ===")
    nearby = [r for r in rows if abs(r["t"] - fn_t) <= 90]
    print(f"  {len(nearby)} frames in window")
    print(f"  {'t':>5} {'wide':>5} {'pL':>3} {'pR':>3} {'|L-R|':>5} {'inC':>3} {'tot':>3} {'bcent':>5}")
    for r in nearby:
        wide = "Y" if r.get("wide_shot") else "n"
        pl = r.get("p_left", 0)
        pr = r.get("p_right", 0)
        diff = abs(pl - pr)
        inc = r.get("in_circle", 0)
        tot = r.get("total_field", 0)
        bc = "Y" if r.get("ball_at_center") else "n"
        marker = ""
        # Would CURRENT formation fire? wide + 18 <= total <= 30 + |L-R| <= 5 + 0 <= inC <= 5
        cur_fire = (r.get("wide_shot") and 18 <= tot <= 30 and diff <= 5 and 0 <= inc <= 5)
        if cur_fire:
            marker = " <- CUR fires"
        else:
            reasons = []
            if not r.get("wide_shot"): reasons.append("not_wide")
            if not (18 <= tot <= 30): reasons.append(f"tot={tot}")
            if diff > 5: reasons.append(f"|L-R|={diff}")
            if not (0 <= inc <= 5): reasons.append(f"inC={inc}")
            marker = " <- miss: " + ",".join(reasons)
        print(f"  {r['t']:>5.0f} {wide:>5} {pl:>3} {pr:>3} {diff:>5} {inc:>3} {tot:>3} {bc:>5}{marker}")
