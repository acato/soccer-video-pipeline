"""Find game_20 2H kickoff timing from per-frame data."""
import json
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/kickoff_game20_1H_frames.jsonl").read_text().splitlines() if l.strip()]
print(f"=== frames 2780-2950 ===")
for r in rows:
    t = r["t"]
    if 2780 <= t <= 2950:
        pl, pr = r.get("p_left", 0), r.get("p_right", 0)
        diff = abs(pl - pr)
        inc = r.get("in_circle", 0)
        tot = r.get("total_field", 0)
        wide = "Y" if r.get("wide_shot") else "n"
        cur = r.get("wide_shot") and 18 <= tot <= 30 and diff <= 5 and 0 <= inc <= 15
        ball = r.get("ball")
        bxy = f"({ball[0]:.2f},{ball[1]:.2f})" if ball else "None"
        m = " <- formation" if cur else ""
        print(f"  t={t:.0f} wide={wide} L={pl} R={pr} diff={diff} inC={inc} tot={tot} ball={bxy}{m}")
