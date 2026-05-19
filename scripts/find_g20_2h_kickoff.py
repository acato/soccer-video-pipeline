"""Find game_20 actual 2H kickoff by scanning for return-of-game-activity."""
import json
from pathlib import Path

# Merge both scan files since 1H scan covers 0-3550 and 2H scan covers 3554-7229
rows_1h = [json.loads(l) for l in Path("/tmp/kickoff_game20_1H_frames.jsonl").read_text().splitlines() if l.strip()]
rows_2h = [json.loads(l) for l in Path("/tmp/kickoff_game20_2H_frames.jsonl").read_text().splitlines() if l.strip()]
rows = rows_1h + rows_2h
rows.sort(key=lambda r: r["t"])

# Find sustained wide_shot region after the long halftime gap
print(f"=== game_20 looking for 2H start (after halftime) ===")
# Find first frame after t=2700 where there's a sustained wide_shot stretch (≥3 wide in next 5 samples)
i = 0
while i < len(rows):
    r = rows[i]
    if r["t"] < 2700 or not r.get("wide_shot"):
        i += 1
        continue
    # Check if next 5 samples are mostly wide
    window = rows[i:i + 6]
    wide_count = sum(1 for w in window if w.get("wide_shot"))
    if wide_count >= 4:
        print(f"  2H likely starts at t={r['t']:.0f} (next 6 frames: {wide_count}/6 wide)")
        # Show context
        for w in window:
            tot = w.get("total_field", 0)
            print(f"    t={w['t']:.0f} wide={w.get('wide_shot')} total={tot} ball={w.get('ball') is not None}")
        break
    i += 1
