"""Filter formation candidates whose ±60s window would exceed video bounds."""
import json
import sys
from pathlib import Path

# Conservative video durations (slightly under actual ends)
DUR = {"game_21": 7140, "game_22": 7140, "rush": 6785, "game_20": 7140}

in_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
game = sys.argv[3]
limit = DUR[game]
cands = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
kept = [c for c in cands if c["start_sec"] + 60 < limit and c["start_sec"] - 60 > 0]
out_path.write_text("\n".join(json.dumps(c) for c in kept) + "\n")
print(f"  {game}: {len(cands)} -> {len(kept)} (filtered {len(cands)-len(kept)} late/early candidates)")
