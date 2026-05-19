"""Check v2 formation candidate coverage of GT goals across 3 games."""
import json
from pathlib import Path

GAMES = {
    "game_22": ("/tmp/kickoff_game_22_formation_v2.jsonl",
                [1755, 2390, 4620, 5738, 5952]),
    "game_21": ("/tmp/kickoff_game_21_formation_v2.jsonl", [1644, 2074]),
    "rush":    ("/tmp/kickoff_rush_formation_v2.jsonl",
                [801, 1042, 4226, 4451]),
}
for game, (path, gts) in GAMES.items():
    cands = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    print(f"\n=== {game}: {len(cands)} candidates ===")
    for gt in gts:
        close = [c["start_sec"] for c in cands if abs(c["start_sec"] - gt) <= 90]
        marker = "" if close else " <- NOT COVERED"
        print(f"  GT {gt}: {close}{marker}")
