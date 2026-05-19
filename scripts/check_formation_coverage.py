"""Does the formation-candidate pool cover GT goal positions?"""
import json
from pathlib import Path

# Calibrated GT video times (from prior calibration; offset_1H from goal-match)
GAMES = {
    "game_22": {
        "candidates": "/tmp/kickoff_game_22_formation.jsonl",
        # offset_1H=195 (calibrated kickoff via density+65s), halftime=690
        "gt_video": [1754.7, 2390.4, 4619.7, 5738.3, 5951.6],
    },
    "game_21": {
        "candidates": "/tmp/kickoff_game_21_formation.jsonl",
        # offset_1H=65
        "gt_video": [1643.6, 2073.6],
    },
    "rush": {
        "candidates": "/tmp/kickoff_rush_formation.jsonl",
        # offset_1H=418, halftime=770 -- canonical
        "gt_video": [801.1, 1065.0, 4226.4, 4451.5],
    },
}

for game, cfg in GAMES.items():
    cands = [json.loads(l) for l in Path(cfg["candidates"]).read_text().splitlines() if l.strip()]
    print(f"\n=== {game}: {len(cands)} candidates ===")
    print(f"  candidate times: {[round(c['start_sec'], 0) for c in cands]}")
    for gt in cfg["gt_video"]:
        close = [c for c in cands if abs(c["start_sec"] - gt) <= 90]
        if close:
            for c in close:
                cs = c.get("_cluster_start", c["start_sec"])
                ce = c.get("_cluster_end", c["start_sec"])
                size = c.get("_cluster_size", 1)
                print(f"  GT {gt:.0f} -> candidate {c['start_sec']:.0f} (size {size}, span {cs:.0f}-{ce:.0f})")
        else:
            print(f"  GT {gt:.0f} -> no candidate within 90s")
