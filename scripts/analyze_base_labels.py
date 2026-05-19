"""Analyze base model label patterns — would a looser aggregation rule pick up TPs?"""
import json
from pathlib import Path
from collections import Counter

GAMES = [
    ("game_22", "/tmp/kickoff_game_22_formation_base.jsonl",
     [1754.7, 2390.4, 4619.7, 5738.3, 5951.6]),
]

for game, path, gt_vid in GAMES:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    print(f"=== {game}: {len(rows)} candidates ===")

    # Overall label frequency
    all_labels = Counter()
    for r in rows:
        for _, lbl in r.get("_vlm_labels", []):
            all_labels[lbl] += 1
    print(f"label freq: {dict(all_labels.most_common())}")
    print()

    # How many candidates have kickoff_restart at least once / twice
    n_with_kr = sum(1 for r in rows
                    if any(lbl == "kickoff_restart" for _, lbl in r.get("_vlm_labels", [])))
    n_with_kr2 = sum(1 for r in rows
                     if sum(1 for _, lbl in r.get("_vlm_labels", []) if lbl == "kickoff_restart") >= 2)
    print(f"candidates with kickoff_restart ≥1: {n_with_kr}")
    print(f"candidates with kickoff_restart ≥2: {n_with_kr2}")
    print()

    # For ≥2 kickoff_restart rule, which would be GOAL? Score against GT.
    new_goals = []
    for r in rows:
        labels = r.get("_vlm_labels", [])
        kr_count = sum(1 for _, lbl in labels if lbl == "kickoff_restart")
        # Compose stricter rule: ≥2 kickoff_restart OR existing rules
        has_celeb = any(l == "celebration" for _, l in labels)
        has_goal = any(l == "goal" for _, l in labels)
        if kr_count >= 2 or has_celeb or has_goal:
            new_goals.append((r["start_sec"], kr_count, labels))

    print(f"=== under looser rule (≥2 kickoff_restart, OR celebration, OR goal): {len(new_goals)} candidates ===")
    for t, kr, labels in sorted(new_goals):
        labels_str = " ".join(f"{o:+d}:{l[:5]}" for o, l in labels)
        nearest_gt = min((abs(t - g), g) for g in gt_vid)
        is_tp = nearest_gt[0] <= 90
        marker = "TP" if is_tp else "FP"
        print(f"  {marker} t={t:.0f}  kr={kr}  near GT {nearest_gt[1]:.0f} (diff {nearest_gt[0]:.0f}s)  {labels_str}")
