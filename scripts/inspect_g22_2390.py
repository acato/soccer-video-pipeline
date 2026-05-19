"""Inspect game_22 formation candidates near GT 2390."""
import json
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/kickoff_game_22_formation_v2_base.jsonl").read_text().splitlines() if l.strip()]
print(f"=== game_22 v2 base candidates within 2300-2500 ===")
for r in rows:
    t = r["start_sec"]
    if 2300 <= t <= 2500:
        labels = r.get("_vlm_labels", [])
        labels_str = " ".join(f"{o:+d}:{l[:5]}" for o, l in labels)
        verdict = r.get("_vlm_verdict", "?")
        cs = r.get("_cluster_start", "?")
        ce = r.get("_cluster_end", "?")
        size = r.get("_cluster_size", "?")
        print(f"  t={t:.0f} ({verdict}) size={size} span={cs}-{ce}")
        print(f"    labels: {labels_str}")
        print(f"    reason: {r.get('_vlm_reason','')[:80]}")
