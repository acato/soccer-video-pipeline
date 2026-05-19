"""Check pattern_v11 verifier output: does it catch rush 2H goal at video 4226?"""
import json
from pathlib import Path

rows = [json.loads(l) for l in Path("/tmp/kickoff_rush_pattern_v11_0191.jsonl").read_text().splitlines() if l.strip()]
goals = [r for r in rows if r.get("_vlm_verdict") == "GOAL"]
print(f"rush pattern_v11 (0.19.1) confirmed: {len(goals)}/{len(rows)}")
for r in goals:
    t = r["start_sec"]
    near = abs(t - 4226) <= 90
    mark = " <- catches rush GT 4226" if near else ""
    method = r["_method"].split("_")[-1] if "_method" in r else "?"
    print(f"  t={t}  method={method}{mark}")

# Detail on the 4230 candidate (was a TP earlier under strict rule)
for r in rows:
    if r["start_sec"] == 4230.0:
        print(f"\nrush@4230 candidate: verdict={r.get('_vlm_verdict')}")
        labels = r.get("_vlm_labels", [])
        if labels:
            labels_str = " ".join(f"{o:+d}:{l[:5]}" for o, l in labels)
            print(f"  labels: {labels_str}")
        print(f"  reason: {r.get('_vlm_reason','')[:100]}")
