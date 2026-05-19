"""Compare base FP8 vs v11 LoRA on game_22 formation candidates."""
import json
from pathlib import Path

base = {r["start_sec"]: r for r in [json.loads(l) for l in
        Path("/tmp/kickoff_game_22_formation_base.jsonl").read_text().splitlines() if l.strip()]}
v11 = {r["start_sec"]: r for r in [json.loads(l) for l in
       Path("/tmp/kickoff_game_22_formation_vlm.jsonl").read_text().splitlines() if l.strip()]}

print(f"base confirmed: {sum(1 for r in base.values() if r.get('_vlm_verdict')=='GOAL')}/{len(base)}")
print(f"v11 confirmed:  {sum(1 for r in v11.values() if r.get('_vlm_verdict')=='GOAL')}/{len(v11)}")
print()

print("=== candidates near GT goals (1690, 1735 near 1755; 5795 near 5738) ===")
for t in [1690.0, 1735.0, 5795.0]:
    if t in base and t in v11:
        b = base[t]
        v = v11[t]
        bl = " ".join(f"{o:+d}:{l[:5]}" for o,l in b.get("_vlm_labels", []))
        vl = " ".join(f"{o:+d}:{l[:5]}" for o,l in v.get("_vlm_labels", []))
        print(f"  t={t:.0f} BASE={b.get('_vlm_verdict')}  {bl}")
        print(f"  t={t:.0f} V11 ={v.get('_vlm_verdict')}  {vl}")
        print()

print("=== ALL base GOAL confirmations ===")
for t in sorted(base.keys()):
    r = base[t]
    if r.get("_vlm_verdict") == "GOAL":
        labels = " ".join(f"{o:+d}:{l[:5]}" for o,l in r.get("_vlm_labels", []))
        print(f"  t={t:.0f}  {labels}  ({r.get('_vlm_reason','')[:50]})")

print()
print("=== ALL v11 GOAL confirmations (for comparison) ===")
for t in sorted(v11.keys()):
    r = v11[t]
    if r.get("_vlm_verdict") == "GOAL":
        labels = " ".join(f"{o:+d}:{l[:5]}" for o,l in r.get("_vlm_labels", []))
        print(f"  t={t:.0f}  {labels}  ({r.get('_vlm_reason','')[:50]})")

print()
print("=== overlap analysis ===")
base_g = {t for t,r in base.items() if r.get('_vlm_verdict')=='GOAL'}
v11_g = {t for t,r in v11.items() if r.get('_vlm_verdict')=='GOAL'}
print(f"both confirmed:  {sorted(base_g & v11_g)}")
print(f"base only:       {sorted(base_g - v11_g)}")
print(f"v11 only:        {sorted(v11_g - base_g)}")
