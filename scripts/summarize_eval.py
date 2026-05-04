"""Print Run-N eval summary table from a json-out result file."""
import json, sys
d = json.load(sys.stdin)
o = d["overall"]
print(f"Overall: tp={o['tp']} fn={o['fn']} fp={o['fp']} P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}")
print()
hdr = ("type", "gt", "det", "tp", "fn", "fp", "P", "R", "F1")
print(f"{hdr[0]:<22}{hdr[1]:>5}{hdr[2]:>5}{hdr[3]:>5}{hdr[4]:>5}{hdr[5]:>5}{hdr[6]:>7}{hdr[7]:>7}{hdr[8]:>7}")
for t, v in d["per_type"].items():
    if t.startswith("__"):
        continue
    print(f"{t:<22}{v['gt']:>5}{v['detected']:>5}{v['tp']:>5}{v['fn']:>5}{v['fp']:>5}{v['precision']:>7.2f}{v['recall']:>7.2f}{v['f1']:>7.2f}")
print()
for t, v in d["per_type"].items():
    if t.startswith("__"):
        print(f"{t:<22}            tp={v['tp']:>3} fn={v['fn']:>3} fp={v['fp']:>3} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")
