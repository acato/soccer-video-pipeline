"""Print a compact summary of an evaluate_detection.py --json-out result."""
import json
import sys

d = json.load(open(sys.argv[1]))
o = d["overall"]
print(f"F1={o['f1']:.3f}  P={o['precision']:.3f}  R={o['recall']:.3f}")
print(f"TP={o['tp']}  FN={o['fn']}  FP={o['fp']}")
print(f"GT total: {d['gt_total']}   Detected total: {d['detected_total']}")
print("\nGT counts:", d["gt_counts"])
print("\nDetected counts:", d["detected_counts"])
print("\nPer-type:")
for t, m in d["per_type"].items():
    if t == "__overall__":
        continue
    print(f"  {t:22s} gt={m['gt']:3d} det={m['detected']:3d} "
          f"TP={m['tp']:3d} FP={m['fp']:3d} "
          f"P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
