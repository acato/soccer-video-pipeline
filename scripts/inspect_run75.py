"""Inspect Run 75 retry: Rush eval + verifier outcomes."""
import json
import sys
from collections import Counter

EVAL_FILE = "/Users/aless/soccer-runs/state/run_75_eval.json"
EVENTS = "/tmp/soccer-pipeline/b7953c9a-721e-4de0-bac2-9dba75fa5812/diagnostics/dual_pass_events.jsonl"


def main():
    d = json.load(open(EVAL_FILE))
    o = d["overall"]
    print("=== Rush eval ===")
    print("F1={:.3f}  P={:.3f}  R={:.3f}  TP={} FP={} FN={}".format(
        o["f1"], o["precision"], o["recall"], o["tp"], o["fp"], o["fn"]))
    print()
    print("Per-type:")
    for t, p in d["per_type"].items():
        if "gt" not in p:
            continue
        print("  {:20s} gt={:3d} det={:3d} tp={:3d} fp={:3d} R={:.2f} P={:.2f} F1={:.2f}".format(
            t, p["gt"], p["detected"], p["tp"], p["fp"], p["recall"], p["precision"], p["f1"]))

    print()
    print("=== events ===")
    bpv = Counter()
    kv = Counter()
    methods = Counter()
    goals = []
    n = 0
    for line in open(EVENTS):
        e = json.loads(line)
        n += 1
        md = e.get("metadata") or {}
        if e["event_type"] == "goal":
            goals.append(e)
            methods[md.get("detection_method", "?")] += 1
        if "ball_presence_verifier_outcome" in md:
            bpv[md["ball_presence_verifier_outcome"]] += 1
        if "kickoff_verifier_outcome" in md:
            kv[md["kickoff_verifier_outcome"]] += 1
    print("total events:", n)
    print("goals:", len(goals), "methods:", dict(methods))
    print("BPV outcomes:", dict(bpv))
    print("KV  outcomes:", dict(kv))
    print()
    print("Per-goal detail (top-level keys + metadata):")
    if goals:
        print("  top-level keys:", list(goals[0].keys()))
        print("  metadata keys :", list((goals[0].get("metadata") or {}).keys()))
    for g in goals:
        md = g.get("metadata") or {}
        ts0 = g.get("timestamp_start") or g.get("start_sec") or g.get("start_seconds") or 0
        ts1 = g.get("timestamp_end") or g.get("end_sec") or g.get("end_seconds") or 0
        meth = md.get("detection_method", "?")
        print("  ts={:.0f}-{:.0f}  method={}  md_keys={}".format(
            ts0, ts1, meth, list(md.keys())[:10]))


if __name__ == "__main__":
    sys.exit(main())
