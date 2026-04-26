#!/usr/bin/env python3
"""Phase A of QL1: is VLM-emitted confidence a usable signal?

For a completed run, sweep confidence thresholds and compute precision/recall
per event type. If P rises monotonically with threshold (and R falls), conf
is a useful filter and the refinement loop can use it cheaply. If P stays
flat, conf is noise — Pass 2 needs to actually re-examine frames, not just
filter on the score.

Usage:
    python analyze_confidence_signal.py <events_jsonl> --tolerance 45 \
        [--video-offset 418.0 --half2-start 3916.0 --half2-game-offset 2700.0] \
        [--gt-file ... --gt-file ...]

Output: per-event-type table of (threshold, n_kept, precision, recall, F1)
plus a summary of which types have a useful confidence signal.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the existing evaluator's GT loading + matching logic
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_detection import (
    load_ground_truth,
    GTEvent,
    DetectedEvent,
    GT_TO_PIPELINE,
    GT_FILES,
)


# Constant kept inline so this script doesn't need ../src on sys.path
KEEPER_TYPES = {
    "shot_stop_diving", "shot_stop_standing", "punch", "catch", "goal_kick",
    "distribution_short", "distribution_long", "one_on_one", "corner_kick",
    "penalty", "goal", "shot_on_target",
}
HIGHLIGHTS_TYPES = {
    "shot_on_target", "shot_off_target", "goal", "near_miss",
    "penalty", "free_kick_shot",
}


def load_detected_with_conf(events_path: str) -> list[dict]:
    """Load detected events, keeping confidence + center timestamp."""
    out = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            etype = d.get("event_type", "")
            if not etype or etype == "none":
                continue
            start = d.get("start_sec", d.get("timestamp_start", 0))
            end = d.get("end_sec", d.get("timestamp_end", start))
            out.append({
                "event_type": etype,
                "start": start,
                "end": end,
                "center": (start + end) / 2,
                "confidence": d.get("confidence", 0),
            })
    return out


def label_detections_tp_fp(
    gt_events: list[GTEvent],
    detections: list[dict],
    tolerance_sec: float,
) -> list[dict]:
    """Mark each detection as TP or FP using the same greedy match as
    evaluate_detection.match_events. Returns detections with added 'is_tp' bool.
    """
    gt_by_type = defaultdict(list)
    for g in gt_events:
        gt_by_type[g.event_type].append(g)

    det_by_type = defaultdict(list)
    for i, d in enumerate(detections):
        d["_idx"] = i
        d["is_tp"] = False
        det_by_type[d["event_type"]].append(d)

    for etype in det_by_type:
        gt_list = sorted(gt_by_type.get(etype, []), key=lambda e: e.video_time_sec)
        det_list = sorted(det_by_type[etype], key=lambda d: d["center"])

        matched_det = set()
        for gt in gt_list:
            best_idx = None
            best_dist = float("inf")
            for i, d in enumerate(det_list):
                if i in matched_det:
                    continue
                dist = abs(d["center"] - gt.video_time_sec)
                if dist < best_dist and dist <= tolerance_sec:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                matched_det.add(best_idx)
                det_list[best_idx]["is_tp"] = True

    return detections


def sweep_thresholds(detections: list[dict], gt_events: list[GTEvent]) -> dict:
    """For each event type, return [(threshold, n_kept, tp, fp, precision, recall, f1)].
    Recall denominator is GT count for the type."""
    gt_count = defaultdict(int)
    for g in gt_events:
        gt_count[g.event_type] += 1

    by_type = defaultdict(list)
    for d in detections:
        by_type[d["event_type"]].append(d)

    thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    out = {}
    for etype, dets in by_type.items():
        gt = gt_count.get(etype, 0)
        rows = []
        for thr in thresholds:
            kept = [d for d in dets if d["confidence"] >= thr]
            tp = sum(1 for d in kept if d["is_tp"])
            fp = len(kept) - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / gt if gt > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            rows.append({"thr": thr, "kept": len(kept), "tp": tp, "fp": fp,
                         "precision": p, "recall": r, "f1": f1})
        out[etype] = {"gt": gt, "rows": rows}
    return out


def conf_distribution_signal(table: dict) -> dict:
    """Score how well confidence discriminates TP from FP per type.
    Returns spread between p@conf>=0.9 and p@conf>=0.3 — large positive means
    high-conf detections are noticeably more likely to be TPs."""
    out = {}
    for etype, info in table.items():
        rows = info["rows"]
        p_low = next((r["precision"] for r in rows if r["thr"] == 0.3), None)
        p_high = next((r["precision"] for r in rows if r["thr"] == 0.9), None)
        # Find threshold maximizing F1
        best = max(rows, key=lambda r: r["f1"])
        out[etype] = {
            "gt": info["gt"],
            "p_at_0.3": p_low,
            "p_at_0.9": p_high,
            "spread": (p_high - p_low) if (p_low is not None and p_high is not None) else None,
            "best_f1_threshold": best["thr"],
            "best_f1": best["f1"],
            "f1_at_0_thr": rows[0]["f1"],  # current behavior (no threshold)
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events_file")
    ap.add_argument("--tolerance", type=float, default=45.0)
    ap.add_argument("--video-offset", type=float, default=418.0)
    ap.add_argument("--half2-start", type=float, default=3916.0)
    ap.add_argument("--half2-game-offset", type=float, default=2700.0)
    ap.add_argument("--gt-file", action="append", default=None,
                    help="Override default GT files (Rush) — provide twice for two halves")
    args = ap.parse_args()

    if args.gt_file:
        # Patch the module-level GT_FILES (load_ground_truth reads it)
        from scripts import evaluate_detection
        evaluate_detection.GT_FILES = list(args.gt_file)
        evaluate_detection.VIDEO_OFFSET = args.video_offset

    gt_events = load_ground_truth(
        half2_video_start=args.half2_start,
        half2_game_offset=args.half2_game_offset,
    )
    detections = load_detected_with_conf(args.events_file)
    detections = label_detections_tp_fp(gt_events, detections, args.tolerance)

    table = sweep_thresholds(detections, gt_events)

    print(f"\nConfidence signal analysis: {args.events_file}")
    print(f"  GT events: {len(gt_events)}  Detected (non-none): {len(detections)}")
    print(f"  Tolerance: {args.tolerance}s")
    print()

    # Per-type tables
    for etype in sorted(table.keys()):
        info = table[etype]
        print(f"\n{etype}  (GT={info['gt']})")
        print(f"  {'thr':>5}  {'kept':>4}  {'tp':>4}  {'fp':>4}  {'P':>6}  {'R':>6}  {'F1':>6}")
        for r in info["rows"]:
            print(f"  {r['thr']:>5.2f}  {r['kept']:>4d}  {r['tp']:>4d}  {r['fp']:>4d}  "
                  f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f1']:>6.3f}")

    # Summary: which types have a usable confidence signal?
    sig = conf_distribution_signal(table)
    print("\n\nConfidence-signal summary (does precision rise with threshold?):")
    print(f"  {'type':<20} {'gt':>4} {'P@0.3':>7} {'P@0.9':>7} {'spread':>7} {'best_F1':>8} {'@thr':>6} {'curr_F1':>8}")
    for etype in sorted(sig.keys()):
        s = sig[etype]
        spread_str = f"{s['spread']:+.3f}" if s["spread"] is not None else "n/a"
        p_low = f"{s['p_at_0.3']:.3f}" if s["p_at_0.3"] is not None else "n/a"
        p_high = f"{s['p_at_0.9']:.3f}" if s["p_at_0.9"] is not None else "n/a"
        print(f"  {etype:<20} {s['gt']:>4} {p_low:>7} {p_high:>7} {spread_str:>7} "
              f"{s['best_f1']:>8.3f} {s['best_f1_threshold']:>6.2f} {s['f1_at_0_thr']:>8.3f}")

    # Reel-weighted view: if we threshold all types at the per-type best_F1,
    # what's the effect on the keeper / highlights aggregates?
    print("\n\nReel-weighted: best per-type threshold vs current (no threshold)")
    def reel_metrics(types: set, threshold_per_type: dict | None) -> tuple:
        kept = []
        for d in detections:
            thr = (threshold_per_type or {}).get(d["event_type"], 0.0)
            if d["confidence"] >= thr:
                kept.append(d)
        tp = sum(1 for d in kept if d["is_tp"] and d["event_type"] in types)
        fp = sum(1 for d in kept if (not d["is_tp"]) and d["event_type"] in types)
        gt_count_local = sum(1 for g in gt_events if g.event_type in types)
        fn = gt_count_local - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return tp, fn, fp, p, r, f1

    best_thr = {t: sig[t]["best_f1_threshold"] for t in sig}

    for label, types in [("keeper", KEEPER_TYPES), ("highlights", HIGHLIGHTS_TYPES),
                          ("reels (union)", KEEPER_TYPES | HIGHLIGHTS_TYPES)]:
        tp_c, fn_c, fp_c, p_c, r_c, f1_c = reel_metrics(types, None)
        tp_t, fn_t, fp_t, p_t, r_t, f1_t = reel_metrics(types, best_thr)
        print(f"\n{label}")
        print(f"  current (no thr):  TP {tp_c}  FN {fn_c}  FP {fp_c}  P {p_c:.3f}  R {r_c:.3f}  F1 {f1_c:.3f}")
        print(f"  best per-type thr: TP {tp_t}  FN {fn_t}  FP {fp_t}  P {p_t:.3f}  R {r_t:.3f}  F1 {f1_t:.3f}")
        print(f"    delta:           dF1 {f1_t - f1_c:+.3f}  dTP {tp_t - tp_c:+d}  dFP {fp_t - fp_c:+d}")


if __name__ == "__main__":
    main()
