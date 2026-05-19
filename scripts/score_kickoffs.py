"""Score kickoff-pattern goal detections against analytics GT.

GT goals come from "Goals Conceded" events (one per scored goal). GT timestamps
are in game-time-ms; detections are in video-time-sec. We auto-fit a per-half
linear offset (video = game + offset) by maximizing TP count within tol=30s,
then report TP/FP/FN.
"""
import argparse
import json
from pathlib import Path


def load_gt_goals(gt_path: Path):
    """Return list of game-time seconds for actual goals in this half file."""
    data = json.loads(gt_path.read_text())["data"]
    goals = []
    for entry in data:
        t_game_sec = entry["event_time"] / 1000.0
        for ev in entry.get("events", []):
            if ev.get("event_name") == "Goals Conceded":
                goals.append(t_game_sec)
    return sorted(goals)


def load_detections(det_path: Path):
    return sorted(
        json.loads(line)["start_sec"]
        for line in det_path.read_text().splitlines()
        if line.strip()
    )


def best_offset(dets, gt_game_times, search_range=(-600, 800), step=1, tol=30.0):
    """Find offset that maps gt_game_time -> video_time maximizing TP.

    Returns (best_offset, tp_count) or (None, 0) if either list is empty.
    """
    if not gt_game_times or not dets:
        return None, 0
    best = (None, -1)
    for off in range(search_range[0], search_range[1] + 1, step):
        gt_vid = [g + off for g in gt_game_times]
        tp = 0
        used = set()
        for d in dets:
            for i, g in enumerate(gt_vid):
                if i in used:
                    continue
                if abs(d - g) <= tol:
                    tp += 1
                    used.add(i)
                    break
        if tp > best[1]:
            best = (off, tp)
    return best


def score(dets, gt_video_times, tol=30.0):
    """Greedy nearest-match TP / FP / FN with tolerance."""
    used = set()
    tp_pairs = []
    fp = []
    for d in dets:
        match = None
        for i, g in enumerate(gt_video_times):
            if i in used:
                continue
            if abs(d - g) <= tol:
                if match is None or abs(d - g) < abs(d - gt_video_times[match]):
                    match = i
        if match is not None:
            used.add(match)
            tp_pairs.append((d, gt_video_times[match]))
        else:
            fp.append(d)
    fn = [g for i, g in enumerate(gt_video_times) if i not in used]
    return tp_pairs, fp, fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--det", required=True, help="detections jsonl")
    p.add_argument("--gt-1h", required=True)
    p.add_argument("--gt-2h", default=None)
    p.add_argument("--tol", type=float, default=30.0)
    p.add_argument("--offset-1h", type=float, default=None,
                   help="override fitted offset for 1H")
    p.add_argument("--offset-2h", type=float, default=None)
    args = p.parse_args()

    dets = load_detections(Path(args.det))

    halves = [("1H", args.gt_1h, args.offset_1h)]
    if args.gt_2h:
        halves.append(("2H", args.gt_2h, args.offset_2h))

    gt_video_all = []
    print(f"=== {args.label} ===")
    print(f"detections ({len(dets)}): {[round(d,1) for d in dets]}")
    for half_name, gt_path, off_override in halves:
        gt_game = load_gt_goals(Path(gt_path))
        if off_override is not None:
            off = off_override
            tp_count = sum(
                1 for g in gt_game if any(abs((g + off) - d) <= args.tol for d in dets)
            )
        else:
            off, tp_count = best_offset(dets, gt_game, tol=args.tol)
        gt_vid = [g + off for g in gt_game] if off is not None else []
        print(f"  {half_name}: gt_game_goals={[round(g,1) for g in gt_game]}  "
              f"best_offset={off}s  gt_video={[round(v,1) for v in gt_vid]}  "
              f"fit_tp={tp_count}")
        gt_video_all.extend(gt_vid)

    tp_pairs, fp, fn = score(dets, sorted(gt_video_all), tol=args.tol)
    print(f"  RESULT: TP={len(tp_pairs)}  FP={len(fp)}  FN={len(fn)}")
    if tp_pairs:
        print(f"    TP pairs (det → gt): {[(round(d,1), round(g,1)) for d,g in tp_pairs]}")
    if fp:
        print(f"    FP detections: {[round(x,1) for x in fp]}")
    if fn:
        print(f"    FN missed GT goals: {[round(x,1) for x in fn]}")
    prec = len(tp_pairs) / (len(tp_pairs) + len(fp)) if (tp_pairs or fp) else 0.0
    rec = len(tp_pairs) / (len(tp_pairs) + len(fn)) if (tp_pairs or fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"    precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")


if __name__ == "__main__":
    main()
