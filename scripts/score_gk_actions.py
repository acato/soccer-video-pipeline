"""Score dual_pass keeper-action detections against GT for all 4 games.

GT lives in /Volumes/transit (raw event-stream schema). Saves are tagged as
event_name="Saves" with property.Type in {"Catches", "Parries"}.

We map:
  GT type "Catches" → detector event_type "catch"
  GT type "Parries" → detector event_type in {shot_stop_diving, shot_stop_standing}
  combined "save"   → all three above

Time mapping: event_time_ms is within-half (0..2400000 = 40min). To convert to
video time:
  video_1H = event_time/1000 + offset_1H
  video_2H = event_time/1000 + offset_2H   (offset_2H = halftime-end in video time)
"""
import json
from pathlib import Path

# Offsets in video time. event_time semantics:
#   1H file: within-half ms (0..2400000)  → video = event_time/1000 + off_1h
#   2H file: CONTINUOUS game-clock from match start (>= 2400000 ms typically)
#            → video = event_time/1000 + off_2h, where off_2h = halftime duration
#              (= offset_1H_kickoff_video + halftime_seconds - 2400)
#
# Off_2h values matched against verified goal offsets:
#   game_20: g+775 for 2H goals  → off_2h = 775
#   game_22: g+885 for 2H goals  → off_2h = 885
#   game_21: density anchor 3105+65s lag = 3170 → off_2h = 3170 - 2400 = 770
#   rush:    g+1188 for 2H goals → off_2h = 1188
GAMES = {
    "game_20": {
        "gt": "/Volumes/transit/Games/20",
        "events": "/tmp/soccer-pipeline/a0f8f93c-8611-466b-934c-8cd48a2aee00/events.jsonl",
        "off_1h": 100.0, "off_2h": 775.0,
    },
    "game_22": {
        "gt": "/Volumes/transit/Games/22",
        "events": "/tmp/soccer-pipeline/031ee71a-7a4f-4605-a642-9bff003e4804/events.jsonl",
        "off_1h": 195.0, "off_2h": 885.0,
    },
    "game_21": {
        "gt": "/Volumes/transit/Games/21",
        "events": "/tmp/soccer-pipeline/60ed91b3-ebcb-4ddb-ac18-768923a17419/events.jsonl",
        "off_1h": 65.0, "off_2h": 770.0,
    },
    "rush": {
        "gt": "/Volumes/transit",
        "events": "/tmp/soccer-pipeline/47d1358c-268c-4577-b4e1-f9943f51be6a/events.jsonl",
        "off_1h": 418.0, "off_2h": 1188.0,
    },
}

CATCH_TYPES = {"catch"}
PARRY_TYPES = {"shot_stop_diving", "shot_stop_standing"}
SAVE_TYPES = CATCH_TYPES | PARRY_TYPES


def _t(e):
    return e.get("start_sec", e.get("timestamp_start"))


def is_real(e):
    return e.get("metadata", {}).get("detection_method", "") in {"dual_pass", "shot_outcome"}


def find_gt_files(d):
    p = Path(d)
    h1 = sorted(p.glob("*_1st Half.json")) + sorted(p.glob("*1st Half.json"))
    h2 = sorted(p.glob("*_2nd Half.json")) + sorted(p.glob("*2nd Half.json"))
    return (h1[0] if h1 else None, h2[0] if h2 else None)


def load_gt_saves(gt_dir, off_1h, off_2h):
    """Return list of (video_time, gt_type, team_name) for every Save event."""
    out = []
    h1_path, h2_path = find_gt_files(gt_dir)
    for path, off in [(h1_path, off_1h), (h2_path, off_2h)]:
        if path is None:
            continue
        data = json.loads(path.read_text())
        for r in data["data"]:
            for e in r["events"]:
                if e["event_name"] == "Saves":
                    ty = e["property"].get("Type")
                    if ty in {"Catches", "Parries"}:
                        out.append((r["event_time"] / 1000.0 + off, ty, r["team_name"]))
    return sorted(out)


def load_detections(path, types):
    return sorted(
        _t(e) for e in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())
        if e.get("event_type") in types and is_real(e)
    )


def score(dets, gts, tol):
    used = set()
    matched_dets, unmatched_dets = [], []
    for d in sorted(dets):
        hit = None
        for i, (g, *_) in enumerate(gts):
            if i in used:
                continue
            if abs(d - g) <= tol:
                hit = i
                break
        if hit is not None:
            used.add(hit)
            matched_dets.append((d, gts[hit]))
        else:
            unmatched_dets.append(d)
    unmatched_gts = [g for i, g in enumerate(gts) if i not in used]
    return matched_dets, unmatched_dets, unmatched_gts


def report(label, dets, gts, tol):
    matched, fps, fns = score(dets, gts, tol)
    tp, fp, fn = len(matched), len(fps), len(fns)
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    return tp, fp, fn, rec, prec


def main():
    print(f"{'game':<10} {'tol':>4} {'type':<14} {'#det':>4} {'#gt':>3} "
          f"{'TP':>3} {'FP':>3} {'FN':>3} {'rec':>5} {'prec':>5}")
    print("-" * 78)
    aggregates = {tol: {"catch": [0,0,0], "parry": [0,0,0], "save": [0,0,0]}
                  for tol in (15, 30, 60, 90)}
    for game, cfg in GAMES.items():
        gts_all = load_gt_saves(cfg["gt"], cfg["off_1h"], cfg["off_2h"])
        gts_catch = [(t, ty, k) for t, ty, k in gts_all if ty == "Catches"]
        gts_parry = [(t, ty, k) for t, ty, k in gts_all if ty == "Parries"]
        det_catch = load_detections(cfg["events"], CATCH_TYPES)
        det_parry = load_detections(cfg["events"], PARRY_TYPES)
        det_save = load_detections(cfg["events"], SAVE_TYPES)
        for tol in (15, 30, 60, 90):
            for name, dets, gts in (("catch", det_catch, gts_catch),
                                    ("parry", det_parry, gts_parry),
                                    ("save",  det_save, gts_all)):
                tp, fp, fn, rec, prec = report(name, dets, gts, tol)
                print(f"{game:<10} {tol:>3}s {name:<14} {len(dets):>4} {len(gts):>3} "
                      f"{tp:>3} {fp:>3} {fn:>3} {rec:>5.2f} {prec:>5.2f}")
                aggregates[tol][name][0] += tp
                aggregates[tol][name][1] += fp
                aggregates[tol][name][2] += fn
            print()
    print("=" * 78)
    print("AGGREGATE across all 4 games:")
    for tol in (15, 30, 60, 90):
        for name in ("catch", "parry", "save"):
            tp, fp, fn = aggregates[tol][name]
            rec = tp / max(1, tp + fn)
            prec = tp / max(1, tp + fp)
            print(f"AGG        {tol:>3}s {name:<14} {'':>4} {'':>3} "
                  f"{tp:>3} {fp:>3} {fn:>3} {rec:>5.2f} {prec:>5.2f}")
        print()


if __name__ == "__main__":
    main()
