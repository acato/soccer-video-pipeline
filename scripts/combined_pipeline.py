"""Combined goal-detection pipeline scorer.

Sources of candidates:
  1. kickoff_pattern detector (detect_kickoffs.py) — wide+ball_at_center triggered
  2. formation generator (generate_formation_candidates.py) — wide+balanced LR

Each candidate is filtered by:
  - VLM v3 verdict (must be GOAL)
  - Game-time bounds: candidate must be within
    [offset_1H - 60, offset_1H + 2 * H_DUR + halftime + 120]

Outputs a deduplicated, scored detection list per game and an aggregate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_kickoffs import (  # type: ignore  # noqa: E402
    derive_flags, find_half_starts, KICKOFF_LAG_AFTER_DENSITY_SECONDS,
)

GAMES = {
    "game_22": {
        "frames": "/tmp/kickoff_game_22_frames.jsonl",
        "pattern_vlm": "/tmp/kickoff_game_22_vlm_v3.jsonl",
        "formation_vlm": "/tmp/kickoff_game_22_formation_vlm.jsonl",
        "formation_base": "/tmp/kickoff_game_22_formation_v2_base.jsonl",
        "gt_1h": "/Users/aless/soccer-runs/gt/game22/2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_1st Half.json",
        "gt_2h": "/Users/aless/soccer-runs/gt/game22/2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_2nd Half.json",
        "h1_dur": 2700,
    },
    "game_21": {
        "frames": "/tmp/kickoff_game_21_frames.jsonl",
        "pattern_vlm": "/tmp/kickoff_game_21_vlm_v3.jsonl",
        "formation_vlm": "/tmp/kickoff_game_21_formation_vlm.jsonl",
        "formation_base": "/tmp/kickoff_game_21_formation_v2_base.jsonl",
        "gt_1h": "/Users/aless/soccer-runs/gt/game21/2026-04-25_Seattle Reign 2011 GA (U15) vs Washington East Surf SC U15 (W)_1st Half.json",
        "gt_2h": "/Users/aless/soccer-runs/gt/game21/2026-04-25_Seattle Reign 2011 GA (U15) vs Washington East Surf SC U15 (W)_2nd Half.json",
        "h1_dur": 2700,
    },
    "rush": {
        "frames": "/tmp/kickoff_rush_frames.jsonl",
        "pattern_vlm": "/tmp/kickoff_rush_vlm_v3.jsonl",
        "formation_vlm": "/tmp/kickoff_rush_formation_vlm.jsonl",
        "formation_base": "/tmp/kickoff_rush_formation_v2_base.jsonl",
        "gt_1h": "/Users/aless/soccer-runs/gt/08 GA (U19) vs Washington Rush U19 (W)_1st Half.json",
        "gt_2h": "/Users/aless/soccer-runs/gt/08 GA (U19) vs Washington Rush U19 (W)_2nd Half.json",
        "h1_dur": 2700,
    },
}

TOL = 90.0
DEDUP_WINDOW = 30.0  # merge confirmed detections within 30s of each other


def load_gt(path):
    data = json.loads(Path(path).read_text())["data"]
    return sorted(
        e["event_time"] / 1000.0
        for e in data
        for ev in e.get("events", [])
        if ev.get("event_name") == "Goals Conceded"
    )


def calibrate(frames_path):
    rows = [json.loads(l) for l in Path(frames_path).read_text().splitlines() if l.strip()]
    raw = [{"ball": r.get("ball"), "p_left": r.get("p_left", 0),
            "p_right": r.get("p_right", 0), "total_field": r.get("total_field", 0),
            "in_circle": r.get("in_circle", 0), "t": r["t"]} for r in rows]
    flags = derive_flags(raw)
    ts = [r["t"] for r in raw]
    return find_half_starts(flags, ts, 5.0)


def gt_video_times(gt_1h_game, gt_2h_game, off_1h, halftime):
    return ([g + off_1h for g in gt_1h_game]
            + [g + off_1h + halftime for g in gt_2h_game])


def load_confirmed(path):
    return [
        r for r in (json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())
        if r.get("_vlm_verdict") == "GOAL"
    ]


def dedup(detections):
    """Merge detections within DEDUP_WINDOW seconds. Keeps the earlier one."""
    dets = sorted(detections, key=lambda d: d["start_sec"])
    out = []
    for d in dets:
        if out and (d["start_sec"] - out[-1]["start_sec"]) <= DEDUP_WINDOW:
            continue
        out.append(d)
    return out


def filter_game_bounds(dets, off_1h, halftime, h1_dur):
    if off_1h is None:
        return dets
    game_start = off_1h - 60
    game_end = off_1h + h1_dur + halftime + h1_dur + 120
    return [d for d in dets if game_start <= d["start_sec"] <= game_end]


def score(dets, gt_video, tol=TOL):
    used = set()
    tp = 0
    pairs = []
    for d in sorted(dets, key=lambda x: x["start_sec"]):
        best_i, best_dt = None, float("inf")
        for i, g in enumerate(gt_video):
            if i in used:
                continue
            dt = abs(d["start_sec"] - g)
            if dt <= tol and dt < best_dt:
                best_i, best_dt = i, dt
        if best_i is not None:
            used.add(best_i)
            tp += 1
            pairs.append((d["start_sec"], gt_video[best_i]))
    return tp, len(dets) - tp, len(gt_video) - tp, pairs


def main():
    print(f"{'game':<10} {'src':<14} {'kept':>5} {'TP':>3} {'FP':>3} {'FN':>3}")
    totals = {"pattern_v11": [0, 0, 0], "formation_v11": [0, 0, 0],
              "formation_base": [0, 0, 0], "v11_combined": [0, 0, 0],
              "best_combined": [0, 0, 0]}
    for game, cfg in GAMES.items():
        gt_1h = load_gt(cfg["gt_1h"])
        gt_2h = load_gt(cfg["gt_2h"])
        t1_d, t2_d = calibrate(cfg["frames"])
        if t1_d is None:
            print(f"{game}: cannot calibrate, skipping")
            continue
        off_1h = t1_d + KICKOFF_LAG_AFTER_DENSITY_SECONDS
        if t2_d is not None:
            off_2h_anchor = t2_d + KICKOFF_LAG_AFTER_DENSITY_SECONDS
            halftime = off_2h_anchor - off_1h - cfg["h1_dur"]
        else:
            halftime = 200.0
        gt_vid = gt_video_times(gt_1h, gt_2h, off_1h, halftime)

        pat = load_confirmed(cfg["pattern_vlm"])
        form_v11 = load_confirmed(cfg["formation_vlm"])
        form_base = load_confirmed(cfg["formation_base"])
        pat = filter_game_bounds(pat, off_1h, halftime, cfg["h1_dur"])
        form_v11 = filter_game_bounds(form_v11, off_1h, halftime, cfg["h1_dur"])
        form_base = filter_game_bounds(form_base, off_1h, halftime, cfg["h1_dur"])
        v11_combined = dedup(pat + form_v11)
        # "best": pattern (v11) for the celebration/goal signals it's strong on,
        # base formation for the kickoff_restart signals v11 suppresses
        best_combined = dedup(pat + form_base)

        for label, dets in (("pattern_v11", pat),
                            ("formation_v11", form_v11),
                            ("formation_base", form_base),
                            ("v11_combined", v11_combined),
                            ("best_combined", best_combined)):
            tp, fp, fn, pairs = score(dets, gt_vid)
            print(f"{game:<10} {label:<14} {len(dets):>5} {tp:>3} {fp:>3} {fn:>3}")
            if label == "best_combined" and pairs:
                for d, g in pairs:
                    print(f"           TP: det {d:.0f} -> GT {g:.0f}")
            totals[label][0] += tp
            totals[label][1] += fp
            totals[label][2] += fn
        print()

    print("=" * 60)
    print(f"{'AGG':<10} {'src':<14} {'':>5} {'TP':>3} {'FP':>3} {'FN':>3} {'P':>5} {'R':>5}")
    for label in ("pattern_v11", "formation_v11", "formation_base",
                  "v11_combined", "best_combined"):
        tp, fp, fn = totals[label]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        print(f"{'AGG':<10} {label:<14} {'':>5} {tp:>3} {fp:>3} {fn:>3} {prec:>5.2f} {rec:>5.2f}")


if __name__ == "__main__":
    main()
