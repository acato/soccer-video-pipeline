"""Score kickoff-pattern goal detections using auto-calibrated offsets.

For each game:
1. Run find_half_starts() on cached per-frame data → (t_1h, t_2h)
2. Compute offset_1H = t_1h and halftime = t_2h - t_1h - 1H_duration_seconds
3. Map GT "Goals Conceded" events (in game-clock ms) to video time
4. Score detections vs GT at tol=90s

Compares the calibrated baseline to the previously-guessed offsets.
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
        "dets":   "/tmp/kickoff_game_22.jsonl",
        "gt_1h":  "/Users/aless/soccer-runs/gt/game22/2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_1st Half.json",
        "gt_2h":  "/Users/aless/soccer-runs/gt/game22/2026-04-26_Seattle Reign 2011 GA (U15) vs Spokane Shadow U15 (W)_2nd Half.json",
        "h1_dur": 2700,
        "old_offset_1h": 460,
    },
    "game_21": {
        "frames": "/tmp/kickoff_game_21_frames.jsonl",
        "dets":   "/tmp/kickoff_game_21.jsonl",
        "gt_1h":  "/Users/aless/soccer-runs/gt/game21/2026-04-25_Seattle Reign 2011 GA (U15) vs Washington East Surf SC U15 (W)_1st Half.json",
        "gt_2h":  "/Users/aless/soccer-runs/gt/game21/2026-04-25_Seattle Reign 2011 GA (U15) vs Washington East Surf SC U15 (W)_2nd Half.json",
        "h1_dur": 2700,
        "old_offset_1h": 24,
    },
    "rush": {
        "frames": "/tmp/kickoff_rush_frames.jsonl",
        "dets":   "/tmp/kickoff_rush.jsonl",
        "gt_1h":  "/Users/aless/soccer-runs/gt/08 GA (U19) vs Washington Rush U19 (W)_1st Half.json",
        "gt_2h":  "/Users/aless/soccer-runs/gt/08 GA (U19) vs Washington Rush U19 (W)_2nd Half.json",
        "h1_dur": 2700,
        "old_offset_1h": 418,
    },
    "game_20_1H_scan": {
        "frames": "/tmp/kickoff_game20_1H_frames.jsonl",
        "dets":   "/tmp/kickoff_game20_1H.jsonl",
        "gt_1h":  "/Users/aless/soccer-runs/gt/game20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_1st Half.json",
        "gt_2h":  "/Users/aless/soccer-runs/gt/game20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_2nd Half.json",
        "h1_dur": 2400,
        "old_offset_1h": 124,
    },
}

TOL = 90.0


def load_gt_goal_times(path: str) -> list[float]:
    """Game-clock seconds for 'Goals Conceded' events in this half file."""
    data = json.loads(Path(path).read_text())["data"]
    return sorted(
        e["event_time"] / 1000.0
        for e in data
        for ev in e.get("events", [])
        if ev.get("event_name") == "Goals Conceded"
    )


def load_dets(path: str) -> list[float]:
    return sorted(
        json.loads(l)["start_sec"]
        for l in Path(path).read_text().splitlines()
        if l.strip()
    )


def calibrate(frames_path: str):
    rows = [json.loads(l) for l in Path(frames_path).read_text().splitlines() if l.strip()]
    raw = [{
        "ball": r.get("ball"),
        "p_left": r.get("p_left", 0),
        "p_right": r.get("p_right", 0),
        "total_field": r.get("total_field", 0),
        "in_circle": r.get("in_circle", 0),
        "t": r["t"],
    } for r in rows]
    flags = derive_flags(raw)
    ts = [r["t"] for r in raw]
    return find_half_starts(flags, ts, 5.0)


def score(dets: list[float], gt_video: list[float], tol: float = TOL):
    used = set()
    tp = 0
    for d in dets:
        best_i, best_dt = None, float("inf")
        for i, g in enumerate(gt_video):
            if i in used:
                continue
            dt = abs(d - g)
            if dt <= tol and dt < best_dt:
                best_i, best_dt = i, dt
        if best_i is not None:
            used.add(best_i)
            tp += 1
    return tp, len(dets) - tp, len(gt_video) - tp


def main():
    print(f"{'game':<18} {'cal_1H':>7} {'cal_2H':>7} {'cal_hf':>7}  "
          f"{'OLD TP/FP/FN':<14} {'CAL TP/FP/FN':<14}")
    cal_totals = [0, 0, 0]
    old_totals = [0, 0, 0]
    for game, cfg in GAMES.items():
        if not Path(cfg["frames"]).exists():
            continue
        dets = load_dets(cfg["dets"])
        gt_1h_game = load_gt_goal_times(cfg["gt_1h"])
        gt_2h_game = load_gt_goal_times(cfg["gt_2h"])

        # Calibrated offsets — add KICKOFF_LAG to density-anchored transitions
        # to land on the actual kickoff (measured ~60-70s after camera commit)
        t1_density, t2_density = calibrate(cfg["frames"])
        if t1_density is None:
            cal = (0, len(dets), len(gt_1h_game) + len(gt_2h_game))
        else:
            off_1h = t1_density + KICKOFF_LAG_AFTER_DENSITY_SECONDS
            if t2_density:
                off_2h_anchor = t2_density + KICKOFF_LAG_AFTER_DENSITY_SECONDS
                hf = off_2h_anchor - off_1h - cfg["h1_dur"]
            else:
                hf = 200.0  # fallback
            # 2H event_time is cumulative game-clock. video = event_time +
            # offset_1H + halftime (since 2H event_time already starts at
            # period_start_time, the +halftime accounts for the real-time gap).
            gt_video = ([g + off_1h for g in gt_1h_game]
                        + [g + off_1h + hf for g in gt_2h_game])
            cal = score(dets, gt_video)

        # Old (hand-picked) offsets
        old_off_1h = cfg["old_offset_1h"]
        gt_old = [g + old_off_1h for g in gt_1h_game] + \
                 [g + old_off_1h + 200 for g in gt_2h_game]  # assumed 200s halftime
        old = score(dets, gt_old)

        cal_totals = [cal_totals[i] + cal[i] for i in range(3)]
        old_totals = [old_totals[i] + old[i] for i in range(3)]

        t1_str = f"{t1_density:.0f}" if t1_density is not None else "-"
        t2_str = f"{t2_density:.0f}" if t2_density is not None else "-"
        hf_str = (f"{(t2_density - t1_density - cfg['h1_dur']):.0f}"
                  if (t1_density is not None and t2_density is not None) else "-")
        print(f"{game:<18} {t1_str:>7} {t2_str:>7} {hf_str:>7}  "
              f"{old[0]}/{old[1]}/{old[2]:<10}  {cal[0]}/{cal[1]}/{cal[2]}")

    print(f"\n{'TOTAL':<18} {'':>7} {'':>7} {'':>7}  "
          f"{old_totals[0]}/{old_totals[1]}/{old_totals[2]:<10}  "
          f"{cal_totals[0]}/{cal_totals[1]}/{cal_totals[2]}")
    o_p = old_totals[0] / max(1, old_totals[0] + old_totals[1])
    o_r = old_totals[0] / max(1, old_totals[0] + old_totals[2])
    c_p = cal_totals[0] / max(1, cal_totals[0] + cal_totals[1])
    c_r = cal_totals[0] / max(1, cal_totals[0] + cal_totals[2])
    print(f"OLD: prec={o_p:.2f} recall={o_r:.2f}")
    print(f"CAL: prec={c_p:.2f} recall={c_r:.2f}")


if __name__ == "__main__":
    main()
