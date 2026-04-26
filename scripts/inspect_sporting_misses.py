#!/usr/bin/env python3
"""Cross-reference sporting_ac GT events against single-pass VLM windows.

Inputs (expected on the Mac running the pipeline):
  - GT JSON files under /Volumes/transit/Games/1/
  - VLM diagnostics at /tmp/soccer-pipeline/<job_id>/diagnostics/single_pass_windows.jsonl
  - Celery log with VLM raw text at /tmp/celery.log
Outputs (stdout):
  - Count of GT events that landed in windows VLM marked empty
  - Breakdown by type
  - Sample miss entries with the VLM's raw response for that window
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

GT_TO_PIPELINE = {
    "Saves/Catches": "catch",
    "Saves/Parries": "shot_stop_diving",
    "Set Pieces/Corners": "corner_kick",
    "Set Pieces/Goal Kicks": "goal_kick",
    "Set Pieces/Throw-Ins": "throw_in",
    "Set Pieces/Freekicks": "free_kick_shot",
    "Set Pieces/Penalty Kicks": "penalty",
    "Goals Conceded": "goal",
    "Shots & Goals": "shot_on_target",
}


def load_gt(paths, vo, h2_start, h2_offset):
    events = []
    for half_idx, p in enumerate(paths):
        with open(p) as f:
            data = json.load(f)
        for entry in data["data"]:
            ms = entry["event_time"]
            team = entry.get("team_name", "")
            for ev in entry.get("events", []):
                name = ev["event_name"]
                prop = ev.get("property", {})
                typ = prop.get("Type", "")
                key = name + ("/" + typ if typ else "")
                ptype = GT_TO_PIPELINE.get(key)
                if ptype is None:
                    continue
                game_sec = ms / 1000.0
                if half_idx == 0:
                    video_sec = game_sec + vo
                else:
                    video_sec = (game_sec - h2_offset) + h2_start
                events.append({
                    "type": ptype,
                    "game_sec": game_sec,
                    "video_sec": video_sec,
                    "team": team,
                    "gt_name": key,
                    "half": half_idx,
                })
    return events


def load_windows(path):
    windows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            windows.append(json.loads(line))
    return windows


def find_window_for_time(windows, t):
    """Return all windows whose [start,end] contains t."""
    return [w for w in windows if w["start_sec"] <= t <= w["end_sec"]]


VLM_LOG_RE = re.compile(
    r"single_pass\.window\s+elapsed=([\d.]+)s\s+end=([\d.]+)\s+events=(\d+)\s+idx=(\d+)\s+start=([\d.]+)\s+text='(.+?)'\s*$",
    re.DOTALL,
)


def build_vlm_text_index(celery_log, job_id):
    """Walk celery.log, keep text responses for windows belonging to job_id.

    The worker processes one job at a time serially per worker; we delimit by
    'single_pass.start job_id=<id>' ... next 'single_pass.scan_complete'.
    """
    idx_to_text = {}
    capturing = False
    with open(celery_log, errors="replace") as f:
        for line in f:
            if "single_pass.start" in line and job_id in line:
                capturing = True
                continue
            if capturing and "single_pass.scan_complete" in line and job_id in line:
                capturing = False
                continue
            if not capturing:
                continue
            if "single_pass.window" not in line:
                continue
            m = VLM_LOG_RE.search(line)
            if not m:
                continue
            _, _, _, win_idx, _, text = m.groups()
            idx_to_text[int(win_idx)] = text
    return idx_to_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--gt-1h", required=True)
    ap.add_argument("--gt-2h", required=True)
    ap.add_argument("--video-offset", type=float, required=True)
    ap.add_argument("--half2-start", type=float, required=True)
    ap.add_argument("--half2-game-offset", type=float, required=True)
    ap.add_argument("--celery-log", default="/tmp/celery.log")
    ap.add_argument("--sample-n", type=int, default=3)
    ap.add_argument("--types", default="throw_in,corner_kick,shot_on_target,goal_kick",
                    help="Comma-separated list of event types to sample misses from")
    args = ap.parse_args()

    diag = f"/tmp/soccer-pipeline/{args.job_id}/diagnostics/single_pass_windows.jsonl"

    gt = load_gt([args.gt_1h, args.gt_2h], args.video_offset,
                 args.half2_start, args.half2_game_offset)
    print(f"GT events: {len(gt)}")
    print(f"  by type: {dict(Counter(e['type'] for e in gt))}")

    windows = load_windows(diag)
    print(f"Windows: {len(windows)}")

    idx_to_text = build_vlm_text_index(args.celery_log, args.job_id)
    print(f"VLM text records for this job: {len(idx_to_text)}")

    # For each GT event find the overlapping windows and whether any flagged anything
    coarse_hit = defaultdict(int)
    coarse_miss = defaultdict(int)
    coarse_type_match = defaultdict(int)  # VLM flagged the specific type
    coarse_type_wrong = defaultdict(int)  # VLM flagged something else
    misses_by_type = defaultdict(list)
    for e in gt:
        overlapping = find_window_for_time(windows, e["video_sec"])
        if not overlapping:
            # GT falls outside scan (unlikely)
            coarse_miss[e["type"]] += 1
            misses_by_type[e["type"]].append({"gt": e, "windows": []})
            continue
        any_event = any(w["n_events"] > 0 for w in overlapping)
        type_match = any(e["type"] in (w.get("event_types") or []) for w in overlapping)
        any_other = any(w["n_events"] > 0 and e["type"] not in (w.get("event_types") or [])
                        for w in overlapping)
        if type_match:
            coarse_hit[e["type"]] += 1
            coarse_type_match[e["type"]] += 1
        elif any_event:
            coarse_type_wrong[e["type"]] += 1
            coarse_hit[e["type"]] += 1
        else:
            coarse_miss[e["type"]] += 1
            misses_by_type[e["type"]].append({"gt": e, "windows": overlapping})

    print("\nCoarse-pass hit/miss (did VLM flag ANYTHING in any overlapping window?):")
    print(f"  {'type':<20}  {'gt':>5}  {'hit':>5}  {'miss':>5}  {'type_match':>10}  {'type_wrong':>10}")
    for t in sorted(set(list(coarse_hit.keys()) + list(coarse_miss.keys()))):
        n_gt = coarse_hit[t] + coarse_miss[t]
        print(f"  {t:<20}  {n_gt:>5}  {coarse_hit[t]:>5}  {coarse_miss[t]:>5}  "
              f"{coarse_type_match[t]:>10}  {coarse_type_wrong[t]:>10}")

    print("\n=== SAMPLE MISSES (VLM said no-event in all windows containing GT) ===")
    target_types = [t.strip() for t in args.types.split(",") if t.strip()]
    for t in target_types:
        samples = misses_by_type.get(t, [])[: args.sample_n]
        print(f"\n--- {t} ({len(misses_by_type.get(t, []))} total misses, showing {len(samples)}) ---")
        for s in samples:
            e = s["gt"]
            print(f"  GT @ video_sec={e['video_sec']:.1f}  team={e['team'][:25]}  half={e['half']}")
            for w in s["windows"]:
                idx = w["window_idx"]
                text = idx_to_text.get(idx, "<not found in log>")
                print(f"    window idx={idx} [{w['start_sec']:.1f}-{w['end_sec']:.1f}]  n_events={w['n_events']}")
                print(f"      VLM: {text[:400]}")


if __name__ == "__main__":
    main()
