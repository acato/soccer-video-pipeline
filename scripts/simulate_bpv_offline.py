"""Apply v9b ball-presence verifier offline against an already-written
dual_pass_events.jsonl. Then recompute F1 against ground truth.

This bypasses the production BPV plumbing — useful when the in-pipeline
verifier no-ops (e.g. because goals don't carry detection_method tags
that BPV targets). It also avoids re-running the entire 5h pipeline.

Strategy: for each goal event, sample N frames evenly across
[start_sec, end_sec], run v9b at conf threshold, drop the goal if
0/N frames have a v9b ball detection.

Usage:
  python simulate_bpv_offline.py \\
      --events /tmp/soccer-pipeline/<job_id>/diagnostics/dual_pass_events.jsonl \\
      --video /Users/aless/soccer-working/<file>.mp4 \\
      --weights /Users/aless/Downloads/soccer-video-pipeline/infra/models/v9b_best.pt \\
      --gt /path/to/gt_h1.json --gt /path/to/gt_h2.json \\
      --video-offset 418.0 --half2-start 3916.0 --half2-game-offset 2700 \\
      --conf 0.10 --imgsz 1920 --n-frames 4 \\
      --out /tmp/bpv_sim_rush.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def extract_frame(video, ts_sec, out_path):
    try:
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error",
             "-ss", f"{ts_sec:.3f}", "-i", str(video),
             "-frames:v", "1", "-q:v", "2", "-y", str(out_path)],
            check=True, timeout=30,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as exc:
        print(f"  ffmpeg failed for ts={ts_sec}: {exc}", file=sys.stderr)
        return False


def probe_goal(model, video, t0, t1, *, conf, imgsz, n_frames):
    """Returns (any_ball_seen, any_yolo_read, n_dets_per_frame)."""
    if t1 <= t0:
        t1 = t0 + 4.0
    if n_frames <= 1:
        times = [(t0 + t1) / 2.0]
    else:
        step = (t1 - t0) / max(1, n_frames - 1)
        times = [t0 + i * step for i in range(n_frames)]
    any_ball = False
    any_read = False
    dets_per_frame = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, t in enumerate(times):
            frame = td / f"f{i}.jpg"
            if not extract_frame(video, t, frame):
                dets_per_frame.append(None)
                continue
            res = model.predict(str(frame), imgsz=imgsz, conf=conf, verbose=False)[0]
            any_read = True
            n = 0 if (res.boxes is None) else int(len(res.boxes))
            dets_per_frame.append(n)
            if n > 0:
                any_ball = True
    return any_ball, any_read, dets_per_frame


def to_gt_seconds(video_sec, *, video_offset, half2_video_start, half2_game_offset):
    """Convert video timestamp → game seconds (matching evaluate_detection logic)."""
    if video_sec >= half2_video_start:
        return (video_sec - half2_video_start) + half2_game_offset
    return video_sec - video_offset


def load_gt(paths):
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        for entry in d.get("data", []):
            game_sec = entry.get("event_time", 0) / 1000.0
            for ev in entry.get("events", []):
                name = ev.get("event_name", "")
                if "Goal" in name and "Conceded" not in name and "Kick" not in name:
                    # heuristic: shots&goals with outcome=goal? skip; use event_name "Goals"
                    pass
                if name == "Shots & Goals":
                    # could be shot or goal — only count if outcome=goal
                    out_field = ev.get("outcome", "")
                    if out_field.lower() == "goal":
                        out.append(("goal", game_sec))
                elif "Goals Conceded" in name:
                    out.append(("goal", game_sec))
    return [(t, s) for t, s in out]


def f1(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r, (2 * p * r / max(1e-9, p + r))


def match_goals_to_gt(detected_goals, gt_goals, *, tolerance_sec=45.0,
                      video_offset, half2_video_start, half2_game_offset):
    """Match detected (video-sec) goals to GT (game-sec) goals."""
    used_gt = set()
    matches = []  # list of (det, gt_idx or None)
    for d in detected_goals:
        d_game_sec = to_gt_seconds(d["start_sec"], video_offset=video_offset,
                                   half2_video_start=half2_video_start,
                                   half2_game_offset=half2_game_offset)
        best = None
        best_dist = None
        for gi, (_, gs) in enumerate(gt_goals):
            if gi in used_gt:
                continue
            dist = abs(d_game_sec - gs)
            if dist <= tolerance_sec and (best_dist is None or dist < best_dist):
                best, best_dist = gi, dist
        if best is not None:
            used_gt.add(best)
            matches.append((d, best))
        else:
            matches.append((d, None))
    return matches, used_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt", action="append", required=True)
    ap.add_argument("--video-offset", type=float, required=True)
    ap.add_argument("--half2-start", type=float, required=True)
    ap.add_argument("--half2-game-offset", type=float, required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=45.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"loading events from {args.events}")
    events = [json.loads(l) for l in Path(args.events).read_text().splitlines() if l.strip()]
    goals = [e for e in events if e["event_type"] == "goal"]
    print(f"  {len(events)} total events, {len(goals)} goal events")

    print(f"loading GT from {args.gt}")
    gt_goals = load_gt(args.gt)
    print(f"  {len(gt_goals)} GT goals")

    # Match BEFORE BPV
    matches_pre, used_pre = match_goals_to_gt(
        goals, gt_goals, tolerance_sec=args.tolerance,
        video_offset=args.video_offset, half2_video_start=args.half2_start,
        half2_game_offset=args.half2_game_offset)
    tp_pre = sum(1 for _, gi in matches_pre if gi is not None)
    fp_pre = sum(1 for _, gi in matches_pre if gi is None)
    fn_pre = len(gt_goals) - tp_pre
    p_pre, r_pre, f1_pre = f1(tp_pre, fp_pre, fn_pre)
    print(f"  pre-BPV : tp={tp_pre} fp={fp_pre} fn={fn_pre}  P={p_pre:.3f} R={r_pre:.3f} F1={f1_pre:.3f}")

    print(f"loading v9b from {args.weights}")
    from ultralytics import YOLO
    model = YOLO(args.weights)

    print(f"running BPV on {len(goals)} goals (conf={args.conf}, imgsz={args.imgsz}, n={args.n_frames})")
    kept = []
    dropped = []
    probe_log = []
    for i, g in enumerate(goals):
        any_ball, any_read, dets = probe_goal(
            model, args.video, g["start_sec"], g["end_sec"],
            conf=args.conf, imgsz=args.imgsz, n_frames=args.n_frames)
        rec = {"idx": i, "start_sec": g["start_sec"], "end_sec": g["end_sec"],
               "any_ball": any_ball, "any_read": any_read, "dets_per_frame": dets,
               "match_idx_pre": matches_pre[i][1]}
        probe_log.append(rec)
        # match for context
        gt_match = matches_pre[i][1]
        is_tp = gt_match is not None
        if any_ball or not any_read:  # keep (ball seen OR fail-open on no read)
            kept.append((g, gt_match))
            verdict = "KEEP"
        else:
            dropped.append((g, gt_match))
            verdict = "DROP"
        flag = "TP" if is_tp else "FP"
        print(f"  goal[{i}] ts={g['start_sec']:.0f}-{g['end_sec']:.0f} "
              f"({flag})  any_ball={any_ball}  dets={dets}  -> {verdict}")

    # Recompute post-BPV
    kept_goals = [g for g, _ in kept]
    matches_post, _ = match_goals_to_gt(
        kept_goals, gt_goals, tolerance_sec=args.tolerance,
        video_offset=args.video_offset, half2_video_start=args.half2_start,
        half2_game_offset=args.half2_game_offset)
    tp_post = sum(1 for _, gi in matches_post if gi is not None)
    fp_post = sum(1 for _, gi in matches_post if gi is None)
    fn_post = len(gt_goals) - tp_post
    p_post, r_post, f1_post = f1(tp_post, fp_post, fn_post)
    print()
    print("=== RESULTS ===")
    print(f"  pre-BPV : tp={tp_pre} fp={fp_pre} fn={fn_pre}  P={p_pre:.3f} R={r_pre:.3f} F1={f1_pre:.3f}")
    print(f"  post-BPV: tp={tp_post} fp={fp_post} fn={fn_post}  P={p_post:.3f} R={r_post:.3f} F1={f1_post:.3f}")
    print(f"  drops: {len(dropped)}  ({sum(1 for _,gi in dropped if gi is not None)} TP-drops, "
          f"{sum(1 for _,gi in dropped if gi is None)} FP-drops)")

    out = {
        "events_file": args.events,
        "video": args.video,
        "weights": args.weights,
        "conf": args.conf, "imgsz": args.imgsz, "n_frames": args.n_frames,
        "gt_goal_count": len(gt_goals),
        "detected_goal_count": len(goals),
        "pre_bpv": {"tp": tp_pre, "fp": fp_pre, "fn": fn_pre,
                    "precision": p_pre, "recall": r_pre, "f1": f1_pre},
        "post_bpv": {"tp": tp_post, "fp": fp_post, "fn": fn_post,
                     "precision": p_post, "recall": r_post, "f1": f1_post},
        "n_dropped": len(dropped),
        "n_tp_dropped": sum(1 for _, gi in dropped if gi is not None),
        "n_fp_dropped": sum(1 for _, gi in dropped if gi is None),
        "probes": probe_log,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
