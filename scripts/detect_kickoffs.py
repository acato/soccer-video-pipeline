"""Kickoff-pattern goal detector — independent of VLM shot calls.

Continuous YOLO scan of the video. For each sampled frame:
  - run v9b ball detector → ball position (if any)
  - run uisikdag player/GK detector → field-player counts
  - derive: wide_shot (≥15 field persons), ball_at_center (within 0.15 of (0.5,0.5))

Then look for the transition pattern:
  celebration (NOT wide_shot for ≥15s)  →  kickoff (wide_shot AND ball_at_center for ≥10s)
                                          ─────────────────►
                                          marks a GOAL at start of celebration

Outputs events.jsonl-compatible JSONL with detected goals.

Usage:
    python scripts/detect_kickoffs.py \\
      --video "/path/to/game.mp4" \\
      --start 0 --end 6000 \\
      --interval 5 \\
      --out /tmp/kickoff_detection_<game>.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2  # type: ignore

# Models (paths assume Mac with /Volumes/transit mounted)
BALL_MODEL_DEFAULT = "/Volumes/transit/soccer-finetune/yolo_ball_v9/weights/v9b_best.pt"
PLAYER_MODEL_DEFAULT = "/Users/aless/Downloads/soccer-video-pipeline/infra/models/yolov8_soccer_uisikdag.pt"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

# Thresholds (tuned from game_20 validation; revisit if other games show different distributions)
WIDE_SHOT_MIN_PERSONS = 15           # ≥15 field persons → wide tactical shot
CENTER_X_LO, CENTER_X_HI = 0.35, 0.65
CENTER_Y_LO, CENTER_Y_HI = 0.30, 0.70  # y-band wider because broadcast cameras tilt down
BALL_MIN_CONF = 0.10                 # v9b ball at 1920px
FIELD_Y_LO, FIELD_Y_HI = 0.20, 0.85  # field band — filter sidelines/scoreboard

# Pattern timing (in sampled-frame steps after collapse to intervals)
MIN_KICKOFF_SUSTAIN_FRAMES = 2       # ≥2 consecutive samples of kickoff pattern
MIN_CELEBRATION_SUSTAIN_FRAMES = 3   # ≥3 consecutive samples of non-wide shot
MAX_CELEBRATION_GAP_FRAMES = 1       # allow 1 sample of "wide" in middle of celebration
MAX_CELEBRATION_DURATION_FRAMES = 30  # cap at 30 samples (~2.5 min at 5s interval)


def extract_frames_batch(video: str, timestamps: list[float], out_dir: Path) -> list[Path]:
    """Extract frames at given timestamps using ffmpeg. Returns paths in order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in timestamps:
        out = out_dir / f"frame_{t:.1f}.jpg"
        if not out.exists():
            subprocess.run(
                [FFMPEG, "-hide_banner", "-loglevel", "error",
                 "-ss", str(t), "-i", video, "-frames:v", "1", "-y", str(out)],
                check=True,
            )
        paths.append(out)
    return paths


def analyze_batch(ball_model, player_model, img_paths: list[Path]) -> list[dict]:
    """Run YOLO inference on a batch of frames, return per-frame analytics."""
    imgs = [cv2.imread(str(p)) for p in img_paths]
    valid = [(i, img) for i, img in enumerate(imgs) if img is not None]
    if not valid:
        return [{"ball": None, "p_left": 0, "p_right": 0, "total_field": 0} for _ in img_paths]

    valid_imgs = [img for _, img in valid]
    bres = ball_model(valid_imgs, imgsz=1920, conf=BALL_MIN_CONF, verbose=False)
    pres = player_model(valid_imgs, imgsz=1280, conf=0.15, verbose=False)

    results = [None] * len(img_paths)
    for (orig_i, _), br, pr in zip(valid, bres, pres):
        ball_pos = None
        if getattr(br, "boxes", None) is not None and len(br.boxes) > 0:
            bconfs = br.boxes.conf.cpu().numpy()
            bxywhn = br.boxes.xywhn.cpu().numpy()
            bi = bconfs.argmax()
            ball_pos = (float(bxywhn[bi][0]), float(bxywhn[bi][1]), float(bconfs[bi]))

        p_left = p_right = 0
        if getattr(pr, "boxes", None) is not None and len(pr.boxes) > 0:
            cls = pr.boxes.cls.cpu().numpy().astype(int)
            xywhn = pr.boxes.xywhn.cpu().numpy()
            for j, c in enumerate(cls):
                if c not in (1, 2):  # GK or player
                    continue
                cx, cy = xywhn[j][0], xywhn[j][1]
                if not (FIELD_Y_LO <= cy <= FIELD_Y_HI):
                    continue
                if cx < 0.5:
                    p_left += 1
                else:
                    p_right += 1

        results[orig_i] = {
            "ball": ball_pos,
            "p_left": p_left,
            "p_right": p_right,
            "total_field": p_left + p_right,
        }

    for i, r in enumerate(results):
        if r is None:
            results[i] = {"ball": None, "p_left": 0, "p_right": 0, "total_field": 0}
    return results


def derive_flags(per_frame: list[dict]) -> list[dict]:
    """Add wide_shot, ball_at_center, kickoff_scene flags per frame."""
    out = []
    for r in per_frame:
        wide = r["total_field"] >= WIDE_SHOT_MIN_PERSONS
        ball_center = False
        if r["ball"]:
            bx, by, _ = r["ball"]
            ball_center = CENTER_X_LO <= bx <= CENTER_X_HI and CENTER_Y_LO <= by <= CENTER_Y_HI
        out.append({**r, "wide_shot": wide, "ball_at_center": ball_center,
                    "kickoff_scene": wide and ball_center})
    return out


def find_kickoff_runs(flags: list[dict], timestamps: list[float]) -> list[tuple[int, int, float, float]]:
    """Find consecutive runs of kickoff_scene=True.

    Returns list of (start_idx, end_idx_inclusive, start_t, end_t)."""
    runs = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]["kickoff_scene"]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]["kickoff_scene"]:
            j += 1
        if (j - i + 1) >= MIN_KICKOFF_SUSTAIN_FRAMES:
            runs.append((i, j, timestamps[i], timestamps[j]))
        i = j + 1
    return runs


def find_celebration_before(flags: list[dict], kickoff_start_idx: int) -> tuple[int, int] | None:
    """Back-scan from kickoff_start_idx for a celebration run.

    Celebration = NOT wide_shot, sustained for ≥MIN_CELEBRATION_SUSTAIN_FRAMES,
    allowing up to MAX_CELEBRATION_GAP_FRAMES wide-shot samples in the middle,
    capped at MAX_CELEBRATION_DURATION_FRAMES.

    Returns (start_idx, end_idx) of celebration, or None if no run found
    within MAX_CELEBRATION_DURATION_FRAMES before the kickoff.
    """
    # End of celebration is the frame immediately before kickoff start
    end_idx = kickoff_start_idx - 1
    if end_idx < 0:
        return None

    # If end_idx is wide_shot, no celebration was happening here
    if flags[end_idx]["wide_shot"]:
        # Try one frame earlier — sometimes the broadcast briefly cuts to wide before kickoff setup
        if end_idx >= 1 and not flags[end_idx - 1]["wide_shot"]:
            end_idx -= 1
        else:
            return None

    # Scan backward from end_idx, allowing up to MAX_CELEBRATION_GAP_FRAMES "wide" samples
    start_idx = end_idx
    gap_budget = MAX_CELEBRATION_GAP_FRAMES
    while start_idx > 0 and (kickoff_start_idx - start_idx) < MAX_CELEBRATION_DURATION_FRAMES:
        prev = start_idx - 1
        if flags[prev]["wide_shot"]:
            if gap_budget <= 0:
                break
            gap_budget -= 1
            start_idx = prev
        else:
            start_idx = prev
            gap_budget = MAX_CELEBRATION_GAP_FRAMES  # reset on each non-wide frame

    duration = end_idx - start_idx + 1
    if duration < MIN_CELEBRATION_SUSTAIN_FRAMES:
        return None
    return (start_idx, end_idx)


def detect_goals(flags: list[dict], timestamps: list[float], interval: float) -> list[dict]:
    """Apply the celebration → kickoff transition rule to derive goals."""
    runs = find_kickoff_runs(flags, timestamps)
    goals = []
    for run_i, (k_start, k_end, k_t_start, k_t_end) in enumerate(runs):
        cel = find_celebration_before(flags, k_start)
        if cel is None:
            continue
        c_start, c_end = cel
        c_t_start = timestamps[c_start]
        c_t_end = timestamps[c_end]
        # Goal happens at the START of celebration (the close-up cut)
        goal_t = c_t_start
        goals.append({
            "event_type": "goal",
            "start_sec": goal_t,
            "end_sec": goal_t + 2.0,
            "confidence": 0.70,
            "reasoning": (
                f"kickoff-pattern: celebration cut at {c_t_start:.1f}s "
                f"(duration {c_t_end - c_t_start + interval:.1f}s) followed by "
                f"kickoff scene at {k_t_start:.1f}-{k_t_end:.1f}s "
                f"(wide-shot + ball-at-center)"
            ),
            "triage_labels": ["kickoff_pattern"],
            "_method": "kickoff_pattern_detector",
            "_celebration_start": c_t_start,
            "_celebration_end": c_t_end,
            "_kickoff_start": k_t_start,
            "_kickoff_end": k_t_end,
        })
    return goals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None,
                    help="end time in seconds; if omitted, runs to end of video")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ball-model", default=BALL_MODEL_DEFAULT)
    ap.add_argument("--player-model", default=PLAYER_MODEL_DEFAULT)
    ap.add_argument("--workdir", default="/tmp/kickoff_detection")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-frame-out", default=None,
                    help="optional path to dump per-frame analytics for inspection")
    args = ap.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"video not found: {args.video}")
    if not Path(args.ball_model).exists():
        sys.exit(f"ball model not found: {args.ball_model}")
    if not Path(args.player_model).exists():
        sys.exit(f"player model not found: {args.player_model}")

    end = args.end
    if end is None:
        # Probe duration via ffprobe
        r = subprocess.run(
            ["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", args.video],
            capture_output=True, text=True, check=True,
        )
        end = float(r.stdout.strip())

    timestamps = []
    t = args.start
    while t < end:
        timestamps.append(round(t, 2))
        t += args.interval
    print(f"sampling {len(timestamps)} frames at {args.interval}s intervals "
          f"from {args.start}-{end:.1f}s", file=sys.stderr)

    workdir = Path(args.workdir) / Path(args.video).stem
    print(f"loading YOLOs", file=sys.stderr)
    from ultralytics import YOLO
    ball_model = YOLO(args.ball_model)
    player_model = YOLO(args.player_model)

    print(f"extracting + analyzing in batches of {args.batch_size}", file=sys.stderr)
    per_frame_all = []
    t0 = time.time()
    for batch_start in range(0, len(timestamps), args.batch_size):
        batch_ts = timestamps[batch_start:batch_start + args.batch_size]
        paths = extract_frames_batch(args.video, batch_ts, workdir)
        results = analyze_batch(ball_model, player_model, paths)
        for ts, r in zip(batch_ts, results):
            r["t"] = ts
            per_frame_all.append(r)
        elapsed = time.time() - t0
        rate = len(per_frame_all) / max(1, elapsed)
        eta = (len(timestamps) - len(per_frame_all)) / max(0.01, rate)
        print(f"  {len(per_frame_all)}/{len(timestamps)} frames analyzed "
              f"({rate:.1f}/s, ETA {eta/60:.1f}min)", file=sys.stderr)

    flags = derive_flags(per_frame_all)
    for r, f in zip(per_frame_all, flags):
        r.update({k: f[k] for k in ("wide_shot", "ball_at_center", "kickoff_scene")})

    goals = detect_goals(flags, timestamps, args.interval)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for g in goals:
            fh.write(json.dumps(g) + "\n")
    print(f"detected {len(goals)} goals via kickoff pattern → {args.out}", file=sys.stderr)
    for g in goals:
        print(f"  goal @ {g['start_sec']:.1f}s (cel {g['_celebration_start']:.1f}-{g['_celebration_end']:.1f}s, "
              f"kickoff {g['_kickoff_start']:.1f}-{g['_kickoff_end']:.1f}s)", file=sys.stderr)

    if args.per_frame_out:
        Path(args.per_frame_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_frame_out, "w") as fh:
            for r in per_frame_all:
                fh.write(json.dumps(r) + "\n")
        print(f"per-frame analytics → {args.per_frame_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
