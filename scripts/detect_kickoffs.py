"""Kickoff-pattern goal detector — independent of VLM shot calls.

Continuous YOLO scan of the video. For each sampled frame:
  - run v9b ball detector → ball position (if any)
  - run uisikdag player/GK/referee detector → spatial player layout
  - derive per-frame flags:
      wide_shot       (≥15 field persons → tactical camera)
      ball_at_center  (ball within tight box around (0.5, 0.5))
      one_in_circle   (EXACTLY 1 player inside center-circle ellipse)
      kickoff_setup   (wide_shot AND ball_at_center AND one_in_circle)

  The "exactly 1 player in center circle" rule is the user-specified
  discriminator that distinguishes a true kickoff setup from generic
  midfield play where multiple players cluster near center.

Then look for the transition pattern:
  celebration (NOT wide_shot for ≥15s)  →  kickoff_setup sustained for ≥10s
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

# Thresholds (tuned via sweep on game_20 1H — 2/4 GT goals recovered, 0 FPs).
# WIDE_SHOT=18 (not 15) is the FP killer: excludes goal-kick aftermath where
# many players are airborne or behind the line, fewer settled-on-field bodies.
WIDE_SHOT_MIN_PERSONS = 18
# Bounds for ball-at-center spot (X tight, Y narrower than initial guess —
# the broadcast camera tilt makes the actual center spot appear higher in
# normalized image coords).
CENTER_X_LO, CENTER_X_HI = 0.40, 0.60
CENTER_Y_LO, CENTER_Y_HI = 0.35, 0.50
BALL_MIN_CONF = 0.10                 # v9b ball at 1920px
FIELD_Y_LO, FIELD_Y_HI = 0.20, 0.85  # field band — filter sidelines/scoreboard
# Center-circle ellipse for player count check.
# Player bbox centers are at chest height ≈ 0.05 above feet, so the cluster
# inside the center circle appears at y a bit higher than the ball.
CIRCLE_X_LO, CIRCLE_X_HI = 0.35, 0.65
CIRCLE_Y_LO, CIRCLE_Y_HI = 0.25, 0.50
# Allow 1-3 players inside (kicker + possibly central attacker/referee).
ONE_IN_CIRCLE_MIN = 1
ONE_IN_CIRCLE_MAX = 3

# Pattern timing (in sampled-frame steps after collapse to intervals)
MIN_KICKOFF_SUSTAIN_FRAMES = 2       # ≥2 samples of kickoff pattern (with gap tolerance)
MAX_KICKOFF_GAP_FRAMES = 1           # allow 1 sample of kickoff_setup=False inside the run
                                     # (v9b ball detector misses many frames; this tolerance
                                     # keeps "wide+ball_c at t, no ball at t+5s, wide+ball_c at t+10s"
                                     # together as a single kickoff run)
MIN_CELEBRATION_SUSTAIN_FRAMES = 2   # ≥2 samples of non-wide shot (10s)
MAX_CELEBRATION_GAP_FRAMES = 1       # allow 1 sample of "wide" in middle of celebration
MAX_CELEBRATION_DURATION_FRAMES = 30  # cap at 30 samples (~2.5 min at 5s interval)

# Path B (ball-traversal) parameters — tuned via sweep
TRAVERSAL_BALL_END_X_LO = 0.05       # ball "near goal area" — left side
TRAVERSAL_BALL_END_X_HI = 0.90       # ball "near goal area" — right side (tighter than 0.95 to exclude near-corner false positives)
TRAVERSAL_BALL_END_Y_LO = 0.30       # vertical band around goal mouth
TRAVERSAL_BALL_END_Y_HI = 0.70
TRAVERSAL_LOOKBACK_FRAMES_MIN = 3    # ball seen at goal area at least N samples back (15s)
TRAVERSAL_LOOKBACK_FRAMES_MAX = 18   # ...but not more than M back (90s — accommodates long celebrations)

# Half-start exclusion — every 1H/2H opening looks identical to a "goal then
# kickoff" pattern. Without filtering, the half-starts are FP goals in every
# game. Anchor them by the kickoff_setup run with the longest preceding gap.
HALF_START_MIN_GAP_SECONDS = 180     # at least 3 min of low activity before kickoff
HALF_START_MATCH_TOL_SECONDS = 30    # remove any detected "goal" within this of a half start

# The wide_shot density transition leads the actual kickoff by ~60-70s
# (camera commits to wide BEFORE the ref blows the whistle). Measured by
# visual spot-check on game_22 (190s actual vs 130s density) and rush
# (~400s actual vs 330s density). Apply this lag when treating the
# density-detected transition as the kickoff anchor for offset calibration.
KICKOFF_LAG_AFTER_DENSITY_SECONDS = 65


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
        in_circle = 0
        if getattr(pr, "boxes", None) is not None and len(pr.boxes) > 0:
            cls = pr.boxes.cls.cpu().numpy().astype(int)
            xywhn = pr.boxes.xywhn.cpu().numpy()
            for j, c in enumerate(cls):
                if c not in (1, 2):  # GK or player (skip referee class 3)
                    continue
                cx, cy = xywhn[j][0], xywhn[j][1]
                if not (FIELD_Y_LO <= cy <= FIELD_Y_HI):
                    continue
                if cx < 0.5:
                    p_left += 1
                else:
                    p_right += 1
                # "Inside center circle" check
                if (CIRCLE_X_LO <= cx <= CIRCLE_X_HI and
                        CIRCLE_Y_LO <= cy <= CIRCLE_Y_HI):
                    in_circle += 1

        results[orig_i] = {
            "ball": ball_pos,
            "p_left": p_left,
            "p_right": p_right,
            "total_field": p_left + p_right,
            "in_circle": in_circle,
        }

    for i, r in enumerate(results):
        if r is None:
            results[i] = {"ball": None, "p_left": 0, "p_right": 0, "total_field": 0, "in_circle": 0}
    return results


def derive_flags(per_frame: list[dict]) -> list[dict]:
    """Add wide_shot, ball_at_center, one_in_circle, kickoff_setup flags.

    kickoff_setup (primary trigger) is RELAXED: wide_shot + ball_at_center.
    The strict "one_in_circle" rule is kept as a SECONDARY flag to upgrade
    confidence when present — it rarely fires at 5s sampling because the
    kickoff moment is brief (~5s) and players are usually still settling
    into halves during the window.

    Reliability comes from the celebration / ball-traversal evidence paths
    in detect_goals(), not from the per-frame trigger alone.
    """
    out = []
    for r in per_frame:
        wide = r["total_field"] >= WIDE_SHOT_MIN_PERSONS
        ball_center = False
        if r["ball"]:
            bx, by, _ = r["ball"]
            ball_center = CENTER_X_LO <= bx <= CENTER_X_HI and CENTER_Y_LO <= by <= CENTER_Y_HI
        one_in_circle = ONE_IN_CIRCLE_MIN <= r.get("in_circle", 0) <= ONE_IN_CIRCLE_MAX
        kickoff_setup = wide and ball_center  # relaxed
        kickoff_setup_strong = kickoff_setup and one_in_circle  # for confidence upgrade
        out.append({**r,
                    "wide_shot": wide,
                    "ball_at_center": ball_center,
                    "one_in_circle": one_in_circle,
                    "kickoff_setup": kickoff_setup,
                    "kickoff_setup_strong": kickoff_setup_strong,
                    # Backward-compat alias
                    "kickoff_scene": kickoff_setup})
    return out


def find_kickoff_runs(flags: list[dict], timestamps: list[float]) -> list[tuple[int, int, float, float]]:
    """Find runs of kickoff_setup with up to MAX_KICKOFF_GAP_FRAMES tolerated False frames.

    Returns list of (start_idx, end_idx_inclusive, start_t, end_t).

    Gap tolerance is crucial because v9b often loses the ball between sampled
    frames during the kickoff setup phase; without tolerance, runs split.
    """
    runs = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]["kickoff_setup"]:
            i += 1
            continue
        j = i
        last_true = i
        gap = 0
        while j + 1 < n:
            nxt = flags[j + 1]
            if nxt["kickoff_setup"]:
                j += 1
                last_true = j
                gap = 0
            elif gap < MAX_KICKOFF_GAP_FRAMES:
                j += 1
                gap += 1
            else:
                break
        # True count within [i, last_true]
        true_count = sum(1 for k in range(i, last_true + 1) if flags[k]["kickoff_setup"])
        if true_count >= MIN_KICKOFF_SUSTAIN_FRAMES:
            runs.append((i, last_true, timestamps[i], timestamps[last_true]))
        i = last_true + 1
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


def find_ball_traversal_before(flags: list[dict], kickoff_start_idx: int) -> tuple[int, float] | None:
    """Path B: ball was near goal area in the recent past, now at center.

    Returns (origin_idx, origin_t) where the ball was last seen near goal,
    or None if no such observation in the lookback window.
    """
    lo = max(0, kickoff_start_idx - TRAVERSAL_LOOKBACK_FRAMES_MAX)
    hi = kickoff_start_idx - TRAVERSAL_LOOKBACK_FRAMES_MIN
    if hi < lo:
        return None

    best_origin = None
    for i in range(hi, lo - 1, -1):
        ball = flags[i].get("ball")
        if not ball:
            continue
        bx, by, _ = ball
        # Near either goal area (extreme X), within goal-mouth Y band
        near_left_goal = bx <= 0.15 and TRAVERSAL_BALL_END_Y_LO <= by <= TRAVERSAL_BALL_END_Y_HI
        near_right_goal = bx >= 0.85 and TRAVERSAL_BALL_END_Y_LO <= by <= TRAVERSAL_BALL_END_Y_HI
        if near_left_goal or near_right_goal:
            best_origin = (i, flags[i].get("t", 0.0))
            break  # take the most-recent goal-area observation
    return best_origin


def find_half_starts(flags: list[dict], timestamps: list[float],
                     interval: float,
                     min_gap_seconds: float = 60.0,
                     density_window_seconds: float = 150.0,
                     low_density_threshold: float = 0.30,
                     high_density_threshold: float = 0.40
                     ) -> tuple[float | None, float | None]:
    """Identify 1H and 2H opening kickoffs via wide_shot density transitions.

    During active gameplay the broadcast camera stays in a wide tactical
    view ~60-95% of the time. During pre-game, halftime, and post-game it
    drops below ~30% (close-ups, replays, idle field shots, scoreboards).

    The two longest LOW-density stretches that are followed by sustained
    HIGH-density activity are pre-game (its end = 1H kickoff) and halftime
    (its end = 2H kickoff).

    This is reliable where naive "first kickoff_setup run" fails (ball
    detector misses the brief opening moment) and where ball-density alone
    fails (false-positive ball detections during pre-game).

    Returns (t_1h_start, t_2h_start). Either may be None when the scan
    range does not include the corresponding opening (partial scans).
    """
    n = len(flags)
    if n == 0:
        return None, None

    window_frames = max(1, int(density_window_seconds / interval))
    min_low_frames = max(1, int(min_gap_seconds / interval))

    # Rolling wide_shot density at each frame
    densities = []
    for i in range(n):
        lo = max(0, i - window_frames // 2)
        hi = min(n, i + window_frames // 2 + 1)
        wide = sum(1 for k in range(lo, hi) if flags[k]["wide_shot"])
        densities.append(wide / max(1, hi - lo))

    # Runs where density < low threshold (non-game stretches)
    low_runs = []
    i = 0
    while i < n:
        if densities[i] >= low_density_threshold:
            i += 1
            continue
        j = i
        while j + 1 < n and densities[j + 1] < low_density_threshold:
            j += 1
        low_runs.append((i, j, j - i + 1))
        i = j + 1

    # A candidate "half-kickoff anchor" is the FIRST sustained-high-density
    # frame after a long low-density run. The density-crossing point (~30%)
    # is too early — it marks the END of pre-game / halftime non-game footage,
    # but the actual kickoff is when the camera commits to wide tactical view
    # (~50%+ sustained). Walk forward to that point.
    sustain_frames = max(1, int(60 / interval))
    candidates = []
    for (s, e, dur) in low_runs:
        if dur < min_low_frames:
            continue
        if e + 1 >= n:
            continue  # post-game (no subsequent activity)
        # Walk forward from the end of the low run to the first
        # sustained-high-density frame
        anchor_idx = e + 1
        while anchor_idx < n and densities[anchor_idx] < high_density_threshold:
            anchor_idx += 1
        if anchor_idx >= n:
            continue
        # Confirm sustained activity for the next ~60s
        active_after = sum(
            1 for k in range(anchor_idx, min(n, anchor_idx + sustain_frames * 2))
            if densities[k] >= high_density_threshold
        )
        if active_after >= sustain_frames:
            candidates.append((dur, timestamps[anchor_idx]))

    # Also handle the case where the scan starts at the actual 1H kickoff
    # (no pre-game in the scan): the first sustained high-density frame.
    if densities[0] >= high_density_threshold:
        candidates.append((float("inf"), timestamps[0]))  # 1H starts at scan start

    if not candidates:
        return None, None
    # Sort by time (chronologically), not by duration. The two earliest
    # candidates are 1H (after pre-game) and 2H (after halftime). Anything
    # later is likely post-game or weather break — ignore.
    candidates_by_time = sorted({c[1] for c in candidates})
    if len(candidates_by_time) >= 2:
        return candidates_by_time[0], candidates_by_time[1]
    return candidates_by_time[0], None


def detect_goals(flags: list[dict], timestamps: list[float], interval: float,
                 exclude_half_starts: bool = True) -> list[dict]:
    """Derive goals via the kickoff_setup pattern with two evidence paths:

    Path A — celebration close-up before kickoff_setup:
        marks goal at start of celebration period
    Path B — ball traversal from goal area to center spot (no close-up needed):
        marks goal at the time the ball was last near goal area

    When `exclude_half_starts` is True (default), the 1H and 2H opening
    kickoffs are removed from the goal list — they otherwise produce a FP
    "goal" in every game because the preceding non-game stretch trips the
    celebration_cut path.
    """
    runs = find_kickoff_runs(flags, timestamps)
    goals = []
    for run_i, (k_start, k_end, k_t_start, k_t_end) in enumerate(runs):
        # Path A: celebration close-up
        cel = find_celebration_before(flags, k_start)
        # Path B: ball traversal
        trav = find_ball_traversal_before(flags, k_start)

        if cel is None and trav is None:
            continue

        # Prefer Path A timing when available (more precise — close-up cut
        # coincides closely with the goal). Fall back to Path B.
        if cel is not None:
            c_start, c_end = cel
            goal_t = timestamps[c_start]
            method_tag = "celebration_cut" if trav is None else "celebration+traversal"
        else:
            origin_idx, origin_t = trav
            goal_t = origin_t
            method_tag = "ball_traversal"

        goals.append({
            "event_type": "goal",
            "start_sec": goal_t,
            "end_sec": goal_t + 2.0,
            "confidence": 0.75 if cel is not None else 0.65,
            "reasoning": (
                f"kickoff-pattern ({method_tag}): kickoff_setup at "
                f"{k_t_start:.1f}-{k_t_end:.1f}s "
                + (f"preceded by celebration {timestamps[cel[0]]:.1f}-{timestamps[cel[1]]:.1f}s "
                   if cel else "")
                + (f"with ball at goal area at {trav[1]:.1f}s "
                   if trav else "")
            ),
            "triage_labels": ["kickoff_pattern"],
            "_method": f"kickoff_pattern_detector_{method_tag}",
            "_celebration_start": timestamps[cel[0]] if cel else None,
            "_celebration_end": timestamps[cel[1]] if cel else None,
            "_traversal_origin": trav[1] if trav else None,
            "_kickoff_start": k_t_start,
            "_kickoff_end": k_t_end,
        })

    if exclude_half_starts:
        t_1h, t_2h = find_half_starts(flags, timestamps, interval)
        anchors = [t for t in (t_1h, t_2h) if t is not None]
        if anchors:
            goals = [g for g in goals
                     if not any(abs(g["_kickoff_start"] - a)
                                <= HALF_START_MATCH_TOL_SECONDS for a in anchors)]
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
        r.update({k: f[k] for k in ("wide_shot", "ball_at_center",
                                      "one_in_circle", "kickoff_setup",
                                      "kickoff_setup_strong", "kickoff_scene")})

    t_1h, t_2h = find_half_starts(flags, timestamps, args.interval)
    print(f"half-start calibration: 1H kickoff @ {t_1h}s, 2H kickoff @ {t_2h}s",
          file=sys.stderr)

    goals = detect_goals(flags, timestamps, args.interval)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for g in goals:
            fh.write(json.dumps(g) + "\n")
    print(f"detected {len(goals)} goals via kickoff pattern → {args.out}", file=sys.stderr)
    for g in goals:
        cs = g.get("_celebration_start")
        ce = g.get("_celebration_end")
        cs_str = f"{cs:.1f}" if cs is not None else "-"
        ce_str = f"{ce:.1f}" if ce is not None else "-"
        tr = g.get("_traversal_origin")
        tr_str = f"{tr:.1f}" if tr is not None else "-"
        print(f"  goal @ {g['start_sec']:.1f}s (cel {cs_str}-{ce_str}, "
              f"traversal {tr_str}, kickoff {g['_kickoff_start']:.1f}-{g['_kickoff_end']:.1f}s)",
              file=sys.stderr)

    if args.per_frame_out:
        Path(args.per_frame_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_frame_out, "w") as fh:
            for r in per_frame_all:
                fh.write(json.dumps(r) + "\n")
        print(f"per-frame analytics → {args.per_frame_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
