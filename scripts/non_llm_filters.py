"""Non-LLM YOLO-based filters to pare down FPs while preserving recall.

Distinguishing signals between real-goal kickoffs and midfield-formation FPs:

  (A) Preceding low-activity gap: real goals have a 5-30s "pause" (ref retrieves
      ball, celebration close-up) before the kickoff. FPs from continuous play
      don't. Use wide_shot dip in -30..-10s.

  (B) Ball stability at center: at a real kickoff the ball sits at the center
      spot for 5-15s. Check ball stability (low motion) around the candidate.

  (C) In-circle count sanity: at kickoff EXACTLY 1-3 players in center circle
      (kicker + ref + ?). The base FP8 generator relaxed to 0-15; tightening
      to ≤4 specifically at the candidate moment should cut FPs.

  (D) Two-team formation: at kickoff, BOTH halves have substantial players
      (≥5 each). Pure midfield play often has one side heavily attacking.

  (E) Sustained wide_shot AFTER candidate: real kickoff transitions to wide
      tactical play sustained for ≥30s. FPs from brief balanced moments don't.
"""
import json
from pathlib import Path

TOL = 90.0


def aggregate_relaxed(labels):
    labs = sorted(labels, key=lambda x: x[0])
    if any(l == "celebration" for _, l in labs):
        return "GOAL"
    for i, (_, l) in enumerate(labs):
        if l == "goal":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL"
    for i, (_, l) in enumerate(labs):
        if l == "kickoff_restart":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL"
            for _, before in labs[:i]:
                if before in ("goal", "celebration", "set_piece"):
                    return "GOAL"
    return "NO"


def load_frames(paths):
    rows = []
    for p in paths:
        if Path(p).exists():
            rows.extend(json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())
    rows.sort(key=lambda r: r["t"])
    return rows


def load_all_goals(formation_paths, pattern_path=None):
    out = []
    for fp in formation_paths:
        if not Path(fp).exists():
            continue
        for line in Path(fp).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if aggregate_relaxed(r.get("_vlm_labels", [])) == "GOAL":
                out.append(r)
    if pattern_path and Path(pattern_path).exists():
        for line in Path(pattern_path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_vlm_verdict") == "GOAL":
                out.append(r)
    return out


def gt_score(dets, gts, tol=TOL):
    tp = 0; used = set()
    for g in gts:
        for i, d in enumerate(dets):
            if i in used:
                continue
            if abs(d["start_sec"] - g) <= tol:
                tp += 1; used.add(i); break
    return tp, len(dets) - len(used)


def frames_in_range(frames, t_lo, t_hi):
    return [f for f in frames if t_lo <= f["t"] <= t_hi]


# --- Filter implementations ---

def filt_preceding_gap(det, frames, lookback_lo=10, lookback_hi=30, max_wide_rate=0.5):
    """Real goals have a wide_shot DIP in the 10-30s before the kickoff.
    Pass if wide_shot rate in [-lookback_hi, -lookback_lo] is BELOW max_wide_rate."""
    base = det["start_sec"]
    window = frames_in_range(frames, base - lookback_hi, base - lookback_lo)
    if len(window) < 2:
        return False
    wide_rate = sum(1 for f in window if f.get("wide_shot")) / len(window)
    return wide_rate <= max_wide_rate


def filt_in_circle_at_candidate(det, frames, max_inc=4):
    """Real kickoff: 1-4 players in center circle. Reject if too many."""
    base = det["start_sec"]
    nearby = frames_in_range(frames, base - 5, base + 5)
    if not nearby:
        return False
    in_circles = [f.get("in_circle", 0) for f in nearby]
    return min(in_circles) <= max_inc


def filt_two_teams_present(det, frames, min_per_side=5):
    """Both halves have ≥min_per_side players at the candidate frame."""
    base = det["start_sec"]
    nearby = frames_in_range(frames, base - 5, base + 5)
    return any(f.get("p_left", 0) >= min_per_side and f.get("p_right", 0) >= min_per_side
               for f in nearby)


def filt_sustained_wide_after(det, frames, lookahead_lo=15, lookahead_hi=60, min_wide_rate=0.6):
    """Wide shots sustained for ≥min_wide_rate fraction of frames in 15-60s after."""
    base = det["start_sec"]
    window = frames_in_range(frames, base + lookahead_lo, base + lookahead_hi)
    if len(window) < 3:
        return False
    wide_rate = sum(1 for f in window if f.get("wide_shot")) / len(window)
    return wide_rate >= min_wide_rate


def filt_ball_seen_drop(det, frames, before_lo=10, before_hi=30,
                       min_drop=0.3):
    """Ball-seen rate drops in [-30, -10]s vs game baseline (post-candidate avg).
    Pause after goal: ball not detected briefly."""
    base = det["start_sec"]
    pre = frames_in_range(frames, base - before_hi, base - before_lo)
    post = frames_in_range(frames, base + 5, base + 30)
    if len(pre) < 2 or len(post) < 2:
        return False
    pre_rate = sum(1 for f in pre if f.get("ball")) / len(pre)
    post_rate = sum(1 for f in post if f.get("ball")) / len(post)
    return (post_rate - pre_rate) >= min_drop


GAMES = {
    "game_22": (["/tmp/kickoff_game_22_formation_v2_base.jsonl"], None,
                ["/tmp/kickoff_game_22_frames.jsonl"],
                [g + 195 for g in [1559.7, 2195.4]] +
                [g + 195 + 690 for g in [3734.7, 4853.3, 5066.6]]),
    "game_21": (["/tmp/kickoff_game_21_formation_v2_base.jsonl"], None,
                ["/tmp/kickoff_game_21_frames.jsonl"],
                [g + 65 for g in [1578.6, 2008.6]]),
    "game_20": (["/tmp/kickoff_game20_1H_formation_base.jsonl",
                 "/tmp/kickoff_game20_2H_formation_base.jsonl"], None,
                ["/tmp/kickoff_game20_1H_frames.jsonl",
                 "/tmp/kickoff_game20_2H_frames.jsonl"],
                [g + 100 for g in [1072.2, 1137.0, 1639.2, 2314.2]] +
                [g + 775 for g in [2442.0, 3106.4, 3596.0, 3976.8, 4065.2]]),
    "rush": (["/tmp/kickoff_rush_formation_v2_base.jsonl"],
             "/tmp/kickoff_rush_pattern_v11_0191.jsonl",
             ["/tmp/kickoff_rush_frames.jsonl"],
             [g + 418 for g in [383.1, 647.0]] +
             [g + 418 + 770 for g in [3038.4, 3263.5]]),
}


FILTERS = [
    ("none", lambda d, f: True),
    ("preceding wide DIP <=50% in -30..-10s", filt_preceding_gap),
    ("preceding wide DIP <=70% in -30..-10s",
     lambda d, f: filt_preceding_gap(d, f, max_wide_rate=0.7)),
    ("preceding wide DIP <=70% in -45..-15s",
     lambda d, f: filt_preceding_gap(d, f, lookback_lo=15, lookback_hi=45, max_wide_rate=0.7)),
    ("in_circle <=4 at candidate", filt_in_circle_at_candidate),
    ("in_circle <=6 at candidate",
     lambda d, f: filt_in_circle_at_candidate(d, f, max_inc=6)),
    ("two-teams (≥5 per side)", filt_two_teams_present),
    ("two-teams (≥4 per side)",
     lambda d, f: filt_two_teams_present(d, f, min_per_side=4)),
    ("sustained wide ≥60% in +15..+60s", filt_sustained_wide_after),
    ("ball_seen drop +0.3", filt_ball_seen_drop),
    ("ball_seen drop +0.2",
     lambda d, f: filt_ball_seen_drop(d, f, min_drop=0.2)),
    # combinations
    ("preceding DIP ≤70% AND in_circle ≤6",
     lambda d, f: filt_preceding_gap(d, f, max_wide_rate=0.7)
                  and filt_in_circle_at_candidate(d, f, max_inc=6)),
    ("preceding DIP ≤70% AND sustained_wide ≥60%",
     lambda d, f: filt_preceding_gap(d, f, max_wide_rate=0.7)
                  and filt_sustained_wide_after(d, f)),
    ("all-of-above (DIP+in_c+two_teams+sustained)",
     lambda d, f: filt_preceding_gap(d, f, max_wide_rate=0.7)
                  and filt_in_circle_at_candidate(d, f, max_inc=6)
                  and filt_two_teams_present(d, f, min_per_side=4)
                  and filt_sustained_wide_after(d, f)),
]


print(f"{'filter':<48} {'TP':>3} {'FP':>4} {'recall':>6} {'prec':>6}")
print("-" * 70)
for label, fn in FILTERS:
    total_tp = 0; total_fp = 0; total_gts = 0
    for game, (formation_paths, pattern_path, frame_paths, gts) in GAMES.items():
        frames = load_frames(frame_paths)
        dets = [d for d in load_all_goals(formation_paths, pattern_path) if fn(d, frames)]
        tp, fp = gt_score(dets, gts)
        total_tp += tp; total_fp += fp; total_gts += len(gts)
    rec = total_tp / total_gts
    prec = total_tp / max(1, total_tp + total_fp)
    print(f"{label:<48} {total_tp:>3} {total_fp:>4} {rec:>6.2f} {prec:>6.2f}")
