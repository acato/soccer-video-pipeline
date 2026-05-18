"""Parameter sweep for kickoff detector on cached per-frame YOLO data.

Tries many threshold combinations against a known GT list, reports
the (recall, precision) of each combo. Use to find a config that
recovers more than 1 of 4 GT goals on game_20.

Usage:
    python scripts/sweep_kickoffs.py \\
        --per-frame /tmp/kickoff_game20_1H_frames.jsonl \\
        --gt 1196.2,1261.0,1763.2,2438.2 \\
        --tolerance 90
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from itertools import product
from copy import deepcopy

# We'll import-and-monkey-patch the detect_kickoffs module to vary thresholds.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_kickoffs as dk  # type: ignore  # noqa: E402


def load_raw(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "ball": r.get("ball"),
                "p_left": r.get("p_left", 0),
                "p_right": r.get("p_right", 0),
                "total_field": r.get("total_field", 0),
                "in_circle": r.get("in_circle", 0),
                "t": r["t"],
            })
    return rows


def run_with_config(raw: list[dict], cfg: dict) -> list[dict]:
    """Apply config to dk module globals, then re-derive + detect."""
    saved = {}
    for k, v in cfg.items():
        saved[k] = getattr(dk, k)
        setattr(dk, k, v)
    try:
        flags = dk.derive_flags(raw)
        ts = [r["t"] for r in raw]
        return dk.detect_goals(flags, ts, 5.0)
    finally:
        for k, v in saved.items():
            setattr(dk, k, v)


def score(detected: list[dict], gt: list[float], tol: float) -> tuple[int, int, int, list]:
    """Greedy match within ±tol seconds. Returns (TP, FP, FN, matches)."""
    matched = set()
    tp = 0
    matches = []
    for d in detected:
        best_g = None
        best_dt = float("inf")
        for gi, g in enumerate(gt):
            if gi in matched:
                continue
            dt = abs(d["start_sec"] - g)
            if dt < best_dt and dt <= tol:
                best_g = gi
                best_dt = dt
        if best_g is not None:
            matched.add(best_g)
            tp += 1
            matches.append((d["start_sec"], gt[best_g], best_dt))
    fp = len(detected) - tp
    fn = len(gt) - tp
    return tp, fp, fn, matches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-frame", required=True, type=Path)
    ap.add_argument("--gt", required=True, help="comma-separated GT goal video_sec list")
    ap.add_argument("--tolerance", type=float, default=90.0)
    args = ap.parse_args()

    gt = [float(x) for x in args.gt.split(",") if x.strip()]
    raw = load_raw(args.per_frame)
    print(f"loaded {len(raw)} frames, {len(gt)} GT goals", file=sys.stderr)

    # Parameter grid
    grid = {
        # Traversal lookback windows (frames at 5s interval → 15-90s)
        "TRAVERSAL_LOOKBACK_FRAMES_MIN": [3, 5, 7],
        "TRAVERSAL_LOOKBACK_FRAMES_MAX": [9, 12, 18],
        # Tightness of "ball at goal area" zone
        "TRAVERSAL_BALL_END_X_LO": [0.05, 0.10],
        "TRAVERSAL_BALL_END_X_HI": [0.90, 0.95],
        # Wide-shot floor
        "WIDE_SHOT_MIN_PERSONS": [12, 15, 18],
        # Center-spot Y bounds (camera-tilt sensitive)
        "CENTER_Y_LO": [0.30, 0.35],
        "CENTER_Y_HI": [0.50, 0.55],
        # Kickoff run gap tolerance
        "MAX_KICKOFF_GAP_FRAMES": [1, 2],
    }

    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    print(f"sweeping {len(combos)} combinations...", file=sys.stderr)

    results = []
    for vals in combos:
        cfg = dict(zip(keys, vals))
        # Validate: MAX > MIN
        if cfg["TRAVERSAL_LOOKBACK_FRAMES_MAX"] <= cfg["TRAVERSAL_LOOKBACK_FRAMES_MIN"]:
            continue
        if cfg["TRAVERSAL_BALL_END_X_HI"] <= cfg["TRAVERSAL_BALL_END_X_LO"] + 0.5:
            continue
        if cfg["CENTER_Y_HI"] <= cfg["CENTER_Y_LO"]:
            continue
        detected = run_with_config(raw, cfg)
        tp, fp, fn, matches = score(detected, gt, args.tolerance)
        results.append((tp, fp, fn, len(detected), cfg, matches))

    # Sort by: max TPs first, then min FPs, then prefer wider lookback (more lenient)
    results.sort(key=lambda r: (-r[0], r[1], r[3]))
    print(f"completed {len(results)} valid configs", file=sys.stderr)
    print()

    # Show the Pareto frontier — best (TP, FP) tradeoffs
    print(f"{'TP':<4}{'FP':<4}{'FN':<4}{'n_det':<7}config")
    seen = set()
    for tp, fp, fn, ndet, cfg, matches in results:
        key = (tp, fp)
        if key in seen:
            continue
        seen.add(key)
        cfg_str = " ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"{tp:<4}{fp:<4}{fn:<4}{ndet:<7}{cfg_str}")
        for d_t, g_t, dt in matches:
            print(f"     match: detected={d_t:.0f}s ↔ GT={g_t:.0f}s (Δ={dt:.0f}s)")
        if len(seen) >= 15:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
