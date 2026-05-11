"""v9b ball-trajectory features for 32B classification prompts (Phase 2).

Consumes per-frame v9b detections across a single classification window
and emits a digested text summary that 32B can reason over directly —
without having to compute trajectory from raw coordinates itself.

Features extracted:
  - n_with_ball / n_total (continuity)
  - track of best-conf detection per frame
  - per-segment velocity + overall direction
  - end-position zone (left/right/top/bottom edges, penalty-area heuristic)
  - max speed, motion descriptor (stationary / moderate / fast / very fast)
  - "ended near edge" flag — strong signal for goal/throw-in/corner

Output is a 2-4 line text block intended to replace per-frame
[ball@...] annotations; richer signal at the cost of one extra block of
text per window.
"""
from __future__ import annotations

from typing import Optional

import structlog

log = structlog.get_logger(__name__)


_BALL_TRAJECTORY_PROMPT_PREFIX = (
    "Below the frames you will see a `[ball-track]` block — a digested summary "
    "of an external ball detector's (v9b YOLO) observations across this "
    "window. It reports how many frames contained a ball, the trajectory "
    "(positions over time, normalized 0-1 coords with 0,0=top-left), "
    "direction, speed, and which zone the ball ended in. The detector is "
    "noisy (~60% recall, low per-frame conf) but the *trajectory* across "
    "frames is reliable signal for events. Use the track to disambiguate:\n"
    "  • shot toward goal → ball moves fast toward left/right edge\n"
    "  • goal → ball ends near left/right edge after fast motion\n"
    "  • save → ball changes direction sharply, GK visible\n"
    "  • throw-in → ball stationary near top/bottom edge of pitch\n"
    "  • goal-kick → ball stationary in left/right penalty area\n"
    "  • corner → ball stationary at extreme corner\n"
    "Treat the track as a HINT — verify against what you see — but a long, "
    "continuous, fast trajectory toward an edge is unambiguous evidence of a "
    "shot, even if you cannot resolve the ball visually.\n\n"
)


def _classify_end_zone(x: float, y: float) -> list[str]:
    """Return list of zone tags for a normalized position."""
    zones = []
    if x < 0.10:
        zones.append("left-edge")
    elif x > 0.90:
        zones.append("right-edge")
    if y < 0.15:
        zones.append("top-edge")
    elif y > 0.85:
        zones.append("bottom-edge")
    if x < 0.20 and 0.20 < y < 0.80:
        zones.append("left-penalty-area")
    elif x > 0.80 and 0.20 < y < 0.80:
        zones.append("right-penalty-area")
    return zones


def _classify_speed(max_speed: float) -> str:
    if max_speed > 0.30:
        return "very fast"
    if max_speed > 0.15:
        return "fast"
    if max_speed > 0.05:
        return "moderate"
    return "stationary"


def _classify_direction(dx: float, dy: float) -> str:
    parts = []
    if abs(dx) > 0.05:
        parts.append("rightward" if dx > 0 else "leftward")
    if abs(dy) > 0.05:
        parts.append("downward" if dy > 0 else "upward")
    return " ".join(parts) if parts else "stationary"


def compute_track(per_frame_dets: list[list[dict]], timestamps: list[float]) -> dict:
    """Compute a single-ball trajectory across the window.

    per_frame_dets: per-frame list of top-K detections (from ball_context.detect_balls).
    timestamps: per-frame video timestamps (seconds).

    Strategy: pick highest-conf detection per frame as "the ball" for that
    frame. This is naive (a low-conf candidate could be a penalty spot) but
    works as a starting point. Future: data-association across frames.
    """
    assert len(per_frame_dets) == len(timestamps)
    n_total = len(timestamps)

    track: list[Optional[dict]] = []
    for dets, t in zip(per_frame_dets, timestamps):
        if not dets:
            track.append(None)
        else:
            best = dets[0]  # already conf-sorted desc
            track.append({"t": t, "x": best["cx"], "y": best["cy"], "conf": best["conf"]})

    n_with_ball = sum(1 for p in track if p is not None)
    # Velocity between each non-None frame and its NEXT non-None frame (skipping
    # missed-detection gaps). Gives a continuous trajectory feel without
    # requiring detection on every frame.
    velocities = []
    last_seen = None
    for p in track:
        if p is None:
            continue
        if last_seen is not None:
            dt = max(0.1, p["t"] - last_seen["t"])
            dx = p["x"] - last_seen["x"]
            dy = p["y"] - last_seen["y"]
            speed = (dx * dx + dy * dy) ** 0.5 / dt
            velocities.append({"from": last_seen, "to": p, "dx": dx, "dy": dy,
                               "dt": dt, "speed": speed})
        last_seen = p

    max_speed = max((v["speed"] for v in velocities), default=0.0)
    first = next((p for p in track if p), None)
    last = next((p for p in reversed(track) if p), None)

    overall_dx = (last["x"] - first["x"]) if first and last else 0.0
    overall_dy = (last["y"] - first["y"]) if first and last else 0.0

    end_zones = _classify_end_zone(last["x"], last["y"]) if last else []

    return {
        "n_with_ball": n_with_ball,
        "n_total": n_total,
        "track": track,
        "velocities": velocities,
        "max_speed": max_speed,
        "first_pos": first,
        "last_pos": last,
        "overall_dx": overall_dx,
        "overall_dy": overall_dy,
        "end_zones": end_zones,
    }


def format_track(traj: dict) -> str:
    """Format trajectory dict as a 2-4 line text summary for the 32B prompt."""
    n_w = traj["n_with_ball"]
    n_t = traj["n_total"]

    if n_w == 0:
        return f"[ball-track] not detected in any of {n_t} frames"

    if n_w == 1:
        p = next(p for p in traj["track"] if p is not None)
        return (f"[ball-track] detected in 1/{n_t} frames at "
                f"({p['x']:.2f},{p['y']:.2f}) conf={p['conf']:.2f}")

    # Multi-frame trajectory
    pts = [p for p in traj["track"] if p is not None]
    path_str = " -> ".join(
        f"({p['x']:.2f},{p['y']:.2f})@t={p['t']:.1f}s" for p in pts
    )
    direction = _classify_direction(traj["overall_dx"], traj["overall_dy"])
    speed_desc = _classify_speed(traj["max_speed"])
    end_zone_str = ", ".join(traj["end_zones"]) if traj["end_zones"] else "mid-field"

    lines = [
        f"[ball-track] detected in {n_w}/{n_t} frames",
        f"  path: {path_str}",
        f"  motion: {direction} (dx={traj['overall_dx']:+.2f}, dy={traj['overall_dy']:+.2f}), "
        f"max_speed={traj['max_speed']:.2f}/sec ({speed_desc})",
        f"  ended in: {end_zone_str}",
    ]
    return "\n".join(lines)


def prompt_prefix() -> str:
    return _BALL_TRAJECTORY_PROMPT_PREFIX
