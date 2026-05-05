"""Probe: does frame_width=1920 vs 1280 lift shot detection on Game 20?

Hypothesis: Game 20's higher/wider camera makes the ball ~3-5 pixels at
production frame_width=1280, invisible to the VLM. Bumping to 1920 should
make it ~5-8 pixels and let the model classify shots correctly.

Test: pick 6 GT shot timestamps from Game 20 + 6 GT goal timestamps. For
each, sample 5 frames at production sliding-window cadence and send to
vLLM with frame_width = {1280, 1920}. Count:
  - shot_on_target events emitted
  - shot_off_target events
  - outcome field (when shot emitted)
  - latency per call

If frame_width=1920 lifts shot emission from 35% to ≥70%, the lever works.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import error, request

os.environ["PATH"] = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:" + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

VIDEO = Path("/Users/aless/soccer-working/2026-04-18 Celtic - Reign GA 11.mp4")
GT_H1 = Path("/Users/aless/soccer-runs/gt/game20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_1st Half.json")
GT_H2 = Path("/Users/aless/soccer-runs/gt/game20/2026-04-18_Seattle Reign 2011 GA (U15) vs Seattle Celtic U15 (W)_2nd Half.json")
VIDEO_OFFSET = 124.0
HALF2_VIDEO = 3554.0
HALF2_GAME_OFFSET = 2400.0
VLLM_URL = "http://10.10.2.222:8000/v1/chat/completions"
MODEL = "qwen3-vl-32b"
WINDOW_SPAN_SEC = 15.0
N_FRAMES = 5

# Production prompt (Run 64+) — ask for outcome on shots.
PROMPT = """\
You are analyzing {n_frames} frames from a soccer match ({start:.0f}s – {end:.0f}s).

For each DISTINCT event you see, classify it. Reply as a JSON array. Each:
{{"event_type": "...", "start_sec": N, "end_sec": N, "confidence": 0.0-1.0,
"reasoning": "brief explanation"}}

Event types: throw_in, goal_kick, corner_kick, free_kick_shot, kickoff,
catch, shot_stop_diving, punch, goal, shot_on_target, shot_off_target.

REQUIRED for shots: every shot_on_target and shot_off_target MUST include:
  "outcome": one of "save" | "corner_kick" | "goal_kick" | "goal"
A shot without an outcome is invalid.

If no notable event is visible, return:
{{"event_type": "none", "start_sec": {start}, "end_sec": {end}, "confidence": 0.9,
"reasoning": "open play"}}
"""


def gt_event_video_times(gt_h1: Path, gt_h2: Path, *, event_filter: set[str]) -> list[tuple[float, str]]:
    """Return list of (video_time_sec, event_name) for events in event_filter."""
    out: list[tuple[float, str]] = []
    for half_idx, fp in enumerate((gt_h1, gt_h2)):
        d = json.loads(fp.read_text())
        for entry in d.get("data", []):
            game_sec = entry.get("event_time", 0) / 1000.0
            for ev in entry.get("events", []):
                name = ev.get("event_name", "")
                prop = ev.get("property", {}) or {}
                key = name + ("/" + prop.get("Type", "") if prop.get("Type") else "")
                if name not in event_filter and key not in event_filter:
                    continue
                if half_idx == 0:
                    video_sec = game_sec + VIDEO_OFFSET
                else:
                    video_sec = (game_sec - HALF2_GAME_OFFSET) + HALF2_VIDEO
                out.append((video_sec, name))
    return out


def extract_frames(video: Path, center_sec: float, n_frames: int,
                   span_sec: float, width: int) -> list[bytes]:
    start = max(0.0, center_sec - span_sec / 2)
    interval = span_sec / n_frames
    out = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(n_frames):
            ts = start + i * interval + interval / 2
            jpg = td / f"f{i:02d}.jpg"
            subprocess.run([
                FFMPEG, "-hide_banner", "-loglevel", "error",
                "-ss", f"{ts:.3f}", "-i", str(video),
                "-frames:v", "1",
                "-vf", f"scale={width}:-2",
                "-q:v", "3",
                "-y", str(jpg),
            ], check=True)
            out.append(jpg.read_bytes())
    return out


def call_vllm(frames: list[bytes], win_start: float, win_end: float) -> tuple[str, dict, float]:
    content: list[dict] = []
    for i, jpg in enumerate(frames):
        b64 = base64.b64encode(jpg).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        ts = win_start + (i + 0.5) * (win_end - win_start) / len(frames)
        content.append({"type": "text", "text": f"t={ts:.1f}s"})
    content.append({"type": "text", "text": PROMPT.format(
        n_frames=len(frames), start=win_start, end=win_end,
    )})
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": content}],
               "max_tokens": 800, "temperature": 0}
    req = request.Request(VLLM_URL, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read().decode())
    return (body["choices"][0]["message"]["content"],
            body.get("usage", {}),
            time.monotonic() - t0)


def parse_events(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def main() -> int:
    # Pick 6 GT goals + 6 GT shots from Game 20.
    goals = gt_event_video_times(GT_H1, GT_H2, event_filter={"Goals Conceded"})
    shots = gt_event_video_times(GT_H1, GT_H2, event_filter={"Shots & Goals"})
    goals = goals[:6]
    # 6 GT shots, evenly spaced and not overlapping with goals (within 30s).
    goal_times = {t for t, _ in goals}
    pure_shots = [(t, n) for t, n in shots if all(abs(t - gt) > 30 for gt in goal_times)]
    step = max(1, len(pure_shots) // 6)
    shots_sample = pure_shots[::step][:6]
    targets = [(t, "goal", n) for t, n in goals] + [(t, "shot", n) for t, n in shots_sample]
    print(f"Probing {len(targets)} windows × 2 frame_widths (1280, 1920)")
    print(f"  6 GT goals + {len(shots_sample)} GT shots\n")

    summary = {
        1280: {"shots_emitted": 0, "outcome_goal": 0, "outcome_save": 0,
               "outcome_corner": 0, "outcome_goalkick": 0, "any_event": 0,
               "lat_total": 0.0},
        1920: {"shots_emitted": 0, "outcome_goal": 0, "outcome_save": 0,
               "outcome_corner": 0, "outcome_goalkick": 0, "any_event": 0,
               "lat_total": 0.0},
    }

    for video_t, kind, gt_name in targets:
        win_start = max(0.0, video_t - WINDOW_SPAN_SEC / 2)
        win_end = win_start + WINDOW_SPAN_SEC
        print(f"━━ {kind} '{gt_name}' @ video_t={video_t:.1f}s win=[{win_start:.0f},{win_end:.0f}]")
        for width in (1280, 1920):
            try:
                frames = extract_frames(VIDEO, video_t, N_FRAMES, WINDOW_SPAN_SEC, width)
            except subprocess.CalledProcessError as e:
                print(f"  [w={width}] frame extract failed: {e}")
                continue
            try:
                content, usage, lat = call_vllm(frames, win_start, win_end)
            except (error.HTTPError, error.URLError, TimeoutError) as e:
                print(f"  [w={width}] vllm error: {e}")
                continue
            events = parse_events(content)
            shots = [e for e in events if e.get("event_type") in {"shot_on_target", "shot_off_target"}]
            outcomes = [e.get("outcome") for e in shots]
            if events:
                summary[width]["any_event"] += 1
            if shots:
                summary[width]["shots_emitted"] += 1
            for o in outcomes:
                if o == "goal": summary[width]["outcome_goal"] += 1
                elif o == "save": summary[width]["outcome_save"] += 1
                elif o == "corner_kick": summary[width]["outcome_corner"] += 1
                elif o == "goal_kick": summary[width]["outcome_goalkick"] += 1
            summary[width]["lat_total"] += lat
            etypes = [e.get("event_type", "?") for e in events]
            print(f"  [w={width}] {lat:.1f}s tokens={usage.get('prompt_tokens')}p/{usage.get('completion_tokens')}c "
                  f"types={etypes} outcomes={outcomes}")

    n = len(targets)
    print(f"\n━━ SUMMARY (n={n} windows per width) ━━")
    print(f"{'metric':<22}{'1280':>10}{'1920':>10}")
    for k in ("any_event", "shots_emitted", "outcome_goal", "outcome_save",
              "outcome_corner", "outcome_goalkick"):
        a, b = summary[1280][k], summary[1920][k]
        print(f"{k:<22}{a:>10}{b:>10}")
    print(f"{'avg_latency_sec':<22}{summary[1280]['lat_total']/n:>10.2f}{summary[1920]['lat_total']/n:>10.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
