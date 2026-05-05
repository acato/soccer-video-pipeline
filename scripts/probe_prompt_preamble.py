"""Probe: does a wide-camera preamble lift detection on Game 20?

Same 12 GT-aligned windows as probe_frame_width.py. Tests two prompt
variants at frame_width=1280:
  A. baseline production prompt
  B. with wide-camera preamble prepended

Hypothesis: the model's "outcome=goal" / shot detection requires confident
pose/ball recognition. On Game 20's high+wide camera, players are small
and the model defaults to "open play". A preamble that explicitly tells
the model the camera context may relax that default toward the trained
event types.
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
FRAME_WIDTH = 1280

WIDE_CAMERA_PREAMBLE = (
    "CAMERA NOTE: this match is filmed from a high, wide vantage point. "
    "Players appear small in the frame. Pose details (arms raised, both "
    "hands on ball, GK diving) are subtle and brief. Judge by overall "
    "kinematics — ball trajectory, player clustering, sideline/end-line "
    "geometry — not by tight pose detail. Be willing to call shots and "
    "GK actions you would otherwise dismiss as ambiguous; the camera "
    "framing rarely gives you the close-up confirmation you would expect.\n\n"
)

PROMPT_BODY = """\
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


def gt_event_video_times(*, event_filter: set[str]) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for half_idx, fp in enumerate((GT_H1, GT_H2)):
        d = json.loads(fp.read_text())
        for entry in d.get("data", []):
            game_sec = entry.get("event_time", 0) / 1000.0
            for ev in entry.get("events", []):
                name = ev.get("event_name", "")
                if name not in event_filter:
                    continue
                if half_idx == 0:
                    video_sec = game_sec + VIDEO_OFFSET
                else:
                    video_sec = (game_sec - HALF2_GAME_OFFSET) + HALF2_VIDEO
                out.append((video_sec, name))
    return out


def extract_frames(center_sec: float) -> list[bytes]:
    start = max(0.0, center_sec - WINDOW_SPAN_SEC / 2)
    interval = WINDOW_SPAN_SEC / N_FRAMES
    out = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(N_FRAMES):
            ts = start + i * interval + interval / 2
            jpg = td / f"f{i:02d}.jpg"
            subprocess.run([
                FFMPEG, "-hide_banner", "-loglevel", "error",
                "-ss", f"{ts:.3f}", "-i", str(VIDEO),
                "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "3",
                "-y", str(jpg),
            ], check=True)
            out.append(jpg.read_bytes())
    return out


def call_vllm(frames: list[bytes], win_start: float, win_end: float, *,
              preamble: bool) -> tuple[str, dict, float]:
    content: list[dict] = []
    for i, jpg in enumerate(frames):
        b64 = base64.b64encode(jpg).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        ts = win_start + (i + 0.5) * (win_end - win_start) / len(frames)
        content.append({"type": "text", "text": f"t={ts:.1f}s"})
    body = PROMPT_BODY.format(n_frames=len(frames), start=win_start, end=win_end)
    prompt = (WIDE_CAMERA_PREAMBLE + body) if preamble else body
    content.append({"type": "text", "text": prompt})
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
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
    goals = gt_event_video_times(event_filter={"Goals Conceded"})[:6]
    shots = gt_event_video_times(event_filter={"Shots & Goals"})
    goal_times = {t for t, _ in goals}
    pure_shots = [(t, n) for t, n in shots if all(abs(t - gt) > 30 for gt in goal_times)]
    step = max(1, len(pure_shots) // 6)
    shots_sample = pure_shots[::step][:6]
    targets = [(t, "goal", n) for t, n in goals] + [(t, "shot", n) for t, n in shots_sample]
    print(f"Probing {len(targets)} windows × 2 prompt variants (baseline, +preamble)")
    print(f"  6 GT goals + {len(shots_sample)} GT shots @ frame_width={FRAME_WIDTH}\n")

    summary = {
        False: {"shots": 0, "goal_outcome": 0, "save_outcome": 0,
                "any_event": 0, "lat": 0.0},
        True: {"shots": 0, "goal_outcome": 0, "save_outcome": 0,
               "any_event": 0, "lat": 0.0},
    }

    for video_t, kind, gt_name in targets:
        win_start = max(0.0, video_t - WINDOW_SPAN_SEC / 2)
        win_end = win_start + WINDOW_SPAN_SEC
        print(f"━━ {kind} '{gt_name}' @ t={video_t:.1f}s")
        try:
            frames = extract_frames(video_t)
        except subprocess.CalledProcessError as e:
            print(f"  frame extract failed: {e}")
            continue
        for preamble in (False, True):
            label = "+preamble" if preamble else "baseline "
            try:
                content, usage, lat = call_vllm(frames, win_start, win_end, preamble=preamble)
            except (error.HTTPError, error.URLError, TimeoutError) as e:
                print(f"  [{label}] vllm error: {e}")
                continue
            events = parse_events(content)
            shots_e = [e for e in events if e.get("event_type") in {"shot_on_target", "shot_off_target"}]
            outcomes = [e.get("outcome") for e in shots_e]
            if events:
                summary[preamble]["any_event"] += 1
            if shots_e:
                summary[preamble]["shots"] += 1
            for o in outcomes:
                if o == "goal": summary[preamble]["goal_outcome"] += 1
                elif o == "save": summary[preamble]["save_outcome"] += 1
            summary[preamble]["lat"] += lat
            etypes = [e.get("event_type", "?") for e in events]
            print(f"  [{label}] {lat:.1f}s tokens={usage.get('prompt_tokens')}p/{usage.get('completion_tokens')}c "
                  f"types={etypes} outcomes={outcomes}")

    n = len(targets)
    print(f"\n━━ SUMMARY (n={n}) ━━")
    print(f"{'metric':<22}{'baseline':>10}{'+preamble':>12}")
    for k in ("any_event", "shots", "goal_outcome", "save_outcome"):
        a, b = summary[False][k], summary[True][k]
        print(f"{k:<22}{a:>10}{b:>12}")
    print(f"{'avg_latency_sec':<22}{summary[False]['lat']/n:>10.2f}{summary[True]['lat']/n:>12.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
