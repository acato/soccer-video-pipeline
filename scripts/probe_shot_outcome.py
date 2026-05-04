"""Probe: does v6 c757 obey a 'shot must declare outcome + justification' instruction?

Sampled windows:
  - 4 GT goal moments (test outcome=goal recall + justification quality)
  - 4 control windows where Run 62 emitted shot_on_target with no goal nearby
    (test outcome={save|goal_kick|corner_kick} variety; goal here would be FP)

Frames extracted via ffmpeg from the local Mac copy of the Rush video.
Posts to vLLM at http://10.10.2.222:8000 with the same image+text content
shape as production. Reports per-window:
  - did the model produce a shot event? (compliance)
  - did it include `outcome` and `outcome_justification`? (schema obedience)
  - what outcome did it pick?
  - readable justification?

Run on Mac:  .venv/bin/python scripts/probe_shot_outcome.py
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib import error, request

# ssh sessions on macOS lack /opt/homebrew on PATH; ffmpeg lives there.
os.environ["PATH"] = "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:" + os.environ.get("PATH", "")
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

# ── Config ─────────────────────────────────────────────────────────────────
VIDEO = Path("/Users/aless/soccer-working/2026-02-07 - Rush - GA2008.mp4")
VLLM_URL = "http://10.10.2.222:8000/v1/chat/completions"
MODEL = "qwen3-vl-32b"
WINDOW_SPAN_SEC = 15.0
N_FRAMES = 5
FRAME_WIDTH = 1280

# ── Targets ────────────────────────────────────────────────────────────────
GT_GOALS = [
    ("goal_802.9", 802.9, "GT goal"),
    ("goal_1064.1", 1064.1, "GT goal"),
    ("goal_2330.2", 2330.2, "GT goal"),
    ("goal_4398.4", 4398.4, "GT goal"),
]
# Run 62 produced shot_on_target events at many timestamps; these four are
# spread across the match and are NOT within ±60s of any GT goal, so any
# outcome="goal" here is a false positive.
CONTROL_SHOTS = [
    ("shot_374", 374.0, "control: random shot, no goal within ±60s"),
    ("shot_1840", 1840.0, "control: random shot, no goal within ±60s"),
    ("shot_3200", 3200.0, "control: random shot, no goal within ±60s"),
    ("shot_5400", 5400.0, "control: random shot, no goal within ±60s"),
]
PROBES = GT_GOALS + CONTROL_SHOTS


# ── Prompt ─────────────────────────────────────────────────────────────────
# Stripped-down version of _DIRECT_CLASSIFY_PROMPT plus the outcome rule.
# Keeping it short for the probe — full production prompt is ~5KB which works
# but adds noise; the model's task here is just "do you obey the new field?"
_PROBE_PROMPT = """\
You are analyzing {n_frames} frames from a soccer match ({start:.0f}s – {end:.0f}s).

For each DISTINCT event you see, classify it. Reply as a JSON array.

Standard event fields: event_type, start_sec, end_sec, confidence, reasoning.

Event types: throw_in, goal_kick, corner_kick, free_kick_shot, kickoff, catch,
shot_stop_diving, punch, goal, shot_on_target, shot_off_target.

ADDITIONAL REQUIREMENT — every shot_on_target and shot_off_target event MUST
include two extra fields:
  "outcome": one of "save" | "corner_kick" | "goal_kick" | "goal"
  "outcome_justification": a single sentence describing the specific visual
                           evidence for that outcome (e.g. "ball crosses line +
                           scorer arms raised", "GK catches at chest",
                           "ball deflects off defender past corner flag",
                           "ball stationary in 6-yard box, GK steps in to kick").

A shot without these two fields is invalid — pick the most likely outcome
from the four. Do NOT use "none" or "unknown" as an outcome.

If no notable event is visible, return:
{{"event_type": "none", "start_sec": {start}, "end_sec": {end}, "confidence": 0.9,
"reasoning": "open play"}}
"""


def extract_frames(video: Path, center_sec: float, n_frames: int,
                   span_sec: float, width: int) -> list[bytes]:
    """Extract n_frames JPEGs evenly across [center-span/2, center+span/2]."""
    start = max(0.0, center_sec - span_sec / 2)
    interval = span_sec / n_frames
    out = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(n_frames):
            ts = start + i * interval + interval / 2
            jpg = td / f"f{i:02d}.jpg"
            cmd = [
                FFMPEG, "-hide_banner", "-loglevel", "error",
                "-ss", f"{ts:.3f}", "-i", str(video),
                "-frames:v", "1",
                "-vf", f"scale={width}:-2",
                "-q:v", "3",
                "-y", str(jpg),
            ]
            subprocess.run(cmd, check=True)
            out.append(jpg.read_bytes())
    return out


def build_payload(frames: list[bytes], win_start: float, win_end: float) -> dict:
    content: list[dict] = []
    for i, jpg in enumerate(frames):
        b64 = base64.b64encode(jpg).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
        ts = win_start + (i + 0.5) * (win_end - win_start) / len(frames)
        content.append({"type": "text", "text": f"t={ts:.1f}s"})
    prompt = _PROBE_PROMPT.format(
        n_frames=len(frames),
        start=win_start,
        end=win_end,
    )
    content.append({"type": "text", "text": prompt})
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1500,
        "temperature": 0,
    }


def call_vllm(payload: dict, timeout: float = 180.0) -> tuple[str, dict]:
    req = request.Request(
        VLLM_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body["choices"][0]["message"]["content"], body["usage"]


def parse_events(text: str) -> list[dict[str, Any]]:
    """Best-effort parse of the JSON-array reply. Tolerates lead/trail prose."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # Sometimes the model emits a single object, not an array
        m2 = re.search(r"\{.*\}", text, re.DOTALL)
        if m2:
            try:
                return [json.loads(m2.group(0))]
            except json.JSONDecodeError:
                return []
        return []


def evaluate(events: list[dict], expected_kind: str) -> dict:
    shots = [e for e in events if e.get("event_type") in
             {"shot_on_target", "shot_off_target"}]
    has_outcome = sum(1 for e in shots if "outcome" in e)
    has_justification = sum(1 for e in shots if "outcome_justification" in e)
    outcomes = [e.get("outcome") for e in shots]
    return {
        "n_events": len(events),
        "n_shots": len(shots),
        "n_with_outcome": has_outcome,
        "n_with_justification": has_justification,
        "outcomes": outcomes,
        "expected_outcome_present": (
            "goal" in outcomes if expected_kind == "GT goal"
            else "goal" not in outcomes  # control: goal here = FP
        ),
    }


def main() -> None:
    print(f"Probing v6 c757 with shot-outcome instruction. {len(PROBES)} windows.")
    print(f"video={VIDEO}  vllm={VLLM_URL}  model={MODEL}")
    print()

    summary = {
        "compliance_total_shots": 0,
        "compliance_with_outcome": 0,
        "compliance_with_justification": 0,
        "gt_goals_total": len(GT_GOALS),
        "gt_goals_outcome_eq_goal": 0,
        "controls_total": len(CONTROL_SHOTS),
        "controls_outcome_eq_goal": 0,  # FPs
    }

    for label, t, kind in PROBES:
        win_start = max(0.0, t - WINDOW_SPAN_SEC / 2)
        win_end = win_start + WINDOW_SPAN_SEC

        print(f"━━ {label} (kind={kind}) — window [{win_start:.1f}, {win_end:.1f}] ━━")
        try:
            frames = extract_frames(VIDEO, t, N_FRAMES, WINDOW_SPAN_SEC, FRAME_WIDTH)
        except subprocess.CalledProcessError as e:
            print(f"  FRAME EXTRACT FAILED: {e}")
            continue
        payload = build_payload(frames, win_start, win_end)

        t0 = time.monotonic()
        try:
            content, usage = call_vllm(payload)
        except (error.HTTPError, error.URLError, TimeoutError) as e:
            print(f"  VLLM CALL FAILED: {e}")
            continue
        latency = time.monotonic() - t0
        events = parse_events(content)
        ev = evaluate(events, kind)

        summary["compliance_total_shots"] += ev["n_shots"]
        summary["compliance_with_outcome"] += ev["n_with_outcome"]
        summary["compliance_with_justification"] += ev["n_with_justification"]
        if kind == "GT goal" and "goal" in ev["outcomes"]:
            summary["gt_goals_outcome_eq_goal"] += 1
        if kind.startswith("control") and "goal" in ev["outcomes"]:
            summary["controls_outcome_eq_goal"] += 1

        print(f"  latency={latency:.1f}s tokens={usage}")
        print(f"  n_events={ev['n_events']} n_shots={ev['n_shots']} "
              f"with_outcome={ev['n_with_outcome']} "
              f"with_justification={ev['n_with_justification']}")
        if ev["outcomes"]:
            print(f"  outcomes: {ev['outcomes']}")
        for e in events:
            etype = e.get("event_type", "?")
            ts = e.get("start_sec", "?")
            outcome = e.get("outcome")
            justif = e.get("outcome_justification") or e.get("reasoning", "")
            tag = f"outcome={outcome}" if outcome else ""
            print(f"    [{etype}@{ts}] {tag}  {justif[:150]}")
        print()

    print("━━ SUMMARY ━━")
    print(f"shots emitted: {summary['compliance_total_shots']}")
    print(f"  with `outcome` field: {summary['compliance_with_outcome']} "
          f"({summary['compliance_with_outcome']*100/max(1,summary['compliance_total_shots']):.0f}%)")
    print(f"  with `outcome_justification`: {summary['compliance_with_justification']} "
          f"({summary['compliance_with_justification']*100/max(1,summary['compliance_total_shots']):.0f}%)")
    print(f"GT goal recall via outcome=goal: "
          f"{summary['gt_goals_outcome_eq_goal']}/{summary['gt_goals_total']}")
    print(f"Control FP via outcome=goal:     "
          f"{summary['controls_outcome_eq_goal']}/{summary['controls_total']}")


if __name__ == "__main__":
    main()
