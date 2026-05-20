"""Test if v11 LoRA gives real binary responses to a goal-detection prompt
with 12 frames of context. Earlier test ('GOAL: / NO:' format) collapsed
to canned 'no GT goal in window'. Try a different output format.
"""
import base64
import json
import subprocess
from pathlib import Path

import requests

VLLM_URL = "http://10.10.2.222:8000/v1/chat/completions"
MODEL = "qwen3-vl-32b"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

VIDEO = "/Users/aless/soccer-working/2026-02-07 - Rush - GA2008.mp4"

# Known TP for rush at GT 1042 — candidate t=990 in pattern_v11 (celebration_cut)
# Probe ±30s around t=990
TEST_CASES = [
    ("rush TP @ 990 (real goal)", 990.0),
    ("rush FP @ 1455 (not a real goal)", 1455.0),
    ("game_22 unrelated middle", 3500.0),
]

PROMPTS = {
    "v1_GOAL_NO": (
        "You see 12 frames sampled across 60 seconds of a soccer match. "
        "Did a GOAL get scored in this sequence (ball entering net, "
        "celebration, kickoff reset)? Reply 'GOAL: ...' or 'NO: ...'."
    ),
    "v2_json_verdict": (
        "Examine these 12 frames from a soccer match (60 seconds total, "
        "5s apart). Look for: ball going into the net, players celebrating, "
        "teams resetting to halfway line. Respond ONLY in JSON: "
        '{"goal_scored": true|false, "evidence": "<one short sentence>"}'
    ),
    "v3_describe_then_decide": (
        "Look at these 12 frames spanning 60 seconds of a soccer match. "
        "Step 1: Describe in one sentence what's happening across the sequence. "
        "Step 2: Did a goal occur? Reply 'Yes' or 'No' on a new line."
    ),
}


def extract(t, out):
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", str(t),
         "-i", VIDEO, "-frames:v", "1", "-vf", "scale=1280:-1",
         "-q:v", "3", "-y", out], check=True,
    )


def b64(p):
    return f"data:image/jpeg;base64,{base64.b64encode(Path(p).read_bytes()).decode()}"


for label, anchor in TEST_CASES:
    print(f"\n{'='*60}\n{label}  anchor={anchor}")
    # extract 12 frames at -30, -25, ..., +25
    frame_urls = []
    for i, off in enumerate([-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]):
        t = max(0.0, anchor + off)
        fp = f"/tmp/probe_v11_{int(anchor)}_{i}.jpg"
        if not Path(fp).exists():
            extract(t, fp)
        frame_urls.append(b64(fp))

    for prompt_name, prompt in PROMPTS.items():
        content = [{"type": "image_url", "image_url": {"url": u}} for u in frame_urls]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 150,
            "temperature": 0.0,
        }
        try:
            r = requests.post(VLLM_URL, json=payload, timeout=120)
            reply = r.json()["choices"][0]["message"]["content"].strip()
            print(f"  [{prompt_name}] {reply[:200]}")
        except Exception as e:
            print(f"  [{prompt_name}] ERROR: {e}")
