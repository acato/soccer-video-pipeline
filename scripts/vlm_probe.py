"""Probe: does the v11 VLM actually look at images, or just emit canned text?

Sends a single frame from a KNOWN TP (rush video 1042s = real goal celebration
visible) with a simple description prompt. If it responds with content that
references what's actually in the image (players, goal box, etc.), it sees.
If it gives a canned generic response, it's collapsing on prompt template.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

import requests

VLLM_URL = "http://10.10.2.222:8000/v1/chat/completions"
MODEL = "qwen3-vl-32b"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

# Known TP frame (real goal sequence per visual check earlier):
# rush video at t=1072 shows the shot/goal box scene
VIDEO = "/Users/aless/soccer-working/2026-02-07 - Rush - GA2008.mp4"
T = 1072

def extract(t, out):
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", str(t),
         "-i", VIDEO, "-frames:v", "1", "-vf", "scale=1280:-1",
         "-q:v", "3", "-y", out], check=True,
    )

frame = "/tmp/vlm_probe.jpg"
extract(T, frame)
b64 = base64.b64encode(Path(frame).read_bytes()).decode()
url = f"data:image/jpeg;base64,{b64}"

prompts = [
    ("describe", "Describe what you see in this frame in one or two sentences."),
    ("scene_label", "Classify this frame. Choose exactly one label from: "
        "active_play, set_piece, replay, idle, celebration, other. "
        "Respond with just the label."),
    ("event_json", 'Look at this frame from a soccer match. What event is happening? '
        'Respond in JSON: {"event": "goal" | "shot" | "active_play" | "corner_kick" | '
        '"throw_in" | "free_kick" | "goal_kick" | "celebration" | "other"}'),
    ("count_players", "How many soccer players do you see in this image?"),
]

for label, prompt in prompts:
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": 150,
        "temperature": 0.0,
    }
    r = requests.post(VLLM_URL, json=payload, timeout=60)
    reply = r.json()["choices"][0]["message"]["content"].strip()
    print(f"=== {label} ===")
    print(f"  prompt: {prompt[:80]}")
    print(f"  reply:  {reply}")
    print()
