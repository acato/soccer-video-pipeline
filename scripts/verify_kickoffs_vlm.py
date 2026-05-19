"""VLM verification of kickoff-pattern goal candidates.

For each detected goal in a kickoff_<game>.jsonl file, extract 4 frames
around the kickoff_setup window (-15s, -5s, +5s, +15s) and ask the VLM
"did a goal happen in this sequence?". Write a new JSONL with verdicts.

Usage:
    python scripts/verify_kickoffs_vlm.py \\
        --video "/path/to/game.mp4" \\
        --in /tmp/kickoff_rush.jsonl \\
        --out /tmp/kickoff_rush_vlm.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

VLLM_URL = "http://10.10.2.222:8000/v1/chat/completions"
MODEL = "qwen3-vl-32b"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

PROMPT = (
    "These 4 frames are consecutive samples from a youth soccer match video. "
    "They are 10 seconds apart, spanning a ~30s window around a candidate goal "
    "moment that a heuristic detector flagged.\n\n"
    "Frame 1 = -15s, Frame 2 = -5s, Frame 3 = +5s, Frame 4 = +15s "
    "(relative to the candidate kickoff moment).\n\n"
    "QUESTION: Was a GOAL scored in this sequence?\n"
    "- YES signals: ball entering the net, goalkeeper retrieving ball from inside "
    "the goal, players celebrating with raised arms, teams returning to the halfway "
    "line in formation for a center-circle kickoff restart.\n"
    "- NO signals: normal midfield play with the ball moving through the center, "
    "corner kicks, throw-ins, goal kicks, fouls being awarded, free kicks, "
    "warm-up or halftime activity.\n\n"
    "Answer in EXACTLY this format on a single line:\n"
    "GOAL: <one short sentence>   (if you see goal-scoring evidence)\n"
    "NO: <one short sentence>     (otherwise)"
)


def extract_frame(video: str, t: float, out_path: Path, width: int = 1280):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-ss", str(max(0.0, t)), "-i", video,
         "-frames:v", "1", "-vf", f"scale={width}:-1",
         "-q:v", "3", "-y", str(out_path)],
        check=True,
    )


def b64_image(path: Path) -> str:
    data = path.read_bytes()
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"


def call_vlm(image_urls: list[str], retries: int = 2) -> tuple[str, str]:
    content = [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
    content.append({"type": "text", "text": PROMPT})
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 80,
        "temperature": 0.0,
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(VLLM_URL, json=payload, timeout=180)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"].strip()
            return reply, _parse(reply)
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"VLM call failed after retries: {last_err}")


def _parse(reply: str) -> str:
    head = reply.upper().lstrip()
    if head.startswith("GOAL"):
        return "GOAL"
    if head.startswith("NO"):
        return "NO"
    return "AMBIGUOUS"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workdir", default="/tmp/kickoff_vlm_frames")
    args = p.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"video not found: {args.video}")
    workdir = Path(args.workdir) / Path(args.video).stem

    dets = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    print(f"verifying {len(dets)} candidates from {args.inp}", file=sys.stderr)

    out_rows = []
    confirmed = 0
    t0 = time.time()
    for i, det in enumerate(dets):
        anchor = det.get("_kickoff_start", det["start_sec"])
        offsets = [-15, -5, +5, +15]
        urls = []
        for off in offsets:
            t = anchor + off
            fp = workdir / f"k{int(anchor)}_o{off:+d}.jpg"
            extract_frame(args.video, t, fp)
            urls.append(b64_image(fp))
        try:
            reply, verdict = call_vlm(urls)
        except Exception as e:
            reply, verdict = str(e), "ERROR"
        det["_vlm_verdict"] = verdict
        det["_vlm_reply"] = reply
        out_rows.append(det)
        if verdict == "GOAL":
            confirmed += 1
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(dets) - i - 1) / max(rate, 0.01)
        print(f"  [{i+1}/{len(dets)}] t={det['start_sec']:.0f}s  verdict={verdict}  "
              f"({rate:.2f}/s, ETA {eta/60:.1f}min)  reply={reply[:80]}",
              file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for d in out_rows:
            f.write(json.dumps(d) + "\n")
    print(f"confirmed {confirmed}/{len(out_rows)} → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
