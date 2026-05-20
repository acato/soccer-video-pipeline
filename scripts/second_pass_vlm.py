"""Second-pass VLM filter for ensemble GOAL candidates.

Sends 12 frames spanning ±30s of each candidate as ONE multi-image prompt
to base FP8, asking a single binary YES/NO question. Different from the
per-frame label aggregation we used in the ensemble — gives the model
full-sequence context to judge if a goal actually occurred.

Usage:
    python scripts/second_pass_vlm.py \\
        --video "/path/to/video.mp4" \\
        --candidates /tmp/kickoff_<game>_formation_v2_base.jsonl \\
        --out /tmp/kickoff_<game>_second_pass.jsonl
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
MODEL = "qwen3-vl-32b-base"
FFMPEG = "/opt/homebrew/bin/ffmpeg"

OFFSETS = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]  # ±60s

PROMPT = (
    "These 12 frames span ~120 seconds of a youth soccer match. "
    "Focus on the LAST 4 frames (the end of the sequence).\n\n"
    "Question: in those last frames, do you see TEAMS RESETTING TO HALVES "
    "for a kickoff at the CENTER CIRCLE? Specifically:\n"
    "- One team in each half of the field (players spread across both halves)\n"
    "- A player or two standing AT or INSIDE the center circle\n"
    "- Most players standing or walking (not all running with the ball)\n"
    "- Wide tactical camera framing (whole field visible)\n\n"
    "This pattern only occurs at the start of a half OR after a goal. "
    "If you see it AND the earlier frames don't look like the kickoff of a "
    "half (which would have a long quiet period before), conclude a goal "
    "happened.\n\n"
    "JSON on ONE line: "
    '{"kickoff_pattern":true|false,"is_half_start":true|false,"reason":"<short>"}'
)


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


def extract(video, t, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-ss", str(max(0.0, t)), "-i", video,
         "-frames:v", "1", "-vf", "scale=1280:-1", "-q:v", "3",
         "-y", str(out_path)], check=True,
    )


def b64(path):
    return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode()}"


def call_vlm(urls, retries=2):
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    content.append({"type": "text", "text": PROMPT})
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 80,
        "temperature": 0.0,
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(VLLM_URL, json=payload, timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == retries:
                return f"ERROR: {e}"
            time.sleep(1.0 * (attempt + 1))


def parse_verdict(reply):
    """Robust JSON parsing — model may add prefix/suffix.

    For the kickoff_pattern prompt: GOAL = kickoff_pattern is true AND
    is_half_start is false. NO otherwise.
    """
    import re
    lower = reply.lower()
    kp_match = re.search(r'"kickoff_pattern"\s*:\s*(true|false)', lower)
    hs_match = re.search(r'"is_half_start"\s*:\s*(true|false)', lower)
    if kp_match:
        kp = kp_match.group(1) == "true"
        hs = hs_match.group(1) == "true" if hs_match else False
        if kp and not hs:
            return "GOAL"
        return "NO"
    # Legacy fallback for old "goal" key
    m = re.search(r'"goal"\s*:\s*(true|false)', lower)
    if m:
        return "GOAL" if m.group(1) == "true" else "NO"
    return "AMBIGUOUS"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--candidates", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workdir", default="/tmp/kickoff_2pass_frames")
    p.add_argument("--filter-relaxed-only", action="store_true",
                   help="only verify candidates that fired GOAL under relaxed rule")
    args = p.parse_args()

    workdir = Path(args.workdir) / Path(args.video).stem
    cands = []
    for line in args.candidates.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if args.filter_relaxed_only:
            if "_vlm_labels" in r:
                if aggregate_relaxed(r["_vlm_labels"]) != "GOAL":
                    continue
            elif r.get("_vlm_verdict") != "GOAL":
                continue
        cands.append(r)
    print(f"verifying {len(cands)} candidates", file=sys.stderr)

    out_rows = []
    confirmed = 0
    t0 = time.time()
    for i, c in enumerate(cands):
        anchor = c.get("_kickoff_start", c["start_sec"])
        urls = []
        for off in OFFSETS:
            t = anchor + off
            fp = workdir / f"k{int(anchor)}_o{off:+d}.jpg"
            extract(args.video, t, fp)
            urls.append(b64(fp))
        reply = call_vlm(urls)
        verdict = parse_verdict(reply)
        c["_2pass_verdict"] = verdict
        c["_2pass_reply"] = reply
        out_rows.append(c)
        if verdict == "GOAL":
            confirmed += 1
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(cands) - i - 1) / max(rate, 0.01)
        print(f"  [{i+1}/{len(cands)}] t={c['start_sec']:.0f}s  {verdict}  "
              f"({rate:.2f}/s, ETA {eta/60:.1f}min)  {reply[:80]}",
              file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"confirmed {confirmed}/{len(out_rows)} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
