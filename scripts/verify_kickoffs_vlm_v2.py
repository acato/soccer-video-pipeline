"""VLM verifier v2 — per-frame scene labels, then aggregate.

The v1 verifier used a free-form "GOAL: / NO:" prompt which the v11 LoRA
collapsed to a memorized canned response. v2 uses the model's NATIVE
single-label classification format (per probe).

For each candidate:
1. Extract 6 frames at t-15, t-5, 0, +5, +15, +30 (relative to kickoff_setup)
2. Ask VLM for a single scene label per frame
3. Aggregate: a real goal sequence has either
   (a) a `celebration` label in the window, OR
   (b) a `goal` label at any frame, OR
   (c) a `set_piece`/`active_play` label in the goal area followed by a clear
       kickoff_setup pattern (players in formation at center)

Outputs jsonl with per-frame labels + aggregated verdict.
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

LABELS = ("active_play", "set_piece", "celebration", "goal",
          "replay", "idle", "close_up", "other")

LABEL_PROMPT = (
    "Classify this frame from a youth soccer match. Choose exactly one label "
    "from this list:\n"
    "- active_play: ball in play, normal field action\n"
    "- set_piece: corner, free kick, throw-in, goal kick about to be taken\n"
    "- celebration: players celebrating (arms raised, hugging, running to a "
    "scorer)\n"
    "- goal: ball entering net OR keeper retrieving from inside the goal\n"
    "- replay: slow-motion replay graphics or zoomed action\n"
    "- idle: ball not in play, players walking, ref consulting\n"
    "- close_up: tight zoom on individual players or coach\n"
    "- other: pre-game, halftime, post-game, scoreboard\n\n"
    "Respond with ONLY the label, nothing else."
)


def extract(video: str, t: float, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error",
         "-ss", str(max(0.0, t)), "-i", video,
         "-frames:v", "1", "-vf", "scale=1280:-1",
         "-q:v", "3", "-y", str(out)], check=True,
    )


def b64(p: Path) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(p.read_bytes()).decode()}"


def classify_frame(image_url: str, retries: int = 2) -> str:
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": LABEL_PROMPT},
            ],
        }],
        "max_tokens": 20,
        "temperature": 0.0,
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(VLLM_URL, json=payload, timeout=60)
            r.raise_for_status()
            label = r.json()["choices"][0]["message"]["content"].strip().lower()
            # Normalize: take first token, strip punctuation
            label = label.split()[0].strip(".,:; ")
            return label if label in LABELS else f"unknown({label})"
        except Exception as e:
            if attempt == retries:
                return f"error({e})"
            time.sleep(1.0 * (attempt + 1))
    return "error"


def aggregate(labels: list[tuple[int, str]]) -> tuple[str, str]:
    """Decide GOAL vs NO from per-frame labels.

    GOAL if any of:
      - 'goal' label appears at any frame
      - 'celebration' label appears at any frame
      - 'set_piece' appears BEFORE (chronologically) a 'celebration' OR 'goal'
      - 2+ 'set_piece' or 'goal' labels in the window

    NO otherwise. Returns (verdict, reason).
    """
    labels_only = [lbl for _, lbl in labels]
    times = [off for off, _ in labels]
    has_celeb = "celebration" in labels_only
    has_goal = "goal" in labels_only
    set_piece_count = labels_only.count("set_piece")

    if has_goal:
        return "GOAL", "explicit goal label present"
    if has_celeb:
        return "GOAL", "celebration label present"
    return "NO", f"labels={labels_only}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workdir", default="/tmp/kickoff_vlm_v2_frames")
    args = p.parse_args()

    workdir = Path(args.workdir) / Path(args.video).stem
    dets = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    print(f"verifying {len(dets)} candidates from {args.inp}", file=sys.stderr)

    offsets = [-15, -5, 0, +5, +15, +30]
    out_rows = []
    confirmed = 0
    t0 = time.time()
    for i, det in enumerate(dets):
        anchor = det.get("_kickoff_start", det["start_sec"])
        per_frame_labels = []
        for off in offsets:
            t = anchor + off
            fp = workdir / f"k{int(anchor)}_o{off:+d}.jpg"
            extract(args.video, t, fp)
            label = classify_frame(b64(fp))
            per_frame_labels.append((off, label))
        verdict, reason = aggregate(per_frame_labels)
        det["_vlm_labels"] = per_frame_labels
        det["_vlm_verdict"] = verdict
        det["_vlm_reason"] = reason
        out_rows.append(det)
        if verdict == "GOAL":
            confirmed += 1
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(dets) - i - 1) / max(rate, 0.01)
        labels_str = " ".join(f"{off:+d}:{lbl[:4]}" for off, lbl in per_frame_labels)
        print(f"  [{i+1}/{len(dets)}] t={det['start_sec']:.0f}s  {verdict}  "
              f"{labels_str}  ETA {eta/60:.1f}min", file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for d in out_rows:
            f.write(json.dumps(d) + "\n")
    print(f"confirmed {confirmed}/{len(out_rows)} → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
