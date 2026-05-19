"""VLM verifier v3 — wider window + tighter aggregation.

Changes from v2:
- Window expanded to ±60s (7 frames at -60, -40, -20, 0, +20, +40, +60) to
  catch goals that fall outside v2's [-15, +30] window (e.g. rush 4230 TP).
- Added `kickoff_restart` label: teams in formation at center circle after a
  goal — explicit signal for goal-aftermath.
- Aggregation requires either:
  (a) a `celebration` label, OR
  (b) a `goal` label followed by `active_play`/`idle`/`kickoff_restart`
      (the model labeled "shot near goal" as `goal` in v2; this filters
      shots that didn't result in a reset), OR
  (c) a `kickoff_restart` label preceded by `goal`/`celebration`/`set_piece`.
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
MODEL = "qwen3-vl-32b"  # produce_ensemble_goals.sh patches to -base during swap
FFMPEG = "/opt/homebrew/bin/ffmpeg"

LABELS = ("active_play", "set_piece", "celebration", "goal",
          "kickoff_restart", "replay", "idle", "close_up", "other")

LABEL_PROMPT = (
    "Classify this frame from a youth soccer match. Choose exactly one label "
    "from this list:\n"
    "- active_play: ball in play, normal field action\n"
    "- set_piece: corner, free kick, throw-in, goal kick being taken\n"
    "- celebration: players celebrating (arms raised, hugging, running to a "
    "scorer, group hug)\n"
    "- goal: ball entering net, ball INSIDE the net behind keeper, or keeper "
    "retrieving ball from inside the goal\n"
    "- kickoff_restart: teams in formation in their own halves, ball at the "
    "center spot, one or two players inside the center circle ready to kick "
    "off (this happens after a goal, at the start of each half)\n"
    "- replay: slow-motion replay graphics, zoomed action\n"
    "- idle: ball not in play, players walking, ref consulting, no clear "
    "action\n"
    "- close_up: tight zoom on individual players or coach\n"
    "- other: pre-game, halftime, post-game, scoreboard, sideline\n\n"
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
            label = label.split()[0].strip(".,:; ")
            return label if label in LABELS else f"unknown({label})"
        except Exception as e:
            if attempt == retries:
                return f"error({e})"
            time.sleep(1.0 * (attempt + 1))
    return "error"


def aggregate(labels: list[tuple[int, str]]) -> tuple[str, str]:
    """Sequence-aware aggregation. GOAL if any of:
      (a) any `celebration` label
      (b) any `goal` followed (chronologically) by `active_play` | `idle` |
          `kickoff_restart` (rules out lone "shot near goal" labels)
      (c) any `kickoff_restart` preceded by `goal` | `celebration` | `set_piece`
      (d) ≥2 `kickoff_restart` labels (sustained kickoff formation — covers
          the case where the base Qwen3-VL emits kickoff_restart without a
          preceding goal/celebration label because the LoRA-suppressed
          "celebration" label is absent on this camera)
    """
    labs = [(off, lbl) for off, lbl in labels]
    labs.sort(key=lambda x: x[0])

    if any(lbl == "celebration" for _, lbl in labs):
        return "GOAL", "celebration label present"

    for i, (_, lbl) in enumerate(labs):
        if lbl == "goal":
            for _, after in labs[i + 1:]:
                if after in ("active_play", "idle", "kickoff_restart"):
                    return "GOAL", f"goal followed by {after}"

    for i, (_, lbl) in enumerate(labs):
        if lbl == "kickoff_restart":
            for _, before in labs[:i]:
                if before in ("goal", "celebration", "set_piece"):
                    return "GOAL", f"kickoff_restart preceded by {before}"

    kr_count = sum(1 for _, lbl in labs if lbl == "kickoff_restart")
    if kr_count >= 2:
        return "GOAL", f"{kr_count} kickoff_restart labels"

    return "NO", f"labels={[lbl for _, lbl in labs]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workdir", default="/tmp/kickoff_vlm_v3_frames")
    args = p.parse_args()

    workdir = Path(args.workdir) / Path(args.video).stem
    dets = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    print(f"verifying {len(dets)} candidates from {args.inp}", file=sys.stderr)

    offsets = [-60, -40, -20, 0, +20, +40, +60]
    out_rows = []
    confirmed = 0
    t0 = time.time()
    for i, det in enumerate(dets):
        anchor = det.get("_kickoff_start", det["start_sec"])
        per_frame = []
        for off in offsets:
            t = anchor + off
            fp = workdir / f"k{int(anchor)}_o{off:+d}.jpg"
            extract(args.video, t, fp)
            label = classify_frame(b64(fp))
            per_frame.append((off, label))
        verdict, reason = aggregate(per_frame)
        det["_vlm_labels"] = per_frame
        det["_vlm_verdict"] = verdict
        det["_vlm_reason"] = reason
        out_rows.append(det)
        if verdict == "GOAL":
            confirmed += 1
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(dets) - i - 1) / max(rate, 0.01)
        labels_str = " ".join(f"{off:+d}:{lbl[:4]}" for off, lbl in per_frame)
        print(f"  [{i+1}/{len(dets)}] t={det['start_sec']:.0f}s  {verdict}  "
              f"{labels_str}  ({reason[:50]})  ETA {eta/60:.1f}min",
              file=sys.stderr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for d in out_rows:
            f.write(json.dumps(d) + "\n")
    print(f"confirmed {confirmed}/{len(out_rows)} → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
