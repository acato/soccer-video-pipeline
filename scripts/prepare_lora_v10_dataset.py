"""Inject v9b ball-trajectory annotations into v6 swift dataset prompts.

Reads:  /mnt/transit/soccer-finetune/lora_dataset_v6_swift/{train,val,smoke,val_small}.jsonl
Writes: /mnt/transit/soccer-finetune/lora_dataset_v10_swift/{train,val,smoke,val_small}.jsonl

For each training example:
  1. Parse the 5 per-frame timestamps from the v6 query text
  2. Run v9b on each frame jpeg → top-K detections
  3. Compute trajectory via ball_trajectory.compute_track / format_track
  4. Inject the prefix into the query (before the analysis prompt) AND
     append the [ball-track] summary block at the end

Result: training examples that teach the 32B to reason over trajectory
text, the same way the runtime prompt is structured.

Run on llm AFTER vLLM is stopped (needs GPU 0 free for v9b).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import cv2  # type: ignore

SRC_DIR = Path("/mnt/transit/soccer-finetune/lora_dataset_v6_swift")
DST_DIR = Path("/mnt/transit/soccer-finetune/lora_dataset_v10_swift")
V9B_WEIGHTS = "/mnt/transit/soccer-finetune/yolo_ball_v9/weights/v9b_best.pt"
CONF = 0.05
IMGSZ = 1920
MAX_DETS = 3

# Import the trajectory module from the runtime so train + inference stay
# in lock-step (same prefix, same formatter, same zone logic).
sys.path.insert(0, str(Path.home() / "soccer-video-pipeline"))
from src.detection.ball_trajectory import compute_track, format_track, prompt_prefix  # noqa: E402

TS_PAT = re.compile(r"<image>t=([\d.]+)s")
PROMPT_START = re.compile(r"You are analyzing")


def _detect_balls_from_jpg(model, jpg_path):
    img = cv2.imread(str(jpg_path))
    if img is None:
        return []
    results = model([img], imgsz=IMGSZ, conf=CONF, verbose=False)
    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    confs = boxes.conf.cpu().numpy()
    xywhn = boxes.xywhn.cpu().numpy()
    order = confs.argsort()[::-1][:MAX_DETS]
    return [
        {"cx": float(xywhn[i][0]), "cy": float(xywhn[i][1]), "conf": float(confs[i])}
        for i in order
    ]


def _inject(query: str, traj_summary: str, prefix: str) -> str:
    """Insert prefix before 'You are analyzing' and append summary at end."""
    m = PROMPT_START.search(query)
    if not m:
        return query + "\n\n" + traj_summary
    pre = query[:m.start()]
    body = query[m.start():]
    return pre + prefix + body + "\n\n" + traj_summary


def _process_split(model, split: str) -> tuple[int, int]:
    src = SRC_DIR / f"{split}.jsonl"
    if not src.exists():
        return 0, 0
    dst = DST_DIR / f"{split}.jsonl"

    n = 0
    n_with_ball = 0
    t0 = time.time()
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            ex = json.loads(line)
            timestamps = [float(t) for t in TS_PAT.findall(ex["query"])]
            if len(timestamps) != len(ex["images"]):
                # Schema mismatch — keep example as-is
                f_out.write(line)
                n += 1
                continue
            per_frame_dets = [_detect_balls_from_jpg(model, p) for p in ex["images"]]
            traj = compute_track(per_frame_dets, timestamps)
            summary = format_track(traj)
            ex["query"] = _inject(ex["query"], summary, prompt_prefix())
            if traj["n_with_ball"] > 0:
                n_with_ball += 1
            f_out.write(json.dumps(ex) + "\n")
            n += 1
            if n % 500 == 0:
                rate = n / max(0.1, time.time() - t0)
                print(f"  [{split}] {n} processed, {rate:.1f}/s, "
                      f"{n_with_ball/n*100:.0f}% with ball")
    print(f"  [{split}] DONE: {n} examples, {n_with_ball/n*100:.0f}% with ball, "
          f"elapsed {(time.time()-t0):.0f}s → {dst}")
    return n, n_with_ball


def main() -> int:
    if not Path(V9B_WEIGHTS).exists():
        print(f"ERROR: v9b weights not found: {V9B_WEIGHTS}", file=sys.stderr)
        return 1
    DST_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO  # type: ignore
    print(f"Loading v9b from {V9B_WEIGHTS}")
    model = YOLO(V9B_WEIGHTS)

    totals = {"n": 0, "with_ball": 0}
    for split in ("smoke", "val_small", "val", "train"):
        n, nb = _process_split(model, split)
        totals["n"] += n
        totals["with_ball"] += nb

    print(f"\n=== TOTALS ===")
    print(f"examples: {totals['n']}, with-ball: {totals['with_ball']} "
          f"({totals['with_ball']/max(1,totals['n'])*100:.0f}%)")
    print(f"\nv10 dataset at {DST_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
