#!/usr/bin/env python3
"""Convert v5 LoRA dataset (our format) to ms-swift training format.

Our v5 format (per-window):
  {game_id, split, window_idx, window_start_sec, window_end_sec,
   frames: [relative paths], frame_timestamps: [...], target: [event dicts]}

ms-swift format:
  {query: "<image>t=...s\n<image>t=...s\n...PROMPT...",
   response: "<JSON string of target list>",
   images: [absolute paths]}

Run on Mac (so /Volumes/transit/ is mounted), but rewrite image paths to
/mnt/transit/... since training runs on the Linux LLM server where the NAS
mounts at /mnt/transit/.

Usage:
    python convert_v5_to_swift.py \
        --src /Volumes/transit/soccer-finetune/lora_dataset_v5 \
        --dst /Volumes/transit/soccer-finetune/lora_dataset_v5_swift \
        [--frames-prefix /mnt/transit/soccer-finetune/lora_dataset_v5/]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.dual_pass_detector import _DIRECT_CLASSIFY_PROMPT


def convert_row(d: dict, frames_prefix: str) -> dict:
    """Convert one v5 row to swift format."""
    n = len(d["frames"])
    ts = d.get("frame_timestamps", [None] * n)

    # Interleaved <image>t=...s\n blocks (matches inference content layout)
    image_block_lines = []
    for t in ts:
        if t is not None:
            image_block_lines.append(f"<image>t={t:.1f}s")
        else:
            image_block_lines.append("<image>")
    image_block = "\n".join(image_block_lines)

    prompt_text = _DIRECT_CLASSIFY_PROMPT.format(
        n_frames=n,
        start=d["window_start_sec"],
        end=d["window_end_sec"],
    )
    query = image_block + "\n\n" + prompt_text

    # Response = compact JSON of target events (matches what the VLM outputs)
    response = json.dumps(d["target"], separators=(",", ":"))

    # Rewrite frame paths: v5 stores them relative to dataset root; swift needs
    # paths the LLM server can resolve.
    images = [frames_prefix + p for p in d["frames"]]

    return {
        "query": query,
        "response": response,
        "images": images,
        "_meta": {
            "game_id": d["game_id"],
            "split": d["split"],
            "window_idx": d["window_idx"],
            "n_gt_events_in_window": d.get("n_gt_events_in_window", 0),
        },
    }


# Default split — matches prepare_lora_dataset.py
DEFAULT_TRAIN = ["game_02", "game_03", "game_05", "game_06", "game_07", "game_08",
                 "game_09", "game_12", "game_14", "game_15", "game_16", "game_17",
                 "game_18", "game_19"]
DEFAULT_EVAL = ["game_01", "game_04", "game_10", "game_11", "game_13"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="v5 dataset root")
    ap.add_argument("--dst", required=True, help="swift dataset output dir")
    ap.add_argument("--frames-prefix",
                    default="/mnt/transit/soccer-finetune/lora_dataset_v5/",
                    help="Path prefix prepended to relative frame paths so the "
                         "LLM server can resolve them.")
    ap.add_argument("--train-games", nargs="*", default=DEFAULT_TRAIN)
    ap.add_argument("--eval-games", nargs="*", default=DEFAULT_EVAL)
    ap.add_argument("--smoke-n", type=int, default=0,
                    help="If >0, write a tiny smoke-test set with N examples to "
                         "{dst}/smoke.jsonl. Used by the memory-fit test.")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    train_path = dst / "train.jsonl"
    val_path = dst / "val.jsonl"

    n_train = 0
    type_counts_train = {}
    with open(train_path, "w") as fout:
        for g in args.train_games:
            p = src / "labels" / f"{g}.jsonl"
            if not p.exists():
                print(f"  WARNING: {p} missing", file=sys.stderr)
                continue
            with open(p) as fin:
                for line in fin:
                    d = json.loads(line)
                    out = convert_row(d, args.frames_prefix)
                    fout.write(json.dumps(out) + "\n")
                    n_train += 1
                    for ev in d.get("target", []):
                        type_counts_train[ev["event_type"]] = type_counts_train.get(ev["event_type"], 0) + 1

    n_val = 0
    type_counts_val = {}
    with open(val_path, "w") as fout:
        for g in args.eval_games:
            p = src / "labels" / f"{g}.jsonl"
            if not p.exists():
                print(f"  WARNING: {p} missing", file=sys.stderr)
                continue
            with open(p) as fin:
                for line in fin:
                    d = json.loads(line)
                    out = convert_row(d, args.frames_prefix)
                    fout.write(json.dumps(out) + "\n")
                    n_val += 1
                    for ev in d.get("target", []):
                        type_counts_val[ev["event_type"]] = type_counts_val.get(ev["event_type"], 0) + 1

    if args.smoke_n > 0:
        smoke_path = dst / "smoke.jsonl"
        with open(train_path) as fin, open(smoke_path, "w") as fout:
            for i, line in enumerate(fin):
                if i >= args.smoke_n:
                    break
                fout.write(line)
        print(f"  smoke set: {smoke_path} ({args.smoke_n} examples)")

    print(f"\nTrain: {n_train} examples → {train_path}")
    print(f"  type distribution: {type_counts_train}")
    print(f"Val:   {n_val} examples → {val_path}")
    print(f"  type distribution: {type_counts_val}")


if __name__ == "__main__":
    main()
