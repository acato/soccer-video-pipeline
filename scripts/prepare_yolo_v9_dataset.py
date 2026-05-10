"""Build YOLOv8 dataset from the manual ball annotations.

Reads:  /mnt/transit/soccer-finetune/yolo_ball_annotation/{frames,labels,manifest.jsonl}
Writes: /mnt/transit/soccer-finetune/yolo_ball_v9/
          images/{train,val}/<game>_<idx>.jpg
          labels/{train,val}/<game>_<idx>.txt   (empty file for background)
          data.yaml

Stratified 80/20 split *within each game* so every game appears in both
train and val. Skipped frames are kept as background examples (empty label
files) — they're how YOLO learns that penalty spots / shadows are not balls.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

SRC = Path("/mnt/transit/soccer-finetune/yolo_ball_annotation")
DST = Path("/mnt/transit/soccer-finetune/yolo_ball_v9")
VAL_FRACTION = 0.20
SEED = 42


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"source not found: {SRC}")

    manifest = [json.loads(l) for l in (SRC / "manifest.jsonl").read_text().splitlines() if l.strip()]
    labels_src = SRC / "labels"
    frames_src = SRC / "frames"

    # Bucket entries by game and by labeled/skipped status
    by_game: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"labeled": [], "skipped": []})
    for entry in manifest:
        idx = entry["idx"]
        gid = entry["game_id"]
        lab = labels_src / f"{idx:04d}.txt"
        skip = labels_src / f"{idx:04d}.skip"
        if lab.exists():
            by_game[gid]["labeled"].append({**entry, "_kind": "labeled"})
        elif skip.exists():
            by_game[gid]["skipped"].append({**entry, "_kind": "skipped"})
        # else: unannotated — drop

    rng = random.Random(SEED)

    # Wipe any prior v9 dataset
    if DST.exists():
        shutil.rmtree(DST)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DST / sub).mkdir(parents=True, exist_ok=True)

    stats = {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}}
    per_game_split = {}

    for gid, buckets in sorted(by_game.items()):
        per_game_split[gid] = {"train": 0, "val": 0, "pos_train": 0, "pos_val": 0}
        for kind in ("labeled", "skipped"):
            entries = list(buckets[kind])
            rng.shuffle(entries)
            n_val = max(1, int(round(len(entries) * VAL_FRACTION))) if entries else 0
            for i, entry in enumerate(entries):
                split = "val" if i < n_val else "train"
                idx = entry["idx"]
                stem = f"{gid}_{idx:04d}"
                src_jpg = frames_src / f"{idx:04d}.jpg"
                dst_jpg = DST / f"images/{split}" / f"{stem}.jpg"
                dst_txt = DST / f"labels/{split}" / f"{stem}.txt"
                shutil.copy2(src_jpg, dst_jpg)
                if kind == "labeled":
                    shutil.copy2(labels_src / f"{idx:04d}.txt", dst_txt)
                    stats[split]["pos"] += 1
                    per_game_split[gid][f"pos_{split}"] += 1
                else:
                    dst_txt.write_text("")  # empty = background sample
                    stats[split]["neg"] += 1
                per_game_split[gid][split] += 1

    # data.yaml
    yaml = (
        f"# v9 ball detector — manually annotated new-venue frames\n"
        f"path: {DST}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: ball\n"
    )
    (DST / "data.yaml").write_text(yaml)

    # Print summary
    print("=== per-game split ===")
    for gid, s in per_game_split.items():
        print(f"  {gid}: train={s['train']} (pos={s['pos_train']})  val={s['val']} (pos={s['pos_val']})")
    print("\n=== totals ===")
    for split in ("train", "val"):
        p, n = stats[split]["pos"], stats[split]["neg"]
        print(f"  {split}: {p+n} ({p} pos / {n} neg)")
    print(f"\ndataset at {DST}")
    print(f"data.yaml:\n{yaml}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
