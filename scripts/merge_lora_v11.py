"""Streaming LoRA merge for v11 LoRA into the v8 merged-bf16-c300 base.

v10 was trained against merged-bf16-c300 (which has v6+v8 contributions
baked in already), with ViT unfrozen — so v10's adapter contains both
language LoRA deltas AND ViT delta weights. Same handling as v8 merger.

Usage:
    python merge_lora_v11.py [CHECKPOINT_TAG]   # e.g. c300 (default)
"""
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

CHECKPOINT_TAG = sys.argv[1] if len(sys.argv) > 1 else "c300"

V11_ROOT = Path("/mnt/transit/soccer-finetune/checkpoints/v11-32b")
BASE = Path("/mnt/transit/soccer-finetune/checkpoints/v8-32b/merged-bf16-c300")
OUT = V11_ROOT / f"merged-bf16-{CHECKPOINT_TAG}"
OUT.mkdir(parents=True, exist_ok=True)

run_dirs = sorted(V11_ROOT.glob("v0-*"))
if not run_dirs:
    sys.exit(f"no v11 training run dirs under {V11_ROOT}")
RUN = run_dirs[-1]
step = int(CHECKPOINT_TAG.lstrip("c"))
ADAPTER = RUN / f"checkpoint-{step}"
if not ADAPTER.exists():
    sys.exit(f"adapter not found: {ADAPTER}")

print(f"[paths]")
print(f"  base    = {BASE}")
print(f"  adapter = {ADAPTER}")
print(f"  out     = {OUT}")

ADAPTER_CFG = json.loads((ADAPTER / "adapter_config.json").read_text())
RANK = ADAPTER_CFG["r"]
ALPHA = ADAPTER_CFG["lora_alpha"]
SCALING = ALPHA / RANK
print(f"[cfg] r={RANK} alpha={ALPHA} scaling={SCALING}")

print("[adapter] loading lora tensors into RAM...")
t0 = time.time()
adapter = {}
with safe_open(ADAPTER / "adapter_model.safetensors", "pt") as f:
    for k in f.keys():
        adapter[k] = f.get_tensor(k)
print(f"[adapter] {len(adapter)} tensors loaded in {time.time()-t0:.1f}s")

deltas = {}
direct_weights = {}
for k in adapter:
    if k.endswith(".lora_A.weight"):
        base_key = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
        b_key = k.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in adapter:
            raise RuntimeError(f"missing lora_B for {k}")
        deltas[base_key] = (adapter[k], adapter[b_key])
    elif k.endswith(".lora_B.weight"):
        continue
    else:
        base_key = k.replace("base_model.model.", "")
        direct_weights[base_key] = adapter[k]

print(f"[map] {len(deltas)} language LoRA deltas, {len(direct_weights)} ViT direct weights")

for f in BASE.iterdir():
    if f.suffix in {".json", ".txt"} and "safetensors" not in f.name:
        shutil.copy(f, OUT / f.name)
        print(f"[copy] {f.name}")

index = json.loads((BASE / "model.safetensors.index.json").read_text())
shard_files = sorted(set(index["weight_map"].values()))
print(f"[shards] {len(shard_files)} to process")

merged_lora, merged_direct, total_keys = 0, 0, 0
for si, shard in enumerate(shard_files, 1):
    sp = BASE / shard
    op = OUT / shard
    print(f"[shard {si}/{len(shard_files)}] {shard} ({sp.stat().st_size/1e9:.2f} GB)")
    t0 = time.time()
    tensors = {}
    with safe_open(sp, "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            total_keys += 1
            if k in deltas:
                lA, lB = deltas[k]
                delta = (lB.float() @ lA.float()) * SCALING
                t = (t.float() + delta).to(t.dtype)
                merged_lora += 1
            elif k in direct_weights:
                t = direct_weights[k].to(t.dtype)
                merged_direct += 1
            tensors[k] = t.contiguous()
    save_file(tensors, op, metadata={"format": "pt"})
    del tensors
    print(f"  -> wrote in {time.time()-t0:.1f}s, "
          f"lora_so_far={merged_lora}/{len(deltas)} "
          f"direct_so_far={merged_direct}/{len(direct_weights)}")

shutil.copy(BASE / "model.safetensors.index.json", OUT / "model.safetensors.index.json")
print(f"[done] merged {merged_lora}+{merged_direct} of {total_keys} weights into {OUT}")
