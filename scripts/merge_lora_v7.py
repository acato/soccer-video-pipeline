"""Streaming LoRA merge for Qwen3-VL-32B v7 LoRA into the merged-bf16-c757
base it was trained on top of.

v7 was trained against merged-bf16-c757 (which already has v6 c757 LoRA
baked in). So merging v7's LoRA on top of merged-bf16-c757 produces a base
that has BOTH v6 + v7 contributions, ready for FP8 quant.

Usage:
    python merge_lora_v7.py [CHECKPOINT_TAG]   # e.g. c250 (default)

Reads the v7 training run dir under /mnt/transit/.../v7-32b/v0-*/
"""
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

CHECKPOINT_TAG = sys.argv[1] if len(sys.argv) > 1 else "c250"

V7_ROOT = Path("/mnt/transit/soccer-finetune/checkpoints/v7-32b")
BASE = Path("/mnt/transit/soccer-finetune/checkpoints/v6-32b/merged-bf16-c757")
OUT = V7_ROOT / f"merged-bf16-{CHECKPOINT_TAG}"
OUT.mkdir(parents=True, exist_ok=True)

# Locate the v7 training run dir (v0-YYYYMMDD-HHMMSS) and the checkpoint
run_dirs = sorted(V7_ROOT.glob("v0-*"))
if not run_dirs:
    sys.exit(f"no v7 training run dirs under {V7_ROOT}")
RUN = run_dirs[-1]  # latest
step = int(CHECKPOINT_TAG.lstrip("c"))
ADAPTER = RUN / f"checkpoint-{step}"
if not ADAPTER.exists():
    sys.exit(f"adapter not found: {ADAPTER}\navailable: {sorted(p.name for p in RUN.iterdir() if p.name.startswith('checkpoint-'))}")

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
for k in adapter:
    if not k.endswith(".lora_A.weight"):
        continue
    base_key = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
    b_key = k.replace(".lora_A.weight", ".lora_B.weight")
    if b_key not in adapter:
        raise RuntimeError(f"missing lora_B for {k}")
    deltas[base_key] = (adapter[k], adapter[b_key])
print(f"[map] {len(deltas)} base weights will receive merge")

# Copy non-safetensors metadata files (tokenizer, configs) from base.
for f in BASE.iterdir():
    if f.suffix in {".json", ".txt"} and "safetensors" not in f.name:
        shutil.copy(f, OUT / f.name)
        print(f"[copy] {f.name}")

index = json.loads((BASE / "model.safetensors.index.json").read_text())
shard_files = sorted(set(index["weight_map"].values()))
print(f"[shards] {len(shard_files)} to process")

merged_keys, total_keys = 0, 0
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
                merged_keys += 1
            tensors[k] = t.contiguous()
    save_file(tensors, op, metadata={"format": "pt"})
    del tensors
    print(f"  -> wrote in {time.time()-t0:.1f}s, merged so far {merged_keys}/{len(deltas)}")

shutil.copy(BASE / "model.safetensors.index.json", OUT / "model.safetensors.index.json")
print(f"[done] merged {merged_keys}/{total_keys} weights into {OUT}")
