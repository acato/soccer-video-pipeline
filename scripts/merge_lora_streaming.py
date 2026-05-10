"""Streaming LoRA merge for Qwen3-VL-32B.
Reads base model shard-by-shard, merges adapter weights in place,
writes merged shards to OUT_DIR. Peak RAM ~10 GB."""
import json, time, shutil
from pathlib import Path
import torch
from safetensors.torch import safe_open, save_file

BASE = Path("/home/aless/.cache/modelscope/hub/models/Qwen/Qwen3-VL-32B-Instruct")
ADAPTER = Path("/mnt/transit/soccer-finetune/checkpoints/v5-32b/v1-20260427-221706/checkpoint-400")
OUT = Path("/mnt/transit/soccer-finetune/checkpoints/v5-32b/merged-bf16-c400")
OUT.mkdir(parents=True, exist_ok=True)

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
