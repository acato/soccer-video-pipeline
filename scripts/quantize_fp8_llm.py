"""FP8_DYNAMIC quant on the LLM server (Linux, RTX 3090).

Adapted from the original Criscato02/Windows version: paths Linux-native,
sequential offload tuned for 3090 (24 GB) instead of 3080 (12 GB).

Run with ~/quant-env/bin/python so vLLM-env stays untouched.
"""
import os, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

CHECKPOINT_TAG = sys.argv[1] if len(sys.argv) > 1 else "c757"
MODEL_GROUP = sys.argv[2] if len(sys.argv) > 2 else "v6-32b"

ROOT = Path(f"/mnt/transit/soccer-finetune/checkpoints/{MODEL_GROUP}")
MERGED = ROOT / f"merged-bf16-{CHECKPOINT_TAG}"
OUT = ROOT / f"fp8-{CHECKPOINT_TAG}"
OUT.mkdir(parents=True, exist_ok=True)

print(f"[start] {time.strftime('%FT%TZ')}")
print(f"[paths] merged={MERGED}  out={OUT}")
if not MERGED.exists():
    sys.exit(f"merged dir does not exist: {MERGED}")

print(f"[gpu] {torch.cuda.get_device_name(0)} free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

print("[model] loading with sequential offload...")
t0 = time.time()
offload_dir = OUT / "offload"
offload_dir.mkdir(exist_ok=True)
model = AutoModelForImageTextToText.from_pretrained(
    str(MERGED),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    max_memory={0: "20GiB", "cpu": "48GiB"},
    offload_folder=str(offload_dir),
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(str(MERGED), trust_remote_code=True)
print(f"[model] loaded in {time.time()-t0:.1f}s")

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head", "re:.*visual.*", "re:.*deepstack_merger.*"],
)
print(f"[recipe] FP8_DYNAMIC, ignore={recipe.ignore}")

print("[quant] running oneshot FP8_DYNAMIC...")
t0 = time.time()
oneshot(model=model, recipe=recipe, output_dir=str(OUT))
print(f"[quant] done in {(time.time()-t0)/60:.1f} min")

tokenizer.save_pretrained(str(OUT))
print(f"[done] FP8 model at {OUT}")
