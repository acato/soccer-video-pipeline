"""FP8 quantization for QL3 merged model on Criscato02 (RTX 3080, 12 GB).
Uses llmcompressor with sequential CPU offload to fit a 32B bf16 model
on a 12 GB GPU. Calibration via tiny WikiText sample (sufficient for FP8
W8A8 dynamic — calibration only needs activation range estimates).

Output is a vLLM-compatible FP8 checkpoint."""
import os, time, json, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForImageTextToText
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MERGED = Path(r"C:\quant\merged-bf16-c400")
OUT = Path(r"C:\quant\fp8-c400")
OUT.mkdir(parents=True, exist_ok=True)

print(f"[start] {time.strftime('%FT%TZ')}")
print(f"[paths] merged={MERGED}  out={OUT}")
print(f"[gpu] {torch.cuda.get_device_name(0)} free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

# Load model with sequential offload — keeps only one block on GPU at a time.
print("[model] loading with sequential offload (this takes ~3-5 min)...")
t0 = time.time()
offload_dir = OUT / "offload"
offload_dir.mkdir(exist_ok=True)
model = AutoModelForImageTextToText.from_pretrained(
    str(MERGED),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    max_memory={0: "10GiB", "cpu": "26GiB"},
    offload_folder=str(offload_dir),
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(str(MERGED), trust_remote_code=True)
print(f"[model] loaded in {time.time()-t0:.1f}s")

# Calibration data: FP8_DYNAMIC needs no calibration data (per-tensor activation
# scales are computed at inference). We pass a tiny placeholder dataset just to
# satisfy llmcompressor's API.
print("[cal] using placeholder dataset (FP8_DYNAMIC needs no calibration)")
calib_texts = ["The quick brown fox jumps over the lazy dog."] * 8
ds = [tokenizer(t, return_tensors="pt", truncation=True, max_length=64) for t in calib_texts]

# FP8 W8A8 dynamic — vLLM-compatible.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head", "re:.*visual.*", "re:.*deepstack_merger.*"],
)
print(f"[recipe] FP8_DYNAMIC, ignore={recipe.ignore}")

print("[quant] running oneshot weight-only FP8 quant (~30-90 min on 3080)...")
t0 = time.time()
oneshot(
    model=model,
    recipe=recipe,
    output_dir=str(OUT),
)
print(f"[quant] done in {(time.time()-t0)/60:.1f} min")

# Save tokenizer alongside
tokenizer.save_pretrained(str(OUT))
print(f"[save] tokenizer saved to {OUT}")
print(f"[done] FP8 model at {OUT}")
