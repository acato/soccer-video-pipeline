import sys
print("python:", sys.version.split()[0])
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "vram_GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else None)
import transformers; print("transformers:", transformers.__version__)
import datasets; print("datasets:", datasets.__version__)
import llmcompressor; print("llmcompressor:", llmcompressor.__version__)
import compressed_tensors; print("compressed_tensors:", compressed_tensors.__version__)
import accelerate; print("accelerate:", accelerate.__version__)
import safetensors; print("safetensors:", safetensors.__version__)
