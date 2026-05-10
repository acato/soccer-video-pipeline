import sys, os
sys.stdout.reconfigure(line_buffering=True)
print("py", sys.version.split()[0])
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
import transformers
print("transformers", transformers.__version__)
from llmcompressor import oneshot
print("oneshot at:", oneshot.__module__)
from llmcompressor.modifiers.quantization import QuantizationModifier
print("QuantizationModifier ok")
print("ALL OK")
os._exit(0)
