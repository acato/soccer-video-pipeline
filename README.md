# Soccer Video Pipeline

Automated event detection and reel assembly for amateur and youth soccer
matches. Drop in a sideline-camera recording, get back **keeper** and
**highlights** reels with measured 96–100% recall on goals + saves.

> **Intended use**: youth and amateur soccer analytics, personal
> experimentation, and research. Not affiliated with, endorsed by, or
> approved by any professional league, federation, or governing body.

## What it does

Given a full-match MP4, the pipeline runs a chain of:

- Per-frame YOLO grounding (player + ball positions)
- Dual-pass VLM event classification (Qwen3-VL-32B with a custom LoRA)
- A kickoff-pattern ensemble that recovers goals the VLM misses
- Three-tier event tagging (`goal_tier`, `save_tier`, `shot_tier`)
- Clip extraction + ffmpeg stream-copy concatenation

…and produces two MP4s per game.

Measured on a 4-game evaluation set with full event-stream ground truth:

| Event type | Union recall | Union precision |
|------------|-------------:|----------------:|
| Goals      | **1.00**     | 0.17            |
| Saves      | **0.96**     | 0.20            |
| Shots      | **0.85**     | 0.55            |

See [docs/pipeline_status.md](docs/pipeline_status.md) for the full
breakdown.

## Hardware

The reference deployment uses two machines:

- **Inference host** — 2× RTX 3090/4090 over NVLink (or one A100/H100), runs vLLM
- **Pipeline host** — Mac or Linux workstation, runs YOLO + scripts + ffmpeg

Single-workstation deployments work if you have ≥48 GB VRAM.

## Quick links

- **[docs/install.md](docs/install.md)** — install, configure, and run end-to-end
- **[docs/architecture.md](docs/architecture.md)** — system design, model dependencies, tier semantics, key decisions
- **[docs/pipeline_status.md](docs/pipeline_status.md)** — current detection performance + known limits
- **[docs/fine-tuning-pipeline.md](docs/fine-tuning-pipeline.md)** — train your own LoRA
- **[docs/vllm-gpu-orchestration.md](docs/vllm-gpu-orchestration.md)** — vLLM deployment on multi-GPU hosts

## Models

All required models are published on HuggingFace Hub under
[`acatorcini`](https://huggingface.co/acatorcini):

| Model | Size | Purpose |
|-------|----:|---------|
| [qwen3-vl-32b-soccer-v11-fp8](https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8) | 34 GB | Primary dual-pass event classifier (LoRA-merged FP8) |
| [qwen3-vl-32b-soccer-v11-lora](https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-lora) | 2.27 GB | LoRA adapter alone, for custom merges or continued training |
| [yolov9-soccer-ball](https://huggingface.co/acatorcini/yolov9-soccer-ball) | 22 MB / 45 MB | Custom ball detector (Ultralytics + ONNX formats) |

Plus the public base model: [`Qwen/Qwen3-VL-32B-Instruct-FP8`](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for
third-party attributions.

The YOLO weights inherit AGPL-3.0 from the Ultralytics training framework
that produced them. An ONNX format is also published so you can run
inference under `onnxruntime` (Apache 2.0) without bundling Ultralytics
at runtime. See [docs/install.md](docs/install.md) §4d.

The Qwen3-VL-based models inherit the Tongyi Qianwen License from the
base model — see the model cards on HuggingFace for the authoritative terms.

## Disclaimer

This project does not grant rights to process or redistribute copyrighted
video content. Users are responsible for compliance with applicable rights
and league policies.
