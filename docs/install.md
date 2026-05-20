# Install + Configure + Run

End-to-end guide to install the soccer-video-pipeline on your own hardware,
get the models in place, run the detection chain, and produce keeper +
highlights reels.

For the high-level architecture of what you'll be running, see
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Hardware requirements

The pipeline assumes a 2-machine deployment but works on a single workstation
if you have enough VRAM:

| Role | What it runs | Minimum hardware |
|------|--------------|------------------|
| **Inference host** | vLLM (Qwen3-VL-32B-FP8 + LoRA, ~34 GB on disk) | 2× RTX 3090 / 4090 (48 GB VRAM total) over NVLink, or 1× A100/H100 (≥40 GB) |
| **Pipeline host** | YOLO + scripts + ffmpeg + scoring | Linux/macOS workstation, 1× consumer GPU helpful (YOLO inference), 16 GB RAM, 100 GB free disk |

Real-world: the reference deployment is one Linux box with 2× RTX 3090 for vLLM
plus a Mac Studio for pipeline orchestration, talking to the inference host
over a LAN.

Both roles can also run on a single beefier workstation (e.g., one
machine with 2× RTX 4090) — start vLLM in a tmux session and run the
pipeline scripts in another.

---

## 2. Operating system + system dependencies

The pipeline is developed on macOS + Linux. Windows works for the
pipeline-host role under WSL2 only.

### Inference host (vLLM)

Ubuntu 22.04+ or any recent Linux with NVIDIA drivers.

```bash
# NVIDIA driver + CUDA must already work — verify:
nvidia-smi

# Python 3.12
sudo apt-get install python3.12 python3.12-venv python3.12-dev

# Pinned vLLM 0.19.1 (newer versions silently break Qwen3-VL-32B-FP8)
python3.12 -m venv ~/vllm-venv
source ~/vllm-venv/bin/activate
pip install vllm==0.19.1 torch==2.10.0
```

> **CRITICAL**: vLLM 0.20.0+ produces garbage token output with the v11
> LoRA-merged FP8 model. Pin to 0.19.1. See
> `docs/feedback_vllm_pin_0_19_1.md` for the full incident report.

### Pipeline host (scripts + assembly)

macOS 14+ or Linux. Apple Silicon supported.

```bash
# Python 3.12 (with venv support)
brew install python@3.12              # macOS
# or:  sudo apt-get install python3.12 python3.12-venv  # Linux

# ffmpeg (required for frame extraction + reel assembly)
brew install ffmpeg                   # macOS
# or:  sudo apt-get install ffmpeg    # Linux

# Verify:
python3.12 --version    # 3.12.x
ffmpeg -version | head -1
```

---

## 3. Clone the repo + install Python deps

```bash
git clone https://github.com/acato/soccer-video-pipeline.git
cd soccer-video-pipeline

# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify install — should print "syntax OK"
python -c "import ast; ast.parse(open('scripts/build_reels.py').read()); print('syntax OK')"
```

The repo includes a `tests/` directory with 350+ unit tests. Run them as a
smoke test:

```bash
python -m pytest tests/unit/ -m unit -q
# Expected: a handful of pre-existing Windows-path failures; everything else passes
```

---

## 4. Download or build models

The pipeline needs four model artifacts. All are now published on
HuggingFace Hub.

### 4a. `Qwen/Qwen3-VL-32B-Instruct-FP8` (public base)

The base model for formation verification (stage 3b). Downloaded
automatically by vLLM on first launch, or pull explicitly:

```bash
hf download Qwen/Qwen3-VL-32B-Instruct-FP8 --local-dir ~/models/qwen3-vl-32b-fp8
```

### 4b. YOLOv8 soccer player detector

```bash
mkdir -p infra/models
# Either obtain from a soccer-fine-tuned YOLOv8 source like:
#   github.com/uisikdag/weed_soccer_models
# Or train your own. Place at:
ls infra/models/yolov8_soccer_uisikdag.pt
```

### 4c. **v11 LoRA-merged FP8** (Qwen3-VL-32B) — published

The custom dual-pass classifier. ~34 GB on disk. Published as
[acatorcini/qwen3-vl-32b-soccer-v11-fp8](https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8).

Pull explicitly (one-time, ~30 min on a fast link):

```bash
hf download acatorcini/qwen3-vl-32b-soccer-v11-fp8 \
  --local-dir ~/models/qwen3-vl-32b-soccer-v11-fp8
```

Or just reference it by HF ID in the vLLM serve command (vLLM downloads
on first launch — see §5).

**To train your own from scratch**: see `docs/fine-tuning-pipeline.md`.
The LoRA adapter alone (2.27 GB) is also published, useful as a starting
point for continued training:
[acatorcini/qwen3-vl-32b-soccer-v11-lora](https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-lora).

### 4d. YOLOv9 ball detector — published

Custom-trained YOLO ball detector for amateur-distance ball detection.
Published as [acatorcini/yolov9-soccer-ball](https://huggingface.co/acatorcini/yolov9-soccer-ball).
Both `.pt` (Ultralytics, AGPL-3.0) and `.onnx` (45 MB, ONNX format usable
with `onnxruntime` under Apache 2.0) formats are provided.

```bash
# Ultralytics path (AGPL-3.0 at runtime):
hf download acatorcini/yolov9-soccer-ball v9b_best.pt --local-dir infra/models/

# ONNX path (no AGPL runtime dependency):
hf download acatorcini/yolov9-soccer-ball v9b_best.onnx --local-dir infra/models/
```

If your project's licensing requires avoiding AGPL runtime dependencies,
use the ONNX path and load via `onnxruntime`. See the model card on HF
for inference snippets in both formats.

The path is configurable in `scripts/dense_yolo_filter.py` (`BALL_MODEL_DEFAULT`)
and at the top of any kickoff script that uses it.

---

## 5. Launch vLLM on the inference host

```bash
ssh inference-host
source ~/vllm-venv/bin/activate

# Stage A — v11 LoRA-merged FP8 (for the dual-pass detector)
vllm serve acatorcini/qwen3-vl-32b-soccer-v11-fp8 \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --served-model-name qwen3-vl-32b \
  --quantization compressed-tensors
```

> First launch downloads ~34 GB from HuggingFace Hub (cached to
> `~/.cache/huggingface/hub/`). Subsequent launches are instant.

Wait until you see `Uvicorn running on http://0.0.0.0:8000`. Verify:

```bash
curl http://inference-host:8000/v1/models
```

Should return a JSON object with `"id": "qwen3-vl-32b"`.

**Sanity check the LoRA didn't break**: send a tiny chat completion request.
If you see garbage tokens (e.g., `  =帛`), you're on the wrong vLLM version.
Downgrade to 0.19.1.

Later (after the dual-pass detector finishes a job) you'll restart vLLM
pointed at the **base** Qwen FP8 for formation verification:

```bash
# Stage B — base FP8 for formation verification (run AFTER stage A completes)
vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype auto \
  --served-model-name qwen3-vl-32b-base \
  --quantization fp8
```

> **Note**: stage A uses `--quantization compressed-tensors` (LoRA-merged
> format); stage B uses `--quantization fp8` (plain FP8). Mixing them will
> cause vLLM to reject the model. The reference deployment has a helper
> script at `/tmp/restore_v11.sh` to switch back to stage A.

---

## 6. Configure the pipeline host

Set environment variables to point at your inference host and storage paths:

```bash
# In your shell rc or a .env file sourced before running scripts:
export VLLM_URL=http://inference-host:8000/v1
export VLLM_MODEL=qwen3-vl-32b              # match --served-model-name
export WORKING_DIR=/tmp/soccer-pipeline
export SOURCE_VIDEOS_DIR=/path/to/your/match/mp4s
export REELS_OUTPUT_DIR=/path/to/output/reels
```

If you're using a shared mount for the videos + reels (e.g., NAS), set
`SOURCE_VIDEOS_DIR` and `REELS_OUTPUT_DIR` to the mount paths.

---

## 7. Process a video — step by step

The pipeline runs in five stages (see [ARCHITECTURE.md §1](ARCHITECTURE.md#1-end-to-end-picture)).
Each stage's outputs are cached in `/tmp/` so you can re-run later stages
without re-doing the expensive ones.

### Stage 1 — YOLO ball + player grounding

Roughly 10-15 minutes per match on a consumer GPU; longer on CPU.

```bash
# From the pipeline host, with .venv active:
.venv/bin/python scripts/detect_kickoffs.py \
  --video "$SOURCE_VIDEOS_DIR/my_game.mp4" \
  --output /tmp/kickoff_my_game_frames.jsonl \
  --sample-every 5.0
```

This writes per-frame YOLO detections to a JSONL. The 5-second sampling rate
is the production default — finer rates produce better ball-position data but
cost proportionally more time.

### Stage 2 — Dual-pass VLM event detection

The most expensive stage; depends on inference host throughput.

The legacy worker (`src/api/worker.py`) handles this via Celery. For
out-of-band runs, use:

```bash
# Submit to a running worker (if you have one):
.venv/bin/python infra/scripts/pipeline_cli.py submit "$SOURCE_VIDEOS_DIR/my_game.mp4" \
  --reel keeper,highlights

# Wait for completion, then events land at:
# /tmp/soccer-pipeline/<job_id>/events.jsonl
```

Without the Celery worker, you can invoke the detector directly (see
`src/detection/dual_pass_detector.py`).

Expect ~30-60 minutes per match against a 2× RTX 3090 inference host.

### Stage 3a — pattern_v11 kickoff detector

Fast (~2 minutes); runs against the cached YOLO frames from stage 1.

```bash
.venv/bin/python scripts/detect_kickoffs.py \
  --frames /tmp/kickoff_my_game_frames.jsonl \
  --emit-events /tmp/kickoff_my_game_pattern_v11_0191.jsonl
```

### Stage 3b — formation_base verifier (requires vLLM stage B)

After this point you need vLLM **switched to base FP8** (see §5).

```bash
.venv/bin/python scripts/generate_formation_candidates.py \
  --frames /tmp/kickoff_my_game_frames.jsonl \
  --output /tmp/formation_candidates_my_game.jsonl

.venv/bin/python scripts/verify_kickoffs_vlm_v3.py \
  --candidates /tmp/formation_candidates_my_game.jsonl \
  --video "$SOURCE_VIDEOS_DIR/my_game.mp4" \
  --vllm-url "$VLLM_URL" \
  --output /tmp/kickoff_my_game_formation_v2_base.jsonl
```

### Stage 4 — Merge + tier tagging

Cheap (~1 second). Fully driven by cached artifacts from earlier stages.

```bash
.venv/bin/python scripts/merge_ensemble_into_events.py \
  --dual-pass /tmp/soccer-pipeline/<job_id>/events.jsonl \
  --ensemble /tmp/kickoff_my_game_pattern_v11_0191.jsonl \
  --ensemble /tmp/kickoff_my_game_formation_v2_base.jsonl \
  --relaxed-aggregation \
  --negative-evidence \
  --save-tiers \
  --shot-tiers \
  --out /tmp/kickoff_my_game_tiered_events.jsonl
```

Output is the same `events.jsonl` schema but with `metadata.goal_tier`,
`metadata.save_tier`, and `metadata.shot_tier` added to relevant events.

### Stage 5 — Build the reels

Fast (~1-3 minutes per reel). `ffmpeg` stream-copies — no re-encode.

```bash
.venv/bin/python scripts/build_reels.py \
  --out-dir "$REELS_OUTPUT_DIR" \
  --games my_game
```

To run on multiple games at once, register them in `scripts/build_reels.py`
under the `GAMES` dict (source path + tiered events path + label) and pass
their keys via `--games`.

Output:

```
$REELS_OUTPUT_DIR/
  my_game_keeper.mp4       ~15-30 minutes of clips
  my_game_highlights.mp4   ~15-25 minutes of clips
```

---

## 8. Optional — score against ground truth

If you have GT data in the transit-format JSON schema (per-half files with
`event_name` ∈ {Saves, Shots & Goals, Goals Conceded, ...}), the scorers
will measure recall/precision per tier:

```bash
# Save scoring
.venv/bin/python scripts/score_saves_tiered.py
# Shot scoring
.venv/bin/python scripts/score_shots.py
# Goal scoring (writes to stdout)
.venv/bin/python scripts/score_tiered.py
```

You'll need to add your game(s) to the `GAMES` dict at the top of each
scorer (game key → GT directory + dual_pass events path + half offsets).

The offsets (`off_1h`, `off_2h`) map ground-truth game-clock to video time:
- `off_1h` is the video time at which the 1st half kickoff occurs
- `off_2h` is the halftime duration in seconds (added to GT timestamps in the 2H file)

Determine them by visual inspection or by using
`scripts/detect_kickoffs.py`'s wide-shot density anchor.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| vLLM emits garbage tokens (`  =帛`, random Chinese chars) | vLLM 0.20.0+ silently broke FP8 LoRA | Downgrade vLLM to 0.19.1 |
| vLLM rejects the model with "unrecognized quantization" | Wrong `--quantization` flag for the checkpoint | `compressed-tensors` for LoRA-merged FP8; `fp8` for plain FP8 base |
| Pipeline hangs at stage 2 | vLLM not reachable / wrong URL | `curl $VLLM_URL/models` should return JSON |
| Stage 3b returns no candidates | vLLM still on stage A (LoRA) — base needed | Restart vLLM with `--quantization fp8` and base path |
| Stage 5 reel has no clips | Tiered events file has no tier-tagged events | Re-run stage 4 with the `--save-tiers` / `--shot-tiers` flags |
| `ffprobe not found` warning from `build_reels.py` | ffmpeg installed but ffprobe missing from PATH | Most ffmpeg installs include ffprobe; check `which ffprobe`. The warning is non-fatal (duration clamping just skipped) |
| Reel includes lots of sideline throws | Throw-in exclusion not applied | Check that `KEEPER_REEL_EXCLUDE_TYPES` in `build_reels.py` includes `"throw_in"` |
| Goal recall < 1.00 on a new game | Half-time offset calibration off | Inspect wide-shot density transitions in the YOLO frames file to find the actual halftime span |

---

## 10. What the legacy API path does differently

`src/api/` + `src/api/worker.py` implement a Celery-based API for submitting
jobs and downloading reels. It exists for production deployments where a
non-technical user submits videos through a web UI.

The legacy path goes:
1. `POST /jobs` → Celery worker → dual_pass detector → events.jsonl
2. Events → `src/segmentation/clipper.py` → ClipBoundary list
3. ClipBoundary list → `src/assembly/composer.py` → reels

The scripts path described in §7 reuses step 1 then diverges. Long-term
plan is to fold the kickoff ensemble + tiering into the API worker so
that the legacy path produces the same quality reels.

For now, if you want the production-quality reels (recall 0.96 saves,
1.00 goals), use the scripts path.

---

## 11. Re-running just the cheap stages

The cost gradient across stages is steep:

| Stage | Time | Why |
|-------|-----:|-----|
| 1 (YOLO) | 10-15 min | Per-frame inference |
| 2 (dual-pass VLM) | 30-60 min | vLLM inference on long video |
| 3a (pattern_v11) | 2 min | Heuristic on cached YOLO |
| 3b (formation_base) | 10-15 min | vLLM inference on candidate windows only |
| 4 (merge + tier) | <1 s | Pure Python |
| 5 (reel assembly) | 1-3 min | ffmpeg stream-copy |

Stages 4 and 5 are essentially free, so once you've done stages 1-3 once,
iterating on tier rules or reel composition is a sub-minute loop. Many
of the choices in `merge_ensemble_into_events.py` (e.g., the throw-in
shot-proximity window) and `build_reels.py` (e.g., padding, merge gap,
type exclusions) were tuned via this fast inner loop.
