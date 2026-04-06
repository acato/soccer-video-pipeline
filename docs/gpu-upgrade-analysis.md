# GPU Hardware Upgrade Analysis: Impact on Soccer Video Pipeline Accuracy

**Date**: 2026-04-03
**Author**: Pipeline architecture analysis
**Status**: Research document -- no code changes

---

## Executive Summary

The pipeline's current VLM bottleneck is **not model size** -- it is the fundamental limitation of sideline camera distance (~50 metres). Upgrading from Qwen3-VL-32B-FP8 to a 72B+ model will produce **marginal accuracy gains (estimated 3-8% overall)** because the hard problems (catch detection, throw-in detection, false-positive kickoffs) are caused by the camera's inability to resolve goalkeeper hand contact and small-body-pose details at distance, not by the model's reasoning capacity.

The highest-ROI investments, in order, are:

1. **Fine-tuning Qwen3-VL-8B on soccer domain data** (cloud GPU, ~$50-150) -- expected 30-50% classification accuracy lift based on published results
2. **Claude Sonnet 4.6 Batch API as drop-in upgrade** (~$2-4/match) -- better spatial reasoning, zero infrastructure
3. **Running current Qwen3-VL-32B at FP16** (single A100 80GB, ~$1/hr) -- eliminates FP8 edge-case rounding, minor gains
4. **Scaling to Qwen3-VL-235B-A22B** (8x A100, ~$8-16/hr) -- large MoE model, diminishing returns for our task

The structural inference pipeline (Phases 3a-3g) already compensates for VLM weaknesses on the hardest categories. Further accuracy improvements should focus on better candidate generation and structural rules, not bigger models.

---

## 1. Current Setup Baseline

### Hardware
- 2x NVIDIA RTX 3090 (24 GB each, 48 GB total via tensor-parallel)
- Model: `Qwen/Qwen3-VL-32B-Instruct-FP8`
- vLLM with `--max-model-len 31488 --kv-cache-dtype fp8 --tensor-parallel-size 2 --max-num-seqs 4 --gpu-memory-utilization 0.95`

### Performance Characteristics
- Inference: 3-8 seconds per VLM call (single image-laden request)
- Throughput: ~120 initial + ~100 follow-up VLM calls per match (~220 total)
- Total VLM time: ~15-25 minutes of a ~96-minute pipeline run
- Frame extraction: 2 FPS, 768px width, JPEG quality 5, max 24 frames per call
- Token budget: ~500 max output tokens, temperature 0

### Current Accuracy (Rush vs Reign ground truth, Apr 2 v4)

| Event Type | Detected | Ground Truth | Recall | Notes |
|-----------|----------|-------------|--------|-------|
| Goals | 5 | 4 | 125% (1 FP) | Kickoff inference works; 1 persistent FP |
| Shots on target | 24 | 16 | 150% | Over-triggered (some off-target miscategorised) |
| Shots off target | 13 | 25 | 52% | Under-detected |
| Saves (all) | 15 | 17 | 88% | 9 parry + 1 VLM catch + 5 structural catch |
| Corners | 9 | 9 | 100% | Perfect |
| Goal kicks | 20 | 23 | 87% | Good |
| Throw-ins | 6 | 48 | 12% | Major gap -- motion scan misses most |
| **Total shots** | **42** | **45** | **93%** | |

### Root Cause Analysis of Failures

Understanding *why* events are missed is critical for evaluating whether hardware upgrades help:

| Failure Mode | Root Cause | Would Bigger VLM Help? |
|-------------|-----------|----------------------|
| Missed catches (2 of 17) | No motion spike for GK holding ball; structural inference catches most | No -- detection gap, not classification |
| Missed throw-ins (42 of 48) | Throw-ins produce minimal motion delta; sideline camera shows small figures | Marginal -- pose detection at 50m is the limit |
| Shot on/off split errors | VLM cannot see ball trajectory at 50m | No -- spatial resolution problem |
| False positive goals | VLM sees players near center circle during normal play and confirms "kickoff" | Possibly -- better reasoning might reduce FPs |
| False positive kickoffs | VLM cannot distinguish center-circle clustering from normal midfield play | Possibly -- better scene understanding |
| Missed goalkeeper saves | Cannot see GK hand contact at 50m; relies on structural inference | No -- physical resolution limit |

**Key insight**: Of the 5 major failure modes, only 2 (false-positive kickoffs/goals) have any chance of improvement from a more capable model. The others are camera resolution or candidate generation problems.

---

## 2. Model Size vs. Accuracy Tradeoffs

### 2.1 Qwen3-VL Model Landscape (as of April 2026)

The Qwen3-VL family does **not include a 72B dense model**. The available sizes are:

| Model | Type | Parameters | Active Params | VRAM (FP16) | VRAM (FP8) |
|-------|------|-----------|---------------|-------------|------------|
| Qwen3-VL-2B | Dense | 2B | 2B | ~5 GB | ~3 GB |
| Qwen3-VL-4B | Dense | 4B | 4B | ~9 GB | ~5 GB |
| Qwen3-VL-8B | Dense | 8B | 8B | ~17 GB | ~10 GB |
| Qwen3-VL-32B | Dense | 32B | 32B | ~65 GB | ~35 GB |
| Qwen3-VL-30B-A3B | MoE | 30B | 3B | ~62 GB | ~34 GB |
| Qwen3-VL-235B-A22B | MoE | 235B | 22B | ~470 GB | ~270 GB |

There is no Qwen3-VL-72B. The previous generation Qwen2.5-VL-72B exists, but Qwen3-VL-32B **outperforms it on 14 of 15 benchmarks** despite being 2.4x smaller, due to architecture improvements.

### 2.2 Benchmark Comparisons Relevant to Our Task

| Benchmark | Qwen3-VL-32B | Qwen2.5-VL-72B | Delta | Relevance to Soccer |
|-----------|-------------|----------------|-------|-------------------|
| VideoMME (w/o sub) | 77.3 | 73.3 | +4.0 | High -- video clip understanding |
| MVBench | 72.8 | 70.4 | +2.4 | High -- motion/action in video |
| LVBench | 63.8 | 47.3 | +16.5 | Medium -- long video comprehension |
| MMMU-Pro | 65.3 | 51.1 | +14.2 | Low -- academic multi-modal reasoning |
| AI2D | 89.5 | 88.4 | +1.1 | Low -- diagram understanding |
| OCRBench | 89.5 | 88.5 | +1.0 | Low -- text recognition |

**Conclusion**: You are already running the strongest Qwen vision model available in a practical dense architecture. Moving to Qwen2.5-VL-72B would be a **downgrade** on video understanding benchmarks.

### 2.3 Qwen3-VL-235B-A22B (Flagship MoE)

The 235B MoE model is the only model in the Qwen3-VL family that clearly exceeds the 32B on all benchmarks:

- Requires **8x A100 80GB minimum** (640 GB total VRAM)
- FP8 weights alone consume ~270 GB; runtime with KV cache needs ~450+ GB
- Excels at visual math and document tasks; advantage on video tasks is more modest
- In "needle-in-a-haystack" video tests: 100% accuracy in 30-minute videos, 99.5% in 2-hour videos
- Cloud cost: 8x A100 at ~$1/hr each = **$8/hour minimum**

**Expected accuracy improvement for our task**: 3-5% overall classification accuracy. The model's advantage is in long-context reasoning and document analysis, not in resolving 50m-distance spatial details.

### 2.4 Claude Sonnet 4.6 / Opus 4.6

Claude models process images natively (no video tokens -- frame-by-frame only, which is what our pipeline already does).

| Aspect | Claude Sonnet 4.6 | Claude Opus 4.6 |
|--------|-------------------|-----------------|
| Input pricing | $3/MTok | $5/MTok |
| Output pricing | $15/MTok | $25/MTok |
| Batch pricing (input) | $1.50/MTok | $2.50/MTok |
| MMMU | ~74.4% | ~76.5% |
| SPORTU-video (sports) | ~69.5% (Claude 3.5 Sonnet) | N/A published |
| Max images per request | 20 | 20 |
| Latency per call | 2-5s | 3-8s |

**Cost estimate per match** (220 calls x ~20 frames x ~200 tokens/frame + prompt):
- Per call: ~4,500 input tokens (prompt + images) + ~100 output tokens
- Total input: 220 x 4,500 = ~990K tokens
- Total output: 220 x 100 = ~22K tokens
- **Sonnet 4.6 cost**: (990K x $3 + 22K x $15) / 1M = **$3.30/match**
- **Sonnet 4.6 Batch**: (990K x $1.50 + 22K x $7.50) / 1M = **$1.65/match**
- **Opus 4.6 cost**: (990K x $5 + 22K x $25) / 1M = **$5.50/match**

Note: Image tokens in Claude are significantly more than 200 tokens per 768px frame. A more realistic estimate is ~1,600 tokens per image (Anthropic uses ~1,600 tokens for a 768x768 image). With 10-20 images per call:

- Per call: ~500 (prompt) + 15 images x 1,600 = ~24,500 input tokens
- Total input: 220 x 24,500 = ~5.4M tokens
- **Sonnet 4.6 cost**: (5.4M x $3 + 22K x $15) / 1M = **$16.50/match**
- **Sonnet 4.6 Batch**: (5.4M x $1.50 + 22K x $7.50) / 1M = **$8.25/match**
- **Opus 4.6 cost**: (5.4M x $5 + 22K x $25) / 1M = **$27.55/match**

Claude's advantage: superior spatial reasoning for complex scenes. The SPORTU benchmark shows Claude 3.5 Sonnet at 69.5% on sports video QA vs GPT-4o at 68.8%, and both significantly outperform open-source models on hard tasks. The newer Claude 4.x models should improve on this.

**Expected accuracy improvement**: 5-10% on multi-class classification, with the biggest gains on false-positive reduction (better reasoning about "is this really a kickoff?"). Claude is unlikely to solve the catch detection or throw-in detection problems because those are spatial resolution issues.

### 2.5 GPT-4o / GPT-4V

| Aspect | GPT-4o |
|--------|--------|
| SPORTU-text accuracy | 71% |
| SPORTU-video accuracy | ~68.8% |
| Hard task accuracy | 57.8% (struggles on scenario reasoning) |
| Pricing | ~$2.50/MTok input, ~$10/MTok output |
| Video input | Native video tokens (not frame-by-frame) |

GPT-4o supports native video input, which could theoretically capture temporal dynamics better than frame-by-frame. However, the SPORTU benchmark shows it **struggles on hard sports tasks** (57.8% vs 70%+ for easy tasks). For our pipeline's hard problems (fine-grained action classification at distance), GPT-4o does not demonstrate a clear advantage over Claude or Qwen3-VL-32B.

**Cost**: Similar to Claude Sonnet 4.6 -- roughly $10-20/match with images.

### 2.6 InternVL3-78B

InternVL3-78B combines a 6B vision transformer with Qwen2.5-72B as the language backbone:

| Aspect | InternVL3-78B |
|--------|--------------|
| VRAM | ~160 GB FP16 (2x A100 80GB) |
| OCRBench | 906 |
| MMT-Bench | 73.2 |
| Video understanding | Competitive with GPT-4o on general video QA |

InternVL3-78B is a strong general multimodal model but has **no demonstrated advantage on sports video** specifically. Its 6B vision encoder processes images at lower resolution than Qwen3-VL's ViT, which could actually hurt our pipeline where spatial detail matters.

### 2.7 Sports-Specific Fine-Tuned VLMs

The most relevant published work is **Jiang et al. (CVPR 2025 Workshop)**:

- Fine-tuned LLaVA-NeXT-Video on 20K curated soccer clips
- **Classification accuracy went from 11.8% to 63.5%** (SoccerNet Ball Action Spotting classes)
- Training used curriculum learning: concept understanding first, then action classification
- Synthetic labels generated by Claude 3.5 Sonnet
- Training cost: feasible on a single A100 for ~24 hours

This is by far the highest-ROI published result. A 5.4x accuracy improvement from domain adaptation vastly exceeds anything achievable by scaling model size alone. See Section 4 for how to apply this to our pipeline.

---

## 3. What Better Hardware Enables

### 3.1 FP16 Instead of FP8

**Current state**: Qwen3-VL-32B-FP8 on 2x RTX 3090 (48 GB).
**FP16 requirement**: ~65 GB VRAM -- needs 1x A100 80GB or 2x A100 40GB.

Published benchmarks consistently show FP8 quantization retains **99-100% of FP16 accuracy** for large language models. The accuracy difference is negligible -- typically <0.5% on standard benchmarks. For our pipeline, where the bottleneck is spatial resolution, not model precision, the expected improvement is **near zero**.

However, FP16 provides one indirect benefit: **numerical stability in edge cases**. FP8's reduced mantissa (3-4 bits) can cause slight differences in confidence scores, which occasionally push a borderline event above or below the 0.5 confidence threshold. This could affect 1-3 events per match.

**Verdict**: Not worth the infrastructure change solely for FP16. If you move to cloud A100 for other reasons, run FP16 as a free bonus.

### 3.2 Higher Batch Sizes / Concurrent Inference

**Current**: `max-num-seqs 4` on 2x RTX 3090. Each call is sequential in the pipeline.

With an A100 80GB or H100 80GB:
- `max-num-seqs` can increase to 8-16 for Qwen3-VL-32B
- Concurrent requests enable **parallel VLM calls** in pipeline phases that are independent

**Throughput impact**: Pipeline VLM time could drop from ~20 minutes to ~8-10 minutes per match. Total pipeline time (currently ~96 min) drops by ~10-12 minutes.

**Accuracy impact**: None directly. Faster inference enables more VLM calls per match without extending runtime, which could allow:
- Increasing the VLM candidate cap from 120 to 200+
- Running two-pass classification (observe then classify) without doubling time
- Adding more spot-check probes in temporal gaps

### 3.3 Larger Context Windows (More Frames Per Call)

**Current**: 24 frames max (12s at 2 FPS), `max-model-len 31488` tokens.

Each 768px JPEG frame in Qwen3-VL generates approximately 200-400 visual tokens (after 2x2 patch merging). With 24 frames:
- Visual tokens: ~4,800-9,600
- Prompt tokens: ~500
- Total input: ~5,300-10,100 tokens
- Safely within the 31K limit

**What more context buys**: With an A100 running `max-model-len 65536+`:
- Could send 48 frames (24s at 2 FPS) or more
- The additional temporal context would help for:
  - **Goal detection**: See both the shot and the celebration/kickoff in one call
  - **Throw-in detection**: Wider window catches the throw-in pose
  - **Context after saves**: See the GK hold the ball and distribute

**Expected improvement**: Marginal. The current 12s window already captures most event-relevant frames. The 5s pre / 15s post clip window sends 20s of content, subsampled to 24 frames. Expanding to 30+ frames per call would add ~200 tokens/frame of cost and ~1-2s latency per call.

Published research (Qwen2.5-VL Technical Report, arxiv 2502.13923) shows continued improvement up to 256 frames but with diminishing returns. For our 2-second events, 24 frames is already sufficient. The bottleneck is spatial resolution, not temporal coverage.

### 3.4 Video-Native Models (Video Tokens vs Frame-by-Frame)

Qwen3-VL supports native video input (not just frame-by-frame images). Video tokens encode temporal relationships between frames, potentially capturing motion patterns that individual frames miss.

**What this could help with**:
- Ball trajectory (direction of travel between frames)
- Goalkeeper dive dynamics (lateral movement over 0.5-1s)
- Player movement patterns (running vs standing)

**What it cannot help with**:
- Spatial resolution at 50m (same camera, same pixels)
- Small object detection (ball is <20 pixels regardless of temporal encoding)

**Infrastructure requirement**: vLLM already supports Qwen3-VL video input via the OpenAI-compatible API. The pipeline would need to send video clips instead of extracted frames. This is a code change, not a hardware change.

**Expected improvement**: 2-5% on motion-dependent classifications (saves, shots), negligible on static events (corners, goal kicks).

---

## 4. Specific Accuracy Improvements Expected

### 4.1 Goal Detection (Currently 125% -- 1 FP)

The current goal detection already works well via structural inference (Phase 3b: kickoff rescan + celebration probe). The single false positive is a case where a coincidental kickoff-like formation occurs after a shot.

| Upgrade | Expected Impact |
|---------|----------------|
| Qwen3-VL-235B | Might reduce FP kickoff by better scene understanding. 50/50. |
| Claude Sonnet 4.6 | Better reasoning about "is this really a kickoff?" -- likely eliminates the FP. |
| Fine-tuned 8B | Trained on kickoff examples could reliably distinguish real vs false kickoffs. |
| Better hardware (same model) | No impact -- this is a reasoning quality issue. |

**Best approach**: The celebration probe already catches most FPs. The remaining FP is best addressed by prompt engineering or a second-opinion probe, not hardware.

### 4.2 Throw-In Detection (Currently 12% -- 6 of 48)

This is the pipeline's worst category. The failure cascade:
1. Motion scan misses most throw-ins (minimal frame-differencing delta)
2. Audio cannot reliably detect throw-ins (no whistle, no crowd reaction)
3. Even when a candidate exists, VLM struggles to identify the throw-in pose at 50m

| Upgrade | Expected Impact |
|---------|----------------|
| Qwen3-VL-235B | +2-3% (slightly better pose recognition at distance) |
| Claude Sonnet/Opus | +3-5% (better spatial reasoning about player pose) |
| Fine-tuned 8B | +10-15% IF training data includes sideline throw-in examples |
| Better candidate generation | +30-50% (the real bottleneck is motion scan missing throw-ins) |

**Best approach**: Throw-in detection requires solving the **candidate generation** problem first. Options:
- Lower motion threshold specifically for throw-in-likely moments (ball near sideline in prior events)
- After every event that goes "out of play", probe +5/+10/+15s for a throw-in
- Use YOLO ball detection to identify ball-near-sideline moments
None of these require better GPU hardware.

### 4.3 Save Type Discrimination at Distance

Saves are currently detected at 88% (15/17) via structural inference:
- 9 parry saves (shot + corner = parry)
- 5 structural catches (no restart within 60s = catch)
- 1 VLM-detected catch (GK visually holding ball)
- 2 missed saves (candidate generation gaps)

| Upgrade | Expected Impact |
|---------|----------------|
| Qwen3-VL-235B | No improvement -- VLM catch probe was 0/9 even with 32B |
| Claude Sonnet/Opus | Slightly better -- but 0/9 catch probes suggests the visual signal is absent |
| Fine-tuned model | If trained on sideline-distance catches, could learn the subtle visual pattern |
| Better candidate generation | Addresses the 2 completely missed saves |

**Key insight**: The VLM catch probe (Phase 3g) sent 9 probes and got 0 confirmations. This is a **spatial resolution failure**, not a model capability failure. At 50m, the goalkeeper holding the ball vs the ball on the ground is a ~5 pixel difference. No VLM will reliably resolve this without better input images.

### 4.4 False-Positive Kickoffs

The VLM sometimes confirms "kickoff" when players cluster near midfield during normal play.

| Upgrade | Expected Impact |
|---------|----------------|
| Qwen3-VL-235B | Moderate improvement -- better reasoning about game state |
| Claude Sonnet/Opus | Good improvement -- Claude's spatial reasoning is strong |
| Fine-tuned model | Best improvement if trained on kickoff vs midfield-play discrimination |
| Structural rules | Already partially addressed by celebration probe requirement |

**Best approach**: Add a temporal constraint -- real post-goal kickoffs take 30-90s after the shot. If a "kickoff" is detected <20s after a shot, it is more likely midfield play.

### 4.5 Overall Event Classification Accuracy

Combining all categories, current overall recall is approximately:

**Weighted overall**: (5+42+15+9+20+6) / (4+45+17+9+23+48) = 97/146 = **66% overall recall**

| Upgrade Path | Expected Overall Recall | Lift | Cost |
|-------------|----------------------|------|------|
| Current (Qwen3-VL-32B-FP8) | 66% | baseline | $0 (self-hosted) |
| Qwen3-VL-32B-FP16 (A100) | 67-68% | +1-2% | ~$1/hr cloud |
| Qwen3-VL-235B (8x A100) | 69-71% | +3-5% | ~$8-16/hr cloud |
| Claude Sonnet 4.6 API | 70-73% | +4-7% | ~$8-17/match |
| Claude Opus 4.6 API | 71-74% | +5-8% | ~$28/match |
| Fine-tuned Qwen3-VL-8B | 75-80% | +9-14% | ~$50-150 training, then $0 self-hosted |
| Fine-tuned 8B + structural | 80-85% | +14-19% | Same |
| Improved candidate generation | 75-80% | +9-14% | Code changes only |

---

## 5. Alternative Approaches Enabled by Cloud GPUs

### 5.1 Fine-Tuning Qwen3-VL-8B on Soccer Data

This is the single highest-ROI investment available.

**Published evidence**: Jiang et al. (CVPR 2025 Workshop) showed that domain-adapting LLaVA-NeXT-Video with 20K curated soccer clips improved classification from 11.8% to 63.5% -- a 5.4x improvement. Our pipeline already achieves higher baseline accuracy (66%) thanks to structural inference, but the principle applies: a small model fine-tuned on domain data beats a large general model.

**Implementation plan**:
1. Generate training data from our existing pipeline runs (verdicts + ground truth corrections)
2. Add SoccerNet Ball Action Spotting clips (publicly available, 12 event classes)
3. Curate ~5,000-20,000 labeled clips from sideline camera footage
4. LoRA fine-tune Qwen3-VL-8B-Instruct on the curated data
5. Deploy the fine-tuned 8B model on the existing 2x RTX 3090 (fits easily)

**Hardware for training**: Single A100 40GB or 80GB, approximately 4-12 hours with LoRA.
**Cost**: ~$50-150 in cloud GPU time for training.
**Inference**: The fine-tuned 8B model runs on a single RTX 3090 at ~2x the speed of the current 32B, with domain-specific accuracy that could exceed the 32B on soccer events.

**Risk**: Requires curated training data. If the training set is too small or unrepresentative of sideline camera conditions, accuracy gains will be smaller. The 20K clip dataset from Jiang et al. was built from broadcast footage, not sideline -- transfer may require adaptation.

### 5.2 YOLO + ByteTrack on GPU for Ball Tracking

**Current state**: YOLO detection runs on the Mac's MPS (Apple Silicon) during the motion scan phase. Ball detection is not used for event classification -- the pipeline relies on motion frame-differencing instead.

**What GPU-accelerated ball tracking enables**:
- Real-time ball position tracking at 30 FPS on GPU
- Ball trajectory features as additional VLM context ("ball moved from position A to position B at speed X")
- Automatic sideline boundary detection (ball near touchline = potential throw-in)
- Ball velocity spike detection (ball kicked hard = potential shot)

**Hardware**: A single RTX 3090 or cloud GPU handles YOLO + ByteTrack at 30+ FPS easily.

**Challenge**: COCO-trained YOLO struggles with soccer ball detection at distance. The ball is often <20 pixels and confused with heads, markings, and compression artifacts. A soccer-specific YOLO model (e.g., from the Roboflow Sports ecosystem) would be needed.

**Expected accuracy improvement**: 
- Throw-in detection: +20-30% (ball near sideline triggers candidate)
- Shot detection: +5-10% (ball velocity spike confirms shot)
- Save detection: negligible (ball is occluded by GK at save moment)

### 5.3 Multi-Model Ensemble

Run a fast, small model for initial screening and a large model for ambiguous cases:

| Stage | Model | Role | Cost |
|-------|-------|------|------|
| Screening | Qwen3-VL-8B (fine-tuned) | Fast binary: "is this an event?" | <1s per call |
| Classification | Qwen3-VL-32B or Claude | Multi-class on confirmed events only | 3-8s per call |
| Arbitration | Rules engine | Tiebreak using structural inference | 0s |

This reduces Claude API costs by ~60% (only the ambiguous 40% of candidates go to the expensive model) while maintaining high accuracy.

**Hardware**: The 8B model fits on a single RTX 3090. The 32B can stay on the existing 2x RTX 3090. No cloud GPU needed for inference.

### 5.4 Real-Time or Near-Real-Time Processing

**Current**: ~96 minutes to process a 114-minute match (roughly 1:1 ratio, but not real-time).

With cloud GPUs:
- VLM inference parallelism: 4-8 concurrent calls on H100 = ~4x speedup on VLM phase
- YOLO + tracking: real-time at 30+ FPS on any modern GPU
- FFmpeg frame extraction: CPU-bound, not improved by GPU

**Achievable speed**: ~30-40 minutes per match with an H100. Near-real-time event detection (within 30s of occurrence) would require streaming pipeline architecture, which is a significant engineering investment beyond hardware.

---

## 6. Cost-Accuracy Matrix

### Self-Hosted Options

| Configuration | GPUs | VRAM | Est. Cloud $/hr | Model | Expected Accuracy | Speed (min/match) |
|--------------|------|------|-----------------|-------|------------------|------------------|
| Current setup | 2x RTX 3090 | 48 GB | $0 (owned) | Qwen3-VL-32B-FP8 | 66% recall | ~96 |
| 1x A100 40GB | 1x A100 | 40 GB | ~$1.00 | Qwen3-VL-32B-FP8 | 66% (same) | ~80 |
| 1x A100 80GB | 1x A100 | 80 GB | ~$1.30 | Qwen3-VL-32B-FP16 | 67% | ~75 |
| 2x A100 80GB | 2x A100 | 160 GB | ~$2.60 | Qwen2.5-VL-72B-FP16 | 65-67% * | ~90 |
| 1x H100 80GB | 1x H100 | 80 GB | ~$2.00 | Qwen3-VL-32B-FP16 | 67% | ~55 |
| 8x A100 80GB | 8x A100 | 640 GB | ~$10.40 | Qwen3-VL-235B-FP8 | 69-71% | ~120 |
| 8x H100 80GB | 8x H100 | 640 GB | ~$16.00 | Qwen3-VL-235B-FP8 | 69-71% | ~80 |

\* Qwen2.5-VL-72B scores lower than Qwen3-VL-32B on video understanding benchmarks. Running a larger older model is a downgrade.

### API Options (Zero Infrastructure)

| Provider | Model | Est. $/match | Expected Accuracy | Speed (min/match) | Notes |
|---------|-------|-------------|-------------------|-------------------|-------|
| Self-hosted (current) | Qwen3-VL-32B-FP8 | $0 | 66% | ~96 | Owned hardware |
| OpenRouter | Qwen3-VL-32B | ~$0.60 | 66% (same model) | ~60 | Faster hardware, same model |
| Anthropic API | Claude Sonnet 4.6 | ~$8-17 | 70-73% | ~50 | Best reasoning |
| Anthropic Batch | Claude Sonnet 4.6 | ~$4-9 | 70-73% | ~120 (async) | 50% discount, delayed |
| Anthropic API | Claude Opus 4.6 | ~$15-28 | 71-74% | ~60 | Highest quality |
| OpenAI | GPT-4o | ~$10-20 | 68-71% | ~50 | Native video tokens |

### Training Investments (One-Time)

| Investment | Cloud Cost | Time | Expected Accuracy Gain | Ongoing Cost |
|-----------|-----------|------|----------------------|-------------|
| Fine-tune Qwen3-VL-8B (LoRA, 5K clips) | ~$50 | 4-6 hrs | +5-10% | $0 (runs on RTX 3090) |
| Fine-tune Qwen3-VL-8B (LoRA, 20K clips) | ~$100-150 | 8-12 hrs | +10-15% | $0 (runs on RTX 3090) |
| Fine-tune Qwen3-VL-32B (LoRA, 20K clips) | ~$300-500 | 24-48 hrs | +8-12% | $0 (runs on 2x RTX 3090) |
| Soccer YOLO ball detector training | ~$30-50 | 2-4 hrs | Enables ball tracking features | $0 |

---

## 7. Recommendations (Prioritised)

### Tier 1: High ROI, Low Cost (Do Now)

**1a. Try Claude Sonnet 4.6 Batch API as a comparison benchmark** ($5-10 for one match)

The pipeline already supports Claude as a backend (`ANTHROPIC_API_KEY` in `.env`). Run the Rush video through Claude Sonnet 4.6 and compare against the Qwen3-VL-32B results. This gives a concrete data point for "how much does a better model help?" before committing to infrastructure changes.

Use Batch API for 50% discount -- latency does not matter for offline processing.

**1b. Improve candidate generation for throw-ins** ($0, code changes only)

Throw-in recall (12%) is the largest accuracy gap and has nothing to do with VLM quality. After every event that results in the ball going out of play (shot off target, clearance), probe the next 5-15 seconds for a throw-in. This is structural inference, not hardware.

### Tier 2: Medium ROI, Medium Cost (Next Quarter)

**2a. Fine-tune Qwen3-VL-8B on soccer domain data** (~$100-150)

Curate 10-20K labeled clips from:
- Existing pipeline runs with corrected labels
- SoccerNet Ball Action Spotting dataset (12 event classes, free)
- Sideline camera footage with manual labels

LoRA fine-tune on a rented A100 for 8-12 hours. Deploy the fine-tuned model on a single RTX 3090, freeing the second 3090 for YOLO tracking.

**2b. Train a soccer-specific YOLO ball detector** (~$30-50)

Fine-tune YOLOv8 on a soccer ball dataset (Roboflow has several with 5K+ images). Use the ball detector to generate spatial features (ball position, velocity) that feed into VLM prompts as text annotations.

### Tier 3: Low ROI for Current Bottlenecks (Consider Later)

**3a. Upgrade to cloud A100/H100 for inference**

Only worth it if:
- Processing volume increases to 5+ matches/week (amortises cloud cost)
- Latency becomes a requirement (need results within 30 minutes)
- Running a fine-tuned model that needs more VRAM

For the current 1-2 matches/week, the 2x RTX 3090 setup is sufficient.

**3b. Qwen3-VL-235B for maximum accuracy**

Only pursue after exhausting Tier 1 and 2 options. The 3-5% accuracy gain does not justify 8x A100 costs unless accuracy requirements increase significantly (e.g., commercial deployment).

---

## 8. What Will NOT Be Solved by Hardware

No amount of GPU hardware will fix these problems:

1. **Spatial resolution at 50m**: The sideline camera produces ~20-pixel representations of the goalkeeper. No VLM can reliably distinguish "ball in hands" from "ball on ground" at this resolution. Solutions: second camera closer to the goal, or higher-resolution camera.

2. **Throw-in candidate generation**: Throw-ins produce minimal motion delta. The motion scan phase fundamentally cannot detect most throw-ins. Solutions: ball tracking (YOLO), structural inference from events that send the ball out.

3. **Subjective events (fouls, offsides)**: These require multi-camera spatial calibration (Hawk-Eye) or human judgment. No single-camera VLM pipeline will reliably detect these.

4. **Replay confusion**: If the video includes broadcast-style replays, the pipeline will detect events twice. Solutions: replay detection (scene boundary detection), or restrict to single-camera non-broadcast footage (which this pipeline already assumes).

5. **Per-team action attribution**: Knowing *which team* performed an action requires player-to-team assignment. The pipeline has team color configuration but the VLM does not consistently use it for attribution. Solutions: YOLO player detection + team clustering (SigLIP features + KMeans).

---

## Sources

- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL) -- model sizes, benchmarks, VRAM guidance
- [Qwen3-VL-32B vs Qwen2.5-VL-72B Comparison](https://llm-stats.com/models/compare/qwen2.5-vl-72b-vs-qwen3-vl-32b-instruct) -- benchmark tables
- [Domain Adaptation of VLM for Soccer Video Understanding](https://arxiv.org/abs/2505.13860) (Jiang et al., CVPR 2025 Workshop) -- fine-tuning 11.8% to 63.5%
- [Do We Need Large VLMs for Spotting Soccer Actions?](https://arxiv.org/abs/2506.17144) (Chakraborty et al., 2025) -- LLM-only approach
- [SPORTU Benchmark](https://arxiv.org/abs/2410.08474) -- VLM sports understanding benchmarks
- [LLM Quantization: BF16 vs FP8 vs INT4](https://research.aimultiple.com/llm-quantization/) -- quantization accuracy loss data
- [FP8 Quantization in Deep Neural Networks](https://www.emergentmind.com/topics/fp8-quantization) -- FP8 accuracy retention
- [Claude API Pricing](https://platform.claude.com/docs/en/about-claude/pricing) -- current Anthropic pricing
- [Qwen3-VL-32B on OpenRouter](https://openrouter.ai/qwen/qwen3-vl-32b-instruct) -- API pricing
- [H100 Rental Prices Compared](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison) -- cloud GPU pricing 2026
- [Cloud GPU Pricing 2026](https://www.synpixcloud.com/blog/cloud-gpu-pricing-comparison-2026) -- A100/H100 market rates
- [vLLM Qwen3-VL Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html) -- deployment requirements
- [InternVL3-78B](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/) -- alternative VLM benchmarks
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) -- frame count vs accuracy analysis
- [MotionBench: Fine-grained Video Motion Understanding](https://openaccess.thecvf.com/content/CVPR2025/papers/Hong_MotionBench_Benchmarking_and_Improving_Fine-grained_Video_Motion_Understanding_for_Vision_CVPR_2025_paper.pdf) -- motion understanding benchmarks
