# AWS Cloud Architecture: Soccer Video Pipeline as On-Demand Service

**Date**: 2026-04-03
**Author**: AI Infrastructure Architect
**Status**: Architecture design -- ready for implementation

---

## Executive Summary

This document designs an AWS-native, on-demand cloud architecture for the soccer video
pipeline. The system processes 90-120 minute match recordings (5-15 GB MP4) through
motion detection, audio analysis, VLM classification, structural inference, and clip
assembly to produce goalkeeper and highlights reels.

The architecture uses Step Functions for orchestration, AWS Batch for compute, S3 for
storage, and a three-tier pricing model (Budget/Standard/Premium) that matches cost to
accuracy requirements. No always-on infrastructure -- every component scales to zero.

**Cost per match**: $3-8 (Budget, API-only), $10-18 (Standard, cloud GPU), $25-50
(Premium, large models + ensemble).

---

## 1. Architecture Overview

```
                              UPLOAD FLOW
  ┌──────────┐    presigned    ┌────────────┐
  │  Client   │───────URL─────►│     S3     │
  │ (Web/CLI) │                │   Ingest   │
  └─────┬─────┘                │   Bucket   │
        │                      └──────┬─────┘
        │ POST /jobs                  │ S3 Event
        ▼                             ▼
  ┌───────────┐              ┌──────────────┐
  │    API    │──────────────►│  Step        │
  │  Gateway  │   start      │  Functions   │
  │  + Lambda │   execution  │  (Orchestr.) │
  └───────────┘              └──────┬───────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
             ┌───────────┐  ┌───────────┐   ┌───────────┐
             │ AWS Batch │  │ AWS Batch │   │ AWS Batch │
             │ CPU Stage │  │ GPU Stage │   │ CPU Stage │
             │ (Motion + │  │ (VLM via  │   │ (Assembly │
             │  Audio)   │  │  vLLM or  │   │  FFmpeg)  │
             └─────┬─────┘  │  API)     │   └─────┬─────┘
                   │        └─────┬─────┘         │
                   │              │               │
                   ▼              ▼               ▼
             ┌─────────────────────────────────────────┐
             │               S3 Working Bucket          │
             │  /jobs/{id}/candidates.jsonl              │
             │  /jobs/{id}/events.jsonl                  │
             │  /jobs/{id}/frames/                       │
             └─────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │    S3 Output     │
                          │ goalkeeper_reel  │
                          │ highlights_reel  │
                          └──────────────────┘
                                    │
                              presigned URL
                                    ▼
                            ┌──────────┐
                            │  Client  │
                            │ download │
                            └──────────┘


                     STEP FUNCTIONS STATE MACHINE
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
  │  │ Ingest  │───►│ Motion  │───►│  Audio  │             │
  │  │ (probe) │    │  Scan   │    │  Boost  │             │
  │  └─────────┘    └─────────┘    └─────────┘             │
  │       │              │              │                   │
  │       │         CPU Batch      CPU Batch                │
  │       ▼              ▼              ▼                   │
  │  ┌─────────┐    ┌─────────────────────┐                │
  │  │ Choose  │───►│   VLM Classify      │                │
  │  │  Tier   │    │  (Budget: API call) │                │
  │  └─────────┘    │  (Std: GPU Batch)   │                │
  │                 │  (Prem: Ensemble)   │                │
  │                 └──────────┬──────────┘                │
  │                            │                           │
  │                            ▼                           │
  │                 ┌─────────────────────┐                │
  │                 │ Structural Phases   │                │
  │                 │ 3a-3g (Lambda)      │                │
  │                 └──────────┬──────────┘                │
  │                            │                           │
  │                            ▼                           │
  │                 ┌─────────────────────┐                │
  │                 │ Clip Assembly       │                │
  │                 │ (CPU Batch/Fargate) │                │
  │                 └──────────┬──────────┘                │
  │                            │                           │
  │                            ▼                           │
  │                 ┌─────────────────────┐                │
  │                 │ Notify + Cleanup    │                │
  │                 │ (Lambda + SNS)      │                │
  │                 └─────────────────────┘                │
  └─────────────────────────────────────────────────────────┘
```

---

## 2. Service Selection and Rationale

### 2.1 Orchestration: Step Functions + AWS Batch

**Why Step Functions + AWS Batch (not SQS+ECS or standalone Batch)**:

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Step Functions + Batch** | Visual workflow, built-in retry/error handling, native Batch integration, per-stage compute selection | $0.025/1000 state transitions (negligible) | **Selected** |
| SQS + ECS Services | Simple queue-driven, familiar | Must build retry/DAG logic, no visual monitoring, always-on ECS services for polling | Rejected |
| Standalone AWS Batch | Handles job dependencies natively | Weaker orchestration, no conditional branching for tier selection | Rejected |
| Celery + Redis (current) | Already built | Requires always-on Redis + worker, not serverless | Keep for local dev only |

Step Functions provides the conditional branching needed for tier selection (Budget jobs skip
GPU stages entirely), built-in error handling with retry policies, and the `.sync` integration
pattern that waits for Batch jobs to complete before advancing.

### 2.2 Compute: AWS Batch (EC2 for GPU, Fargate for CPU)

**CPU stages** (motion scan, audio, assembly): **Fargate** on Batch.
- No GPU needed, scales to zero, no AMI management.
- 4 vCPU / 16 GB RAM is sufficient for FFmpeg frame extraction.
- Fargate cold start ~30-60s is acceptable (not on the critical path).

**GPU stages** (VLM inference): **EC2** on Batch with managed compute environments.
- Fargate does not support GPU -- EC2 is the only option.
- Spot instances with on-demand fallback for cost savings.
- Custom AMI with pre-pulled Docker images to reduce cold start from ~10 min to ~3-5 min.

### 2.3 Storage: S3 + EBS

| Purpose | Service | Config | Notes |
|---------|---------|--------|-------|
| Video upload | S3 Ingest bucket | Intelligent-Tiering | Presigned multipart upload |
| Working scratch | S3 Working bucket | Standard | Frames, candidates JSONL, intermediate state |
| Output reels | S3 Output bucket | Intelligent-Tiering | Presigned download URLs |
| GPU scratch | EBS gp3 | 100 GB, attached to Batch EC2 | Fast local I/O for vLLM model weights |
| Model weights | S3 + EBS snapshot | AMI-baked EBS | Pre-loaded Qwen3-VL weights avoid download |

**Why S3 for scratch (not EFS)**:
- Pipeline stages are sequential, not concurrent reads. S3 multipart upload/download between
  stages adds ~30s overhead but avoids EFS hourly cost ($0.30/GB-month) and throughput charges.
- Frame images are small (10-50 KB each JPEG), written in bulk, read once. S3 PutObject
  is cheaper than EFS for this access pattern.
- Exception: If VLM calls need to read frames from disk during inference, the GPU Batch job
  downloads them from S3 to local EBS at job start (~15s for 300 frames).

### 2.4 API Layer: API Gateway + Lambda

- **API Gateway** (HTTP API, not REST): Lower latency, lower cost ($1.00/million requests).
- **Lambda functions**: Job submission, status queries, presigned URL generation, webhook callbacks.
- **DynamoDB**: Job state table (replaces the current file-backed JSON JobStore).

No always-on API server. The current FastAPI app stays for local dev; cloud deployment
replaces it with API Gateway + Lambda handlers that share the same Pydantic models.

---

## 3. Pipeline Stage Mapping

### 3.1 Stage-by-Stage Compute Assignment

| Pipeline Stage | Duration | Compute | Instance | Why |
|---------------|----------|---------|----------|-----|
| **Ingest** (ffprobe + SHA-256) | <1 min | Lambda | 1 GB, 60s timeout | Stateless, fast |
| **Motion scan** (frame-diff) | ~15 min | Batch Fargate | 4 vCPU, 16 GB | CPU-bound, reads every frame from S3 |
| **Audio analysis** | ~5 min | Batch Fargate | 2 vCPU, 8 GB | FFmpeg spectral, CPU-bound |
| **VLM classify** (Budget) | ~30-40 min | Lambda (fan-out) | 256 MB each, 90s | API calls to Claude/GPT-4o, highly parallel |
| **VLM classify** (Standard) | ~25-35 min | Batch EC2 GPU | g5.xlarge (1x A10G) | vLLM serving Qwen3-VL-32B-FP8 |
| **VLM classify** (Premium) | ~20-30 min | Batch EC2 GPU | g5.12xlarge (4x A10G) | vLLM with higher concurrency + API ensemble |
| **Structural inference** | <1 min | Lambda | 512 MB, 30s | Pure Python temporal logic |
| **Clip assembly** | ~5 min | Batch Fargate | 4 vCPU, 8 GB | FFmpeg concat, I/O bound |
| **Notify** | <1s | Lambda + SNS/SES | 128 MB | Webhook/email on completion |

### 3.2 Budget Tier: VLM via API (No GPU)

The Budget tier replaces the GPU VLM stage with parallel API calls:

```
  Step Functions
       │
       ▼
  ┌──────────────────────────────────┐
  │ Map State: VLM API Fan-Out       │
  │                                  │
  │  ┌────────┐ ┌────────┐          │
  │  │Lambda 1│ │Lambda 2│ ... x220 │
  │  │ Call   │ │ Call   │          │
  │  │Claude  │ │Claude  │          │
  │  └────────┘ └────────┘          │
  │                                  │
  │  Concurrency: 20-50 parallel     │
  │  Per-call: extract frames from   │
  │  S3, base64 encode, POST to API  │
  └──────────────────────────────────┘
```

Each Lambda invocation:
1. Reads 10-24 pre-extracted frames from S3 Working bucket
2. Base64 encodes them
3. Calls Claude Sonnet 4.6 (or GPT-4o) API
4. Writes verdict to S3 as JSONL

With 20-50 concurrent Lambdas, 220 VLM calls complete in ~2-5 minutes (vs 30-40 min serial).
This makes the Budget tier **faster** than self-hosted GPU for the VLM stage.

**Frame pre-extraction**: A Fargate Batch job runs before the VLM fan-out to extract all
frames for all candidates and write them to S3. This takes ~5 min and avoids each Lambda
needing FFmpeg + the full source video.

### 3.3 Standard Tier: Self-Hosted vLLM on GPU

```
  Step Functions
       │
       ├── Start vLLM server (Batch EC2 GPU job, sidecar pattern)
       │     └── g5.xlarge: pull model weights from EBS, start vLLM
       │         Wait for /health endpoint (readiness probe)
       │         Cold start: 3-5 min (AMI-baked) / 8-10 min (fresh pull)
       │
       ├── Run VLM worker (same Batch job, second container)
       │     └── Sequential VLM calls to localhost:8000
       │         ~220 calls x 3-8s = 15-30 min
       │
       └── Terminate GPU instance on completion
```

**Model weight strategy**: Bake Qwen3-VL-32B-FP8 weights (~35 GB) into a custom AMI's EBS
snapshot. When Batch launches the instance, the volume is already attached with weights
pre-loaded. This eliminates the ~10 min S3 download and reduces cold start to 3-5 min
(vLLM model loading only).

**Single-container approach**: Run vLLM and the pipeline worker in the same Batch job
as two processes. The worker calls vLLM at `localhost:8000`. This avoids network latency
and simplifies the deployment.

### 3.4 Premium Tier: Ensemble + Larger Models

The Premium tier adds a multi-model ensemble strategy:

```
  ┌─────────────────────────────────────────────┐
  │  Premium VLM Stage                          │
  │                                             │
  │  Pass 1: Qwen3-VL-32B on g5.xlarge         │
  │    └── Classify all 220 candidates          │
  │    └── Mark low-confidence (<0.7) subset    │
  │                                             │
  │  Pass 2: Claude Opus 4.6 API (parallel)     │
  │    └── Re-classify low-confidence subset    │
  │    └── ~60-80 calls (30-40% of total)       │
  │                                             │
  │  Arbitration: majority vote + structural    │
  │    └── Lambda: merge verdicts               │
  └─────────────────────────────────────────────┘
```

This reduces Claude API costs by 60-70% (only ambiguous candidates go to the expensive
model) while getting the best of both worlds: Qwen's speed and Claude's reasoning.

**Alternative Premium option**: Use g5.12xlarge (4x A10G, 96 GB VRAM) to run
Qwen3-VL-32B at FP16 with higher batch sizes for faster throughput, or experiment with
the Qwen3-VL-235B-A22B MoE model (needs p4d.24xlarge for 8x A100).

---

## 4. Instance Selection Deep Dive

### 4.1 GPU Instance Comparison

| Instance | GPUs | VRAM | On-Demand $/hr | Spot $/hr (est.) | Model Fits? | Notes |
|----------|------|------|----------------|-------------------|-------------|-------|
| g5.xlarge | 1x A10G | 24 GB | $1.006 | $0.35-0.50 | Qwen3-VL-32B-FP8 (tight) | A10G has 24 GB; FP8 model is ~18 GB, leaves ~6 GB for KV cache. Works with `max-model-len ~16K`. |
| g5.2xlarge | 1x A10G | 24 GB | $1.212 | $0.40-0.60 | Same GPU, more CPU/RAM | Extra CPU helps frame extraction |
| g5.12xlarge | 4x A10G | 96 GB | $5.672 | $1.70-2.50 | Qwen3-VL-32B-FP16 + headroom | Tensor-parallel across 4 GPUs, higher throughput |
| g6e.xlarge | 1x L40S | 48 GB | $1.861 | $0.55-0.80 | Qwen3-VL-32B-FP8 (comfortable) | 48 GB VRAM gives full 32K context + larger batches |
| g6e.2xlarge | 1x L40S | 48 GB | $2.35 (est.) | $0.70-1.00 | Same GPU, more CPU | Better for concurrent frame extraction |
| p4d.24xlarge | 8x A100 40GB | 320 GB | $32.77 | $10-15 | Qwen3-VL-235B-A22B | Overkill for 32B, needed for 235B MoE |

**Recommended GPU instances by tier**:

- **Standard**: g6e.xlarge (1x L40S, 48 GB VRAM, $1.86/hr on-demand)
  - L40S has 48 GB VRAM -- fits Qwen3-VL-32B-FP8 comfortably with full 32K context window
  - Better than g5.xlarge where A10G's 24 GB forces reduced context length
  - L40S FP8 throughput: ~2x faster than A10G for transformer inference
  - If g6e availability is limited, fall back to g5.xlarge with reduced max-model-len

- **Premium GPU option**: g5.12xlarge (4x A10G, 96 GB, $5.67/hr)
  - Run Qwen3-VL-32B-FP16 with tensor-parallel across 4 GPUs
  - Higher batch sizes (max-num-seqs 8-16) for 2-3x throughput
  - Or run Qwen3-VL-32B-FP8 with massive KV cache for 64K+ context

- **Premium 235B option**: p4d.24xlarge (8x A100 40GB, $32.77/hr on-demand)
  - Only for Qwen3-VL-235B-A22B experiments
  - At ~$33/hr and ~1 hr per match, this adds ~$33 per match in GPU cost alone
  - Not recommended unless accuracy testing proves >5% improvement

### 4.2 A10G 24GB Feasibility Check for Qwen3-VL-32B-FP8

The A10G has 24 GB VRAM. Qwen3-VL-32B-FP8 model weights consume ~18 GB. That leaves
~6 GB for KV cache, activation memory, and framework overhead.

**vLLM memory budget on A10G 24GB**:
- Model weights: ~18 GB (FP8)
- Framework overhead (CUDA, vLLM): ~1 GB
- Available for KV cache: ~5 GB
- With `gpu-memory-utilization 0.95`: ~4.8 GB KV cache

**KV cache math**: Qwen3-VL-32B has 64 layers, 40 KV heads, 128 dim per head.
- Per token: 2 (K+V) x 64 layers x 40 heads x 128 dim x 1 byte (FP8 cache) = ~655 KB
- 4.8 GB / 655 KB = ~7,300 tokens of KV cache
- With `max-num-seqs 1`: supports ~7K context per request (sufficient -- our requests
  are ~5K-10K tokens)
- With `max-num-seqs 2`: ~3.5K context each (tight but workable for smaller frame sets)

**Verdict**: A10G works but is tight. The pipeline must use `max-model-len 8192` and
`max-num-seqs 1-2`. For a more comfortable fit, g6e.xlarge (L40S, 48 GB) is preferred.

### 4.3 CPU Instance Selection

For Fargate (CPU stages), the sizing is straightforward:

| Stage | vCPU | Memory | Ephemeral Storage | Duration |
|-------|------|--------|-------------------|----------|
| Frame extraction | 4 | 16 GB | 100 GB | ~15 min |
| Audio analysis | 2 | 8 GB | 20 GB | ~5 min |
| Clip assembly | 4 | 8 GB | 60 GB | ~5 min |

Fargate ephemeral storage goes up to 200 GB (20 GB free, then $0.000111/GB-hour for the
rest). For the frame extraction stage, 100 GB covers ~300 candidates x 24 frames x 50 KB
= ~360 MB of frames plus the source video working copy.

**Wait -- the source video is 5-15 GB**. Fargate's 200 GB ephemeral storage handles this
fine, but we should stream the video from S3 rather than downloading the full file first.
FFmpeg supports S3 input via `s3://` URLs or HTTPS presigned URLs with seeking.

---

## 5. Data Flow Detail

### 5.1 Upload Flow

```
1. Client requests presigned URL:
   POST /upload/init → Lambda → returns { uploadId, presignedUrls[] }
   (Multipart upload, 100 MB parts for a 10 GB video = ~100 parts)

2. Client uploads parts directly to S3:
   PUT presignedUrls[0] → S3 (part 1)
   PUT presignedUrls[1] → S3 (part 2)
   ... (parallel uploads, ~5 min on 100 Mbps connection)

3. Client completes upload:
   POST /upload/complete → Lambda → S3 CompleteMultipartUpload

4. Client submits job:
   POST /jobs → Lambda →
     - Validate S3 object exists
     - Run ffprobe (Lambda, <30s)
     - Create job record in DynamoDB
     - Start Step Functions execution
     - Return job_id
```

**S3 Transfer Acceleration**: Enable on the ingest bucket for users uploading from distant
regions. Adds ~$0.04/GB but can double upload speed.

### 5.2 Inter-Stage Data Passing

Each pipeline stage reads inputs from and writes outputs to S3, with the job's working
prefix as the namespace:

```
s3://pipeline-working/{job_id}/
  source.mp4                    # Copied from ingest bucket (or referenced directly)
  metadata.json                 # ffprobe output, duration, fps
  candidates.jsonl              # Phase 1+2 output: motion + audio candidates
  frames/                       # Pre-extracted frames for VLM
    candidate_001/
      frame_00.jpg
      frame_01.jpg
      ...
    candidate_002/
      ...
  verdicts.jsonl                # Phase 3 output: VLM classification results
  events.jsonl                  # Phase 3a-3g output: final event list
  clips/                        # Phase 5 output: individual clip files
    clip_001.mp4
    clip_002.mp4
  reels/
    goalkeeper_reel.mp4          # Final output
    highlights_reel.mp4          # Final output
```

Step Functions passes S3 paths between states via the state machine's JSON payload.
Each Batch job receives the job_id and derives all paths from it.

### 5.3 Video Access Pattern

The source video (5-15 GB) is accessed differently by each stage:

| Stage | Access Pattern | Strategy |
|-------|---------------|----------|
| Motion scan | Sequential read, every frame at 0.5s intervals | Stream via S3 presigned URL with byte-range seeks. FFmpeg `-ss` seeks to each timestamp. |
| Audio analysis | Sequential read of audio track | `ffmpeg -i s3://... -vn -acodec pcm_s16le` streams audio only (~100 MB) |
| Frame extraction | Random-access seeks to ~220 timestamps | Download to EBS/ephemeral first, then extract. Random seeks over S3 are slow. |
| Clip assembly | Random-access cuts at ~20-40 timestamps | Download source + cut. Or use S3 byte-range reads with FFmpeg `-ss -t` per clip. |

**Recommendation**: For motion scan and audio, stream from S3. For frame extraction and
clip assembly, download to local storage first (Fargate ephemeral or Batch EBS).

---

## 6. Cost Analysis

### 6.1 Shared Costs (All Tiers)

These costs are incurred regardless of tier:

| Item | Cost Per Match | Calculation |
|------|---------------|-------------|
| S3 storage (source, 10 GB, 7 days) | $0.007 | $0.023/GB-month x 10 GB x 7/30 |
| S3 storage (working, 30 GB, 1 day) | $0.023 | $0.023/GB-month x 30 GB x 1/30 |
| S3 storage (output, 0.5 GB, 30 days) | $0.012 | $0.023/GB-month x 0.5 GB |
| S3 requests (PUT/GET) | $0.05 | ~10,000 requests |
| S3 data transfer (upload) | $0.00 | Inbound is free |
| S3 data transfer (download output) | $0.045 | $0.09/GB x 0.5 GB |
| Step Functions | $0.001 | ~40 state transitions x $0.025/1000 |
| DynamoDB | $0.001 | ~20 reads + writes |
| API Gateway | $0.001 | ~10 API calls |
| Lambda (ingest + structural) | $0.01 | ~5 invocations x 1 GB x 10s |
| Fargate: motion scan | $0.17 | 4 vCPU x $0.04/hr x 0.25 hr + 16 GB x $0.004/hr x 0.25 hr |
| Fargate: audio analysis | $0.04 | 2 vCPU x $0.04/hr x 0.08 hr + 8 GB x $0.004/hr x 0.08 hr |
| Fargate: frame extraction | $0.17 | 4 vCPU x $0.04/hr x 0.25 hr + 16 GB x $0.004/hr x 0.25 hr |
| Fargate: clip assembly | $0.06 | 4 vCPU x $0.04/hr x 0.08 hr + 8 GB x $0.004/hr x 0.08 hr |
| **Shared total** | **~$0.60** | |

### 6.2 Budget Tier: API-Only VLM ($3-8 per match)

No GPU instances. VLM classification via cloud API calls.

| VLM Backend | Input Tokens | Output Tokens | API Cost | Total (API + shared) |
|-------------|-------------|---------------|----------|---------------------|
| Claude Sonnet 4.6 Batch | 5.4M | 22K | $8.25 | **$8.85** |
| Claude Sonnet 4.6 (real-time) | 5.4M | 22K | $16.50 | **$17.10** |
| GPT-4o | ~5M | 22K | $12.50-20 | **$13-21** |
| OpenRouter Qwen3-VL-32B | ~1M | 22K | $0.60 | **$1.20** |

**Important token math clarification**: The gpu-upgrade-analysis doc estimates 5.4M input
tokens per match for Claude (220 calls x 15 images x 1,600 tokens/image + prompt). This
is the realistic estimate. The lower $3.30 estimate assumed only 200 tokens per image,
which underestimates Claude's image token consumption.

**Recommended Budget config**: Claude Sonnet 4.6 Batch API at $8.25/match. The 50% batch
discount is significant, and the async nature (results within 24 hours) is fine for offline
match processing. If faster results are needed, real-time Claude at $16.50/match.

For the absolute cheapest option, OpenRouter's Qwen3-VL-32B endpoint at ~$0.60/match is
remarkably affordable, though you lose the accuracy advantage of Claude's spatial reasoning.

**VLM fan-out Lambda costs**:
- 220 Lambda invocations x 256 MB x 10s average = ~$0.01 (negligible)

| Component | Cost |
|-----------|------|
| Shared infrastructure | $0.60 |
| Claude Sonnet 4.6 Batch API | $8.25 |
| Lambda fan-out | $0.01 |
| **Budget tier total** | **$8.86** |

Or with OpenRouter Qwen3-VL-32B:

| Component | Cost |
|-----------|------|
| Shared infrastructure | $0.60 |
| OpenRouter Qwen3-VL-32B | $0.60 |
| Lambda fan-out | $0.01 |
| **Budget tier total (OpenRouter)** | **$1.21** |

### 6.3 Standard Tier: Self-Hosted VLM on Cloud GPU ($10-18 per match)

GPU instance runs vLLM with Qwen3-VL-32B-FP8 for the VLM classification stage.

**GPU runtime estimate**: 3-5 min cold start + 25-35 min VLM inference = ~30-40 min total.

| Instance | VRAM | On-Demand $/hr | Spot $/hr | GPU Cost (40 min) | Total |
|----------|------|----------------|-----------|-------------------|-------|
| g5.xlarge (A10G 24GB) | 24 GB | $1.006 | ~$0.40 | $0.67 OD / $0.27 Spot | $1.27 / $0.87 |
| g6e.xlarge (L40S 48GB) | 48 GB | $1.861 | ~$0.65 | $1.24 OD / $0.43 Spot | $1.84 / $1.03 |
| g5.2xlarge (A10G 24GB) | 24 GB | $1.212 | ~$0.48 | $0.81 OD / $0.32 Spot | $1.41 / $0.92 |

Wait -- that is surprisingly cheap. Let me double-check. At $1.86/hr for g6e.xlarge,
40 minutes = 0.667 hours = $1.24. Plus $0.60 shared = $1.84 total. That is less than the
Claude API cost.

**But**: GPU cold start adds to the wall-clock time. The 3-5 min cold start does not change
the billing much (adds ~$0.15 at g6e.xlarge rates), but it does make the VLM stage take
~35-45 min total including startup.

**Spot interruption risk**: VLM inference is ~30 min. If a Spot interruption occurs mid-job,
the pipeline must restart the VLM stage from scratch (verdicts already written to S3 are
preserved, so only the in-flight call is lost). The Step Functions retry logic handles this
automatically. With g5/g6e Spot interruption rates typically <5% for a 40-min window, this
is an acceptable risk.

**Recommended Standard config**: g6e.xlarge Spot with on-demand fallback.

| Component | Cost (Spot) | Cost (On-Demand) |
|-----------|-------------|------------------|
| Shared infrastructure | $0.60 | $0.60 |
| g6e.xlarge GPU (40 min) | $0.43 | $1.24 |
| Cold start overhead (5 min) | $0.05 | $0.16 |
| **Standard tier total** | **$1.08** | **$2.00** |

This is dramatically cheaper than the API-based Budget tier. The tradeoff: Budget tier
gets Claude's potentially better reasoning, Standard tier gets the same Qwen3-VL-32B
model the pipeline was tuned on.

**Revised cost tiers**: The original $8-20 estimate for Standard was too high. Actual
cloud GPU costs for the 32B model are $1-2/match. The cost range should be:

| Tier | Actual Cost Range | What You Get |
|------|-------------------|-------------|
| Budget (API) | $1-9 | OpenRouter ($1) to Claude Batch ($9) |
| Standard (GPU) | $1-2 | Qwen3-VL-32B on cloud GPU (same as local) |
| Premium (Ensemble) | $5-35 | GPU + Claude double-check on hard cases |

### 6.4 Premium Tier: Ensemble ($5-35 per match)

The Premium tier runs Qwen3-VL-32B on GPU for all candidates, then sends low-confidence
results to Claude for a second opinion.

**Ensemble strategy**: Of ~220 VLM calls, approximately 30-40% produce low-confidence
(<0.7) or ambiguous results. These ~70-80 candidates get re-classified by Claude.

| Component | Cost |
|-----------|------|
| Shared infrastructure | $0.60 |
| g6e.xlarge GPU (40 min, Spot) | $0.48 |
| Claude Opus 4.6 API (80 calls x 24.5K tokens) | ~$10.30 |
| Lambda ensemble orchestration | $0.01 |
| **Premium tier total (Opus ensemble)** | **~$11.40** |

Or with Claude Sonnet 4.6 instead of Opus:

| Component | Cost |
|-----------|------|
| Shared infrastructure | $0.60 |
| g6e.xlarge GPU (40 min, Spot) | $0.48 |
| Claude Sonnet 4.6 API (80 calls) | ~$6.00 |
| **Premium tier total (Sonnet ensemble)** | **~$7.10** |

For maximum accuracy with the 235B MoE model:

| Component | Cost |
|-----------|------|
| Shared infrastructure | $0.60 |
| p4d.24xlarge (60 min, Spot) | ~$12.00 |
| **Premium tier total (235B)** | **~$12.60** |

### 6.5 Cost Summary Table

| Tier | VLM Backend | GPU Instance | $/Match | Wall-Clock Time | Accuracy (est.) |
|------|-------------|-------------|---------|-----------------|-----------------|
| Budget-Lite | OpenRouter Qwen3-VL-32B | None | **$1.20** | ~35 min | 66% (same model) |
| Budget | Claude Sonnet 4.6 Batch | None | **$8.85** | ~120 min (async) | 70-73% |
| Standard | Qwen3-VL-32B-FP8 (vLLM) | g6e.xlarge Spot | **$1.08** | ~55 min | 66% |
| Standard-OD | Qwen3-VL-32B-FP8 (vLLM) | g6e.xlarge On-Demand | **$2.00** | ~55 min | 66% |
| Premium-Sonnet | Qwen3-VL-32B + Sonnet ensemble | g6e.xlarge + API | **$7.10** | ~65 min | 72-75% |
| Premium-Opus | Qwen3-VL-32B + Opus ensemble | g6e.xlarge + API | **$11.40** | ~65 min | 73-76% |
| Premium-235B | Qwen3-VL-235B-A22B | p4d.24xlarge Spot | **$12.60** | ~80 min | 69-71% |

**Key insight**: The ensemble approach (Standard GPU + Claude on hard cases) is more
cost-effective AND more accurate than running a single larger model. The 235B MoE model
costs more and delivers lower accuracy than a Qwen3-VL-32B + Claude Sonnet ensemble.

---

## 7. Cold Start and Pre-Warming Strategy

### 7.1 GPU Instance Cold Start Breakdown

| Phase | Duration | Mitigation |
|-------|----------|------------|
| EC2 instance launch | 60-90s | Capacity reservation (costs $0.10/hr idle) |
| EBS volume attach | 10-20s | AMI-baked EBS with model weights |
| Docker image pull | 60-120s | Custom AMI with pre-pulled images |
| vLLM model loading | 90-180s | Pre-loaded weights on EBS, not S3 |
| vLLM warmup (first inference) | 10-30s | Health check in Step Functions wait loop |
| **Total cold start** | **3-5 min** | With AMI optimization |
| **Without optimization** | **8-12 min** | Pulling weights from S3 + fresh Docker pull |

### 7.2 AMI Strategy

Build a custom AMI monthly (or on model change) that includes:
- NVIDIA drivers + CUDA runtime
- Docker with vLLM image pre-pulled
- EBS snapshot with Qwen3-VL-32B-FP8 weights at `/models/`
- Python + pipeline worker code

**AMI build pipeline** (CodeBuild, monthly):
```
1. Launch g6e.xlarge
2. Install NVIDIA drivers, Docker
3. Pull vllm/vllm-openai:latest
4. Download Qwen3-VL-32B-FP8 from HuggingFace to EBS
5. Create AMI snapshot
6. Update Batch compute environment to use new AMI
```

Cost: ~$2/month (1 hour of g6e.xlarge for AMI build).

### 7.3 Pre-Warming Option

For low-latency requirements, keep a GPU instance warm between jobs:

- **EC2 instance with Hibernate**: Suspend to EBS, resume in ~60s. Costs EBS storage only
  while hibernated (~$0.08/GB-month for 100 GB = $8/month).
- **Scheduled warm-up**: If jobs follow a predictable pattern (e.g., weekend match processing),
  use EventBridge Scheduler to launch the GPU instance 5 min before expected job submission.
- **AWS Batch managed compute environment min-vCPU**: Set `minvCpus > 0` to keep one instance
  warm. Costs the full on-demand rate while idle -- only worth it at 3+ matches/day.

**Recommendation**: Accept the 3-5 min cold start. For a tool processing post-match video
(not live), 5 minutes of startup in a 55-minute pipeline is a negligible overhead.

---

## 8. Scaling: 1 to 10+ Concurrent Jobs

### 8.1 CPU Stages: Naturally Parallel

Fargate scales horizontally without configuration. 10 concurrent motion scan jobs simply
launch 10 Fargate tasks. The limiting factor is S3 throughput, which handles thousands of
concurrent requests without issue.

### 8.2 GPU Stages: Batch Compute Environment

AWS Batch managed compute environments scale GPU instances automatically:

```
Compute Environment Config:
  type: MANAGED
  computeResources:
    type: SPOT                          # Spot with on-demand fallback
    allocationStrategy: SPOT_PRICE_CAPACITY_OPTIMIZED
    maxvCpus: 40                        # Up to 10x g6e.xlarge (4 vCPU each)
    minvCpus: 0                         # Scale to zero
    instanceTypes:
      - g6e.xlarge                      # Primary: L40S 48GB
      - g5.xlarge                       # Fallback: A10G 24GB
      - g5.2xlarge                      # Fallback: A10G 24GB + more CPU
    ec2Configuration:
      imageIdOverride: ami-xxxxx        # Custom AMI with pre-baked weights
    spotIamFleetRole: arn:aws:iam::role/AmazonEC2SpotFleetRole
```

At 10 concurrent jobs, the system requests 10 GPU instances. AWS Batch handles Spot
capacity management, retrying with on-demand instances if Spot capacity is unavailable.

**GPU memory isolation**: Each job gets its own GPU instance. There is no GPU sharing --
vLLM on a g6e.xlarge uses the full L40S for one job. This is simpler and more reliable
than trying to share a large multi-GPU instance.

### 8.3 API Tier Scaling

The Budget tier with API calls scales trivially -- Lambda fan-out handles any concurrency.
The limit is the API provider's rate limit:
- Claude: default 4,000 requests/min (more than enough for 10 jobs x 220 calls)
- GPT-4o: default 10,000 requests/min

### 8.4 Cost at Scale (10 Concurrent Jobs)

| Tier | 1 Job | 10 Jobs (parallel) | 10 Jobs (sequential) |
|------|-------|-------------------|---------------------|
| Budget (Claude Batch) | $8.85 | $88.50 | $88.50 |
| Standard (GPU Spot) | $1.08 | $10.80 | $10.80 |
| Premium (Ensemble) | $7.10 | $71.00 | $71.00 |

Costs scale linearly. There are no shared-infrastructure savings at 10x because there is
no shared infrastructure to amortize -- every component is per-job.

---

## 9. Container Strategy

### 9.1 Docker Images

Three container images, stored in Amazon ECR:

| Image | Base | Contents | Size | Used By |
|-------|------|----------|------|---------|
| `pipeline-cpu` | python:3.11-slim + ffmpeg | Pipeline code, all Python deps except torch | ~800 MB | Fargate jobs (motion, audio, assembly) |
| `pipeline-gpu` | nvidia/cuda:12.x + python:3.11 + ffmpeg | Pipeline code, PyTorch CUDA, YOLO weights | ~4 GB | Batch EC2 jobs (VLM worker) |
| `vllm-server` | vllm/vllm-openai:latest | vLLM server (model weights on EBS, not in image) | ~8 GB | Batch EC2 jobs (VLM server sidecar) |

**ECR lifecycle policy**: Keep last 5 image tags, delete untagged images after 7 days.

### 9.2 Multi-Container Batch Job (Standard/Premium Tier)

The GPU Batch job runs two containers in the same task:

```yaml
# Batch job definition (conceptual)
containerProperties:
  # Container 1: vLLM server
  - name: vllm-server
    image: {account}.dkr.ecr.{region}.amazonaws.com/vllm-server:latest
    command: ["--model", "/models/Qwen3-VL-32B-FP8", "--max-model-len", "16384",
              "--gpu-memory-utilization", "0.95", "--max-num-seqs", "2"]
    resourceRequirements:
      - type: GPU
        value: 1
    mountPoints:
      - sourceVolume: model-weights
        containerPath: /models
    healthCheck:
      command: ["curl", "-f", "http://localhost:8000/health"]
      interval: 10
      retries: 30

  # Container 2: Pipeline worker
  - name: pipeline-worker
    image: {account}.dkr.ecr.{region}.amazonaws.com/pipeline-gpu:latest
    command: ["python", "-m", "src.batch.vlm_stage", "--job-id", "Ref::job_id"]
    environment:
      - name: VLLM_URL
        value: "http://localhost:8000"
      - name: S3_WORKING_BUCKET
        value: "pipeline-working"
    dependsOn:
      - containerName: vllm-server
        condition: HEALTHY
```

Note: AWS Batch multi-container jobs require Batch on ECS (not Fargate). The EC2 compute
environment handles this natively.

### 9.3 Build Pipeline

```
GitHub Actions / CodePipeline:
  on: push to main

  1. Build pipeline-cpu image → push to ECR
  2. Build pipeline-gpu image → push to ECR
  3. (Monthly) Build vllm-server image → push to ECR
  4. (Monthly) Build custom AMI with updated images + model weights
  5. Update Batch job definitions with new image tags
```

---

## 10. Reliability and Error Handling

### 10.1 Step Functions Retry Policies

```json
{
  "Retry": [
    {
      "ErrorEquals": ["Batch.JobFailed"],
      "IntervalSeconds": 60,
      "MaxAttempts": 2,
      "BackoffRate": 2.0,
      "Comment": "Retry failed Batch jobs (Spot interruption, OOM, etc.)"
    },
    {
      "ErrorEquals": ["Lambda.TooManyRequestsException"],
      "IntervalSeconds": 5,
      "MaxAttempts": 3,
      "BackoffRate": 1.5,
      "Comment": "Retry throttled Lambda invocations"
    }
  ],
  "Catch": [
    {
      "ErrorEquals": ["States.ALL"],
      "Next": "MarkJobFailed",
      "Comment": "On unrecoverable error, update DynamoDB and notify user"
    }
  ]
}
```

### 10.2 Spot Interruption Handling

When a Spot instance is interrupted mid-VLM-inference:
1. AWS Batch marks the job as FAILED with reason "Host EC2 instance terminated"
2. Step Functions retry policy relaunches the job
3. The VLM worker resumes from the last saved verdict in S3
   - Verdicts are written to S3 after each VLM call (append to JSONL)
   - On restart, worker reads existing verdicts and skips already-classified candidates
   - Worst case: lose 1 in-flight VLM call (~5s of work)

**Checkpoint granularity**: Per-VLM-call checkpointing to S3 adds ~100ms per call
(S3 PutObject for ~1 KB JSONL line). This is negligible compared to the 3-8s VLM latency.

### 10.3 Dead Letter and Alerting

- **DLQ**: Step Functions execution history provides built-in tracing. Failed executions
  trigger a CloudWatch alarm.
- **SNS notifications**: On job completion or failure, send email/webhook via SNS.
- **CloudWatch dashboards**: GPU utilization, VLM calls/min, S3 throughput, cost per job.

---

## 11. Security

### 11.1 Network

- **VPC**: All Batch jobs run in a private subnet with NAT gateway for outbound access.
- **S3 Gateway Endpoint**: Free, keeps S3 traffic off the internet.
- **No public IPs** on Batch instances. API Gateway is the only public endpoint.

### 11.2 Secrets

| Secret | Storage | Access |
|--------|---------|--------|
| Anthropic API key | Secrets Manager | Lambda execution role, Batch task role |
| OpenAI API key | Secrets Manager | Lambda execution role |
| HuggingFace token | Secrets Manager | AMI build pipeline only |

### 11.3 IAM

Least-privilege roles:
- **Lambda execution role**: S3 read/write on pipeline buckets, DynamoDB CRUD on job table,
  Step Functions StartExecution, Secrets Manager read.
- **Batch task role**: S3 read/write on pipeline buckets, ECR pull.
- **Batch service role**: EC2 launch/terminate, EBS attach, CloudWatch logs.

### 11.4 Video Data

- Videos are encrypted at rest (S3 SSE-S3 default encryption).
- Presigned URLs expire after 1 hour (upload) / 24 hours (download).
- Working bucket objects expire after 7 days (S3 lifecycle rule).
- No video data leaves the AWS region (except when calling external APIs -- Claude/GPT-4o
  receive individual frames, not full videos).

---

## 12. Migration Path from Current Architecture

### 12.1 Phase 1: Shared Code Refactoring (1-2 weeks)

The current pipeline code is tightly coupled to local filesystem paths. Refactor to
support S3 as a storage backend:

1. **Abstract storage layer**: Create `StorageBackend` protocol with `LocalStorage` and
   `S3Storage` implementations. Methods: `read_video()`, `write_frames()`,
   `read_candidates()`, `write_verdicts()`, `write_reel()`.
2. **Decouple VLM verifier**: The `VLMVerifier` currently extracts frames inline.
   Split into `FrameExtractor` (runs once, writes to storage) and `VLMClassifier`
   (reads frames from storage, calls VLM API). This enables the Lambda fan-out pattern.
3. **Replace Celery orchestration**: The worker's `_run_pipeline()` becomes a simple
   function that runs a single stage. Step Functions replaces Celery for stage sequencing.
   Keep Celery for local dev (no change to `make deploy`).

### 12.2 Phase 2: Cloud Infrastructure (1-2 weeks)

1. **CDK or Terraform**: Define all AWS resources as IaC.
   - S3 buckets, DynamoDB table, ECR repos
   - Lambda functions, API Gateway
   - Batch compute environments, job definitions
   - Step Functions state machine
   - IAM roles, VPC, security groups
2. **Build AMI pipeline**: CodeBuild project that builds the GPU AMI monthly.
3. **Deploy pipeline**: ECR push, Lambda deploy, Step Functions update.

### 12.3 Phase 3: Integration Testing (1 week)

1. Run the Rush test video through all three tiers.
2. Compare event detection results against local ground truth.
3. Validate cost estimates against actual AWS billing.

### 12.4 What Stays Local

The local development workflow (`make deploy`, `make test-unit`) remains unchanged.
Cloud deployment is additive, not a replacement:

- **Local**: Same as today. Celery + Redis + local FFmpeg + local/remote vLLM.
- **Cloud**: New deployment target. Same pipeline code, different orchestration and storage.
- **Hybrid**: Submit jobs via the cloud API, but point `VLLM_URL` at your home server's
  vLLM instance (requires port forwarding or VPN). $0 GPU cost, cloud-managed orchestration.

---

## 13. Tradeoff Analysis

### 13.1 Spot vs On-Demand for GPU

| Factor | Spot | On-Demand |
|--------|------|-----------|
| Cost | 60-70% cheaper ($0.65/hr vs $1.86/hr for g6e.xlarge) | Full price |
| Interruption risk | ~5% for 40-min window | None |
| Recovery | Auto-retry via Step Functions, lose ~5s of work | N/A |
| Availability | May be unavailable in some AZs at peak times | Always available |
| **Verdict** | **Use Spot with OD fallback** | Fallback only |

The pipeline's per-call checkpointing makes Spot interruptions nearly free. The retry adds
3-5 min (new instance cold start) but saves 60-70% on every job.

### 13.2 Self-Hosted VLM vs API

| Factor | Self-Hosted (GPU instance) | API (Claude/GPT-4o) |
|--------|--------------------------|---------------------|
| Cost per match | $0.50-1.50 (Spot) | $1-17 (model dependent) |
| Cold start | 3-5 min | None |
| Accuracy control | Full (same model as local, reproducible) | Dependent on provider versioning |
| Parallelism | Limited by VRAM (1-2 concurrent calls) | Highly parallel (50+ concurrent) |
| Wall-clock time | 30-40 min for VLM stage | 2-5 min for VLM stage (parallel) |
| Maintenance | AMI builds, model updates, CUDA compat | None |
| **Verdict** | **Best for cost and reproducibility** | **Best for speed and simplicity** |

The surprising finding is that self-hosted VLM on cloud GPU is cheaper than most API
options. The main advantage of APIs is speed (parallel calls) and zero maintenance.

### 13.3 Single Large Instance vs Multiple Smaller GPUs

| Factor | 1x g6e.xlarge per job | 1x g5.12xlarge shared |
|--------|----------------------|----------------------|
| Isolation | Full -- each job has its own GPU | Shared -- must partition GPU memory |
| Scaling | Linear (10 jobs = 10 instances) | Complex (vLLM multi-tenant, request queuing) |
| Cost efficiency | Slightly higher (per-instance overhead) | Better for 3+ concurrent jobs |
| Spot resilience | One interruption affects one job | One interruption affects all jobs |
| **Verdict** | **Use for 1-5 concurrent jobs** | Consider for 5+ concurrent steady-state |

For the expected workload (1-10 jobs, bursty), one instance per job is simpler and
more resilient.

### 13.4 Fargate vs EC2 for CPU Stages

| Factor | Fargate | EC2 (Batch managed) |
|--------|---------|---------------------|
| Cold start | 30-60s | 60-90s (instance launch) |
| Management | Zero (no AMI, no patching) | AMI lifecycle management |
| Cost | $0.04/vCPU-hr + $0.004/GB-hr | ~$0.02/vCPU-hr (Spot c6i) |
| Max storage | 200 GB ephemeral | Unlimited (EBS) |
| **Verdict** | **Use Fargate** for simplicity | Only if cost optimization needed at 50+ jobs/day |

Fargate is 2x more expensive per vCPU-hour than EC2 Spot, but the management simplicity
is worth it for CPU stages that cost <$0.50/job total.

---

## 14. Future Enhancements

### 14.1 Fine-Tuned Model Serving

When a fine-tuned Qwen3-VL-8B is available (per gpu-upgrade-analysis recommendations):
- Replace the g6e.xlarge (48 GB) with a g5.xlarge (24 GB) -- the 8B model fits easily
- GPU cost drops to ~$0.35/match (Spot)
- Inference speed doubles (8B vs 32B) -- VLM stage drops to ~15 min
- Accuracy potentially exceeds 32B on soccer-specific tasks

### 14.2 Streaming / Near-Real-Time

For live match processing (results within minutes of the event):
1. Replace S3 inter-stage storage with Kinesis Data Streams
2. Run motion scan as a continuous ECS service during the match
3. VLM calls fire immediately on candidate detection
4. Requires always-on GPU instance during the match (~$1.86/hr for 2 hours = $3.72)

### 14.3 Multi-Region

For international users with high-bandwidth video uploads:
- S3 Transfer Acceleration for upload
- S3 Cross-Region Replication to processing region
- Or: CloudFront upload endpoint with Lambda@Edge routing

### 14.4 Web Application Frontend

A simple React/Next.js frontend hosted on Amplify or S3+CloudFront:
- Upload page with drag-and-drop, multipart upload progress bar
- Team color picker (reuse existing Open-WebUI kit selection UI pattern)
- Job status dashboard (polling DynamoDB via API Gateway)
- Reel preview and download
- Cost: ~$5/month for hosting (Amplify) + API Gateway costs

---

## 15. Implementation Checklist

### IaC Resources (CDK/Terraform)

- [ ] S3: ingest bucket, working bucket, output bucket (lifecycle policies)
- [ ] DynamoDB: jobs table (job_id PK, status GSI)
- [ ] ECR: pipeline-cpu, pipeline-gpu, vllm-server repositories
- [ ] Lambda: job-submit, upload-presign, structural-inference, notify
- [ ] API Gateway: HTTP API with Lambda integrations
- [ ] Step Functions: pipeline state machine (three tier variants)
- [ ] Batch: CPU compute env (Fargate), GPU compute env (EC2 Spot)
- [ ] Batch: job definitions for each stage
- [ ] IAM: Lambda role, Batch task role, Batch service role
- [ ] VPC: private subnet, NAT gateway, S3 gateway endpoint
- [ ] Secrets Manager: API keys
- [ ] CloudWatch: dashboard, alarms
- [ ] SNS: job completion notifications
- [ ] CodeBuild: AMI build pipeline, ECR image pipeline

### Code Changes

- [ ] `StorageBackend` protocol with `LocalStorage` and `S3Storage`
- [ ] `FrameExtractor` split from `VLMVerifier`
- [ ] `src/batch/` module with per-stage entry points
- [ ] Lambda handlers in `src/lambda/`
- [ ] Step Functions state machine definition (ASL JSON)
- [ ] Batch job definitions
- [ ] Per-call VLM checkpointing (already partially exists)
- [ ] DynamoDB JobStore implementation
- [ ] CLI tool for cloud job submission (`pipeline_cli.py cloud submit`)

### Testing

- [ ] Unit tests for S3Storage backend (moto)
- [ ] Integration test: full pipeline on a 30s synthetic video
- [ ] Cost validation: run Rush test video, compare to estimates
- [ ] Spot interruption test: manually terminate GPU instance mid-job
- [ ] Concurrent jobs test: submit 3 jobs simultaneously

---

## 16. Appendix: Detailed AWS Service Pricing Reference

All prices are us-east-1, as of April 2026. Prices may vary by region.

### Compute

| Service | Unit | Price |
|---------|------|-------|
| Fargate (Linux, x86) | per vCPU-hour | $0.04048 |
| Fargate (Linux, x86) | per GB-hour | $0.004445 |
| Fargate ephemeral storage | per GB-hour (above 20 GB) | $0.000111 |
| g5.xlarge (1x A10G 24GB) | per hour, on-demand | $1.006 |
| g5.2xlarge (1x A10G 24GB) | per hour, on-demand | $1.212 |
| g5.12xlarge (4x A10G 96GB) | per hour, on-demand | $5.672 |
| g6e.xlarge (1x L40S 48GB) | per hour, on-demand | $1.861 |
| p4d.24xlarge (8x A100 40GB) | per hour, on-demand | $32.77 |
| Lambda | per GB-second | $0.0000166667 |
| Lambda | per request | $0.0000002 |

### Storage

| Service | Unit | Price |
|---------|------|-------|
| S3 Standard | per GB-month | $0.023 |
| S3 PUT/POST | per 1,000 requests | $0.005 |
| S3 GET | per 1,000 requests | $0.0004 |
| S3 data transfer out | per GB (first 100 TB) | $0.09 |
| DynamoDB write | per WCU-hour (on-demand: per write) | $1.25 per million |
| DynamoDB read | per RCU-hour (on-demand: per read) | $0.25 per million |

### Other

| Service | Unit | Price |
|---------|------|-------|
| Step Functions | per state transition | $0.000025 |
| API Gateway (HTTP API) | per million requests | $1.00 |
| Secrets Manager | per secret/month | $0.40 |
| SNS | per notification | $0.00 (first 1M free) |
| ECR | per GB-month storage | $0.10 |
| CloudWatch Logs | per GB ingested | $0.50 |

---

## Sources

- [AWS EC2 On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
- [g5.xlarge specs and pricing](https://instances.vantage.sh/aws/ec2/g5.xlarge)
- [g6e.xlarge specs and pricing](https://instances.vantage.sh/aws/ec2/g6e.xlarge)
- [AWS Fargate Pricing](https://aws.amazon.com/fargate/pricing/)
- [AWS Step Functions + Batch Integration](https://docs.aws.amazon.com/step-functions/latest/dg/connect-batch.html)
- [Video Processing on AWS with Step Functions and Batch](https://andrii-shykhov.medium.com/video-processing-workflow-on-aws-with-step-functions-and-batch-6a5f29e4055a)
- [S3 Multipart Upload with Presigned URLs](https://aws.amazon.com/blogs/compute/uploading-large-objects-to-amazon-s3-using-multipart-upload-and-transfer-acceleration/)
- [AWS Batch GPU Jobs](https://docs.aws.amazon.com/batch/latest/userguide/gpu-jobs.html)
- [GPU Upgrade Analysis (internal)](gpu-upgrade-analysis.md)
- [Claude API Pricing](https://platform.claude.com/docs/en/about-claude/pricing)

---

## Architecture Review -- AWS Expert Assessment

**Reviewer**: AWS Cloud Architect (all certifications, 15+ years production)
**Date**: 2026-04-03
**Overall Grade**: B+ (solid foundation, several critical gaps to address before production)

---

### Findings Summary Table

| # | Finding | Severity | Section | Recommendation |
|---|---------|----------|---------|----------------|
| 1 | NAT Gateway cost not budgeted | **CRITICAL** | Cost | NAT Gateway is $0.045/GB processed + $0.045/hr (~$32/month baseline). For a "scale to zero" architecture, this is the largest hidden fixed cost. |
| 2 | No WAF on API Gateway | **CRITICAL** | Security | Public API with presigned URL generation is an abuse vector. Add WAF with rate limiting. |
| 3 | Fargate ephemeral storage insufficient for source video | **HIGH** | Storage | Motion scan needs the full 5-15 GB source video. Streaming via S3 presigned URL with FFmpeg `-ss` seeking works for sequential access but the doc contradicts itself (says "stream" then "download first"). Clarify and test. |
| 4 | No VPC endpoint for ECR, CloudWatch, Secrets Manager | **HIGH** | Network | Every Docker pull, log push, and secret fetch goes through NAT Gateway at $0.045/GB. A 4 GB image pull = $0.18 in NAT charges alone. Interface VPC endpoints eliminate this. |
| 5 | g6e Spot availability is poor | **HIGH** | Compute | g6e instances (L40S) are new and have limited Spot capacity. The fallback list (g5.xlarge, g5.2xlarge) is good but the A10G 24 GB VRAM constraint with max-model-len 8192 is a material accuracy risk. |
| 6 | Multi-container Batch job is fragile | **HIGH** | Compute | AWS Batch multi-container on ECS has limited health-check integration. If vLLM OOMs or crashes, the pipeline worker may hang. Need a sidecar supervisor pattern. |
| 7 | No S3 bucket policy restricting presigned URL scope | **HIGH** | Security | Presigned URLs should be scoped to specific key prefixes. Without bucket policies, a compromised presigned URL could access unrelated data. |
| 8 | Missing CloudWatch Logs VPC endpoint | **MEDIUM** | Network | Batch and Fargate logs go through NAT. For GPU jobs producing verbose vLLM logs, this adds cost. |
| 9 | DynamoDB on-demand pricing for job table is fine but no TTL configured | **MEDIUM** | Storage | Job records will accumulate forever. Add TTL to auto-expire completed jobs after 90 days. |
| 10 | No cost alerting or AWS Budgets configured | **MEDIUM** | Cost | A runaway Step Functions loop or stuck GPU instance could generate unbounded charges. |
| 11 | Step Functions Express vs Standard not evaluated | **MEDIUM** | Orchestration | Standard workflows cost $0.025/1000 transitions but have a 1-year execution limit. Express workflows cost by duration but cap at 5 minutes. The current choice is correct for long-running jobs, but should be explicit. |
| 12 | AMI build pipeline has no rollback mechanism | **MEDIUM** | Operations | If a new AMI has a broken CUDA driver or corrupted model weights, all GPU jobs fail. Need AMI versioning and a canary deployment. |
| 13 | No X-Ray tracing configured | **LOW** | Observability | Step Functions + Lambda + Batch distributed tracing is hard to debug without X-Ray. |
| 14 | Presigned URL expiry too generous for download | **LOW** | Security | 24-hour download URLs are long. Consider 4-hour URLs with a refresh endpoint. |
| 15 | No consideration of Graviton (ARM) for CPU stages | **LOW** | Cost | Fargate on ARM (Graviton) is 20% cheaper than x86. FFmpeg and Python work fine on ARM. |

---

### 1. Architecture Critique

#### What Is Well-Designed

**Step Functions + AWS Batch is the correct choice.** The document correctly identifies that the pipeline's stage-based, DAG-like execution with conditional branching (tier selection) maps perfectly to Step Functions. The `.sync` integration pattern for Batch is exactly right -- it avoids polling and simplifies error handling. The comparison table against SQS+ECS and standalone Batch is accurate and well-reasoned.

**The three-tier pricing model is genuinely clever.** Budget (API-only), Standard (self-hosted VLM), and Premium (ensemble) match different cost/accuracy/speed tradeoffs. The insight that the Standard tier at $1-2/match is cheaper than API calls is correct and non-obvious. Most architects would default to "just use the API" without running the GPU cost math.

**S3 over EFS for inter-stage data is correct.** The analysis that sequential, write-once/read-once access patterns favor S3 over EFS is right. EFS would add $0.30/GB-month plus throughput charges for no benefit. The one exception (frame extraction to local disk for random access) is correctly identified.

**Per-call VLM checkpointing to S3 makes Spot viable.** Losing at most 5 seconds of work on a Spot interruption is excellent design. This is the kind of detail that separates production architectures from whiteboard exercises.

**Presigned multipart upload for large video files.** Correct approach. Avoids Lambda payload limits, enables parallel upload parts, and keeps the video out of the API layer entirely.

#### What Is Missing or Wrong

**1. NAT Gateway is the elephant in the room.** The entire document describes a "scale to zero" architecture, but the VPC design requires a NAT Gateway for private subnet outbound access (S3 API calls from Batch, ECR pulls, CloudWatch Logs, Secrets Manager, external VLM API calls). A NAT Gateway costs **$0.045/hour ($32.40/month)** just to exist, plus $0.045/GB of data processed. For a pipeline processing 2-4 matches/month, the NAT Gateway alone could be the single largest line item.

**Mitigation options (pick one)**:
- **VPC endpoints everywhere + public subnet for Batch**: Use S3 Gateway Endpoint (free), plus Interface VPC Endpoints for ECR ($0.01/hr each, 2 needed: `ecr.api` + `ecr.dkr`), CloudWatch Logs ($0.01/hr), and Secrets Manager ($0.01/hr). Total: ~$0.04/hr = $28.80/month. This is not cheaper than NAT Gateway for light workloads.
- **Run Batch in a public subnet with security groups**: Batch EC2 instances can run in a public subnet with auto-assigned public IPs. Security groups restrict inbound to nothing (no public ports needed). Outbound goes directly to the internet. This eliminates NAT Gateway entirely. Fargate tasks can also run in public subnets with `assignPublicIp: ENABLED`. **This is the right choice for a low-volume pipeline.** The "private subnet + NAT" pattern is security theater when the instances have no inbound services.
- **Hybrid**: Public subnet for Batch/Fargate tasks + S3 Gateway Endpoint (free). No NAT Gateway, no interface endpoints. Total additional network cost: $0/month.

**Recommendation**: Use public subnets + security groups + S3 Gateway Endpoint. Add NAT Gateway only if compliance requirements mandate private subnets (e.g., HIPAA, SOC2 -- unlikely for a soccer video tool).

**2. The Fargate video streaming claim needs validation.** Section 5.3 says "stream via S3 presigned URL" for motion scan, but then section 3.1 says Fargate needs 100 GB ephemeral storage for frame extraction. The actual behavior of `ffmpeg -ss <timestamp> -i https://s3-presigned-url` over HTTP depends on whether the S3 presigned URL supports byte-range requests (it does). However, FFmpeg's HTTP seeking is unreliable for large files with many random seeks. For motion scan (sequential at 0.5s intervals), streaming works. For frame extraction at 220 random timestamps, it will be slow and may fail on network hiccups. The document correctly recommends downloading to local storage for frame extraction but does not account for the 5-15 GB download time in the Fargate task duration estimate (15 min for motion scan should include the transfer time).

**3. Batch multi-container health check timing.** The document describes a `dependsOn: HEALTHY` relationship between the vLLM server and the pipeline worker. In practice, AWS Batch multi-container health checks have a minimum interval of 5 seconds and a maximum of 300 retries. With vLLM cold start at 90-180s, setting `interval: 10, retries: 30` means the health check passes after at most 300 seconds (5 min). But if vLLM takes longer (e.g., first-time FP8 kernel compilation on a new GPU type), the worker will start before vLLM is ready, or the health check will time out and the entire job will fail. **Add a readiness probe in the pipeline worker code** that polls `localhost:8000/health` with exponential backoff, independent of the Batch health check.

**4. No consideration of ECS Capacity Providers as an alternative to Batch.** AWS Batch is the right choice for this workload, but it is worth noting that ECS with Capacity Providers (including GPU-enabled EC2 capacity providers) can achieve the same result with finer control over placement and scaling. For this project's complexity, Batch is simpler and correct.

#### Service Choices Assessment

| Service Choice | Verdict | Notes |
|---------------|---------|-------|
| Step Functions (Standard) | Correct | Long-running jobs (up to 1 year execution). Express would not work for 30-60 min pipelines. |
| AWS Batch (GPU) | Correct | Only option for GPU + Spot + auto-scaling. |
| Fargate (CPU) | Correct | Simpler than EC2 for CPU stages. 200 GB ephemeral storage is enough. |
| API Gateway HTTP API | Correct | Lower latency, lower cost than REST API. No need for REST API features (API keys, usage plans) here since auth is handled differently. |
| DynamoDB | Correct | Perfect for job status with GSI on status. Pay-per-request mode avoids idle cost. |
| S3 for inter-stage | Correct | As analyzed in the doc. |
| Lambda for structural inference | Correct | Pure Python, <1 min runtime, no state needed. |
| ECR for container images | Correct | Only practical option for Batch/Fargate. |

---

### 2. Cost Validation

#### GPU Instance Pricing

The on-demand prices in Section 16 are **accurate for us-east-1 as of early 2026**:

| Instance | Doc Price | Verified Price | Delta |
|----------|-----------|----------------|-------|
| g5.xlarge | $1.006/hr | $1.006/hr | Correct |
| g5.12xlarge | $5.672/hr | $5.672/hr | Correct |
| g6e.xlarge | $1.861/hr | $1.861/hr | Correct |
| p4d.24xlarge | $32.77/hr | $32.77/hr | Correct |

**Spot price estimates are reasonable but volatile.** The doc estimates g6e.xlarge Spot at ~$0.65/hr (65% discount). Historical g6e Spot discounts in us-east-1 range from 50-75% off on-demand. The $0.65 estimate is within range, but Spot prices can spike to near on-demand during periods of high demand (ML training runs, month-end). **Budget at 60% discount ($0.74/hr) for planning, not 65%.**

#### Missing Cost Items

| Item | Monthly Cost | Per-Match Impact | Notes |
|------|-------------|------------------|-------|
| NAT Gateway (if used) | $32.40 fixed + data | ~$8-16/match at 2-4 matches/month | **Largest hidden cost** |
| ECR storage (3 images) | ~$1.30/month | Negligible | ~13 GB total across 3 images |
| CloudWatch Logs ingestion | $0.50/GB | ~$0.05-0.25/match | vLLM produces verbose logs; set log level to WARN |
| Secrets Manager (3 secrets) | $1.20/month | Negligible | $0.40/secret/month |
| EBS snapshot for AMI | ~$0.40/month | Negligible | 100 GB gp3 snapshot at $0.05/GB-month (only ~8 GB changed blocks) |
| S3 multipart upload fees | ~$0.05/match | Included in doc | Correct |

**Total hidden infrastructure cost**: ~$35/month if using NAT Gateway, ~$3/month without it.

#### Cost Estimate Corrections

The doc's per-match cost estimates are largely accurate. One correction:

**Standard tier is slightly underestimated.** The doc says $1.08 (Spot) but does not include:
- S3 data transfer for source video download by Batch job (~$0 because S3 Gateway Endpoint keeps it in-VPC)
- EBS costs for ephemeral volumes ($0.08/GB-month for gp3, but prorated to 40 minutes = ~$0.004 -- negligible)
- CloudWatch Logs from vLLM (~$0.05)

Corrected Standard tier: **$1.13 Spot / $2.05 On-Demand** (the difference is negligible).

**Budget tier Claude cost is accurate.** The doc correctly revised the token estimate from the initial 200 tokens/image to 1,600 tokens/image for Claude. The $8.25 Batch / $16.50 real-time estimates for Claude Sonnet 4.6 are correct.

#### Spot Instance Strategy

The Spot strategy is **sound but needs a Spot Capacity Diversification enhancement**:

1. **Current approach**: `SPOT_PRICE_CAPACITY_OPTIMIZED` allocation strategy with instance type fallback list. This is the correct allocation strategy for short-duration jobs.

2. **Missing: multi-AZ Spot capacity**. The Batch compute environment should specify **all AZs in the region** (not just one) to maximize Spot pool diversity. GPU Spot capacity varies significantly by AZ.

3. **Missing: Spot interruption notification handler.** AWS sends a 2-minute warning before Spot termination. The VLM worker should listen for the EC2 metadata service interruption notice (`http://169.254.169.254/latest/meta-data/spot/instance-action`) and flush the current verdict to S3 before termination. This saves the cost of re-running the in-flight VLM call.

4. **Missing: Spot max price cap.** Without a `maxPrice` in the Batch compute environment, AWS can charge up to the on-demand price for Spot instances during capacity crunches. Set `maxPrice` to 70% of on-demand to avoid surprise charges.

#### Reserved Capacity / Savings Plans

**Not recommended at current scale.** Savings Plans and Reserved Instances require 1-year or 3-year commitments. At 2-4 matches/month with 40 minutes of GPU time each, the total GPU usage is ~2-3 hours/month. No Savings Plan tier justifies this. Revisit when processing 50+ matches/month consistently.

The compute Savings Plans ($0.001/hr minimum commit) could save on Lambda and Fargate, but the savings would be under $1/month. Not worth the commitment.

---

### 3. Operational Concerns

#### Cold Start Mitigation

The AMI-baked approach (pre-pulled Docker images + model weights on EBS) is the correct mitigation. The 3-5 minute estimate is realistic for vLLM model loading from local EBS. A few improvements:

**Add a "warm pool" for GPU instances.** EC2 warm pools (introduced 2021) can keep pre-initialized instances in a `Stopped` state. A stopped g6e.xlarge incurs only EBS storage costs (~$8/month for 100 GB gp3). On launch, the instance resumes from the stopped state in ~30 seconds instead of the full 3-5 minute cold start. Configure the Batch compute environment's underlying Auto Scaling Group to maintain a warm pool of 1 instance if processing frequency exceeds 1 match/day.

**EC2 Hibernate (mentioned in the doc) is a better option but requires validation.** Hibernation preserves the in-memory state of vLLM, meaning the model is already loaded on resume (~60s startup). However, GPU instances with large VRAM may have issues with CUDA context restoration after hibernate. **Test this with g6e.xlarge before relying on it.**

#### Spot Interruption Recovery

The doc describes this well: Step Functions retry launches a new instance, the VLM worker resumes from the last checkpoint. However:

**The 3-5 minute cold start on retry is the real penalty.** The doc says "lose ~5s of work" but the wall-clock penalty is 3-5 minutes (new instance cold start). For a 55-minute pipeline, this is ~6-9% overhead. Acceptable, but should be tracked as a metric.

**What if Spot capacity is unavailable for the retry?** The Batch compute environment's on-demand fallback handles this, but the fallback adds cost. Track the metric "percentage of jobs falling back to on-demand" to catch capacity issues early.

#### Monitoring and Alerting Gaps

The doc mentions "CloudWatch dashboards" and SNS notifications but does not specify:

1. **GPU utilization metric collection.** Batch EC2 instances need the CloudWatch Agent or NVIDIA DCGM Exporter to publish GPU memory utilization and GPU compute utilization. Without this, you cannot detect OOM conditions or underutilized GPU instances.

2. **VLM inference latency tracking.** The pipeline worker should publish a custom CloudWatch metric for per-call VLM latency. A latency spike (e.g., >15s per call) indicates VRAM pressure or model loading issues.

3. **Step Functions execution duration alarm.** Set a CloudWatch alarm on execution duration exceeding 2x the expected time (e.g., 120 minutes for Standard tier). This catches stuck jobs.

4. **S3 storage accumulation alarm.** The working bucket should have a CloudWatch alarm on total size exceeding a threshold (e.g., 500 GB). This catches orphaned job data from failed cleanup.

5. **Cost anomaly detection.** Enable AWS Cost Anomaly Detection on the GPU instance spend category. This catches runaway instances.

**Recommended CloudWatch alarms** (minimum):

| Alarm | Threshold | Action |
|-------|-----------|--------|
| Step Functions execution failed | >0 in 5 min | SNS email |
| Step Functions execution duration | >120 min | SNS email |
| Batch job stuck in RUNNABLE | >15 min (indicates no capacity) | SNS email |
| GPU instance count | >10 (unexpected scale) | SNS email |
| Monthly GPU spend | >$50 (budget guard) | SNS email + auto-terminate |
| S3 working bucket size | >500 GB | SNS email |

#### Log Aggregation

The doc does not describe a log aggregation strategy. Recommendations:

- **Fargate tasks**: Logs go to CloudWatch Logs automatically via `awslogs` log driver. No config needed.
- **Batch EC2 tasks**: Must configure the `awslogs` log driver in the job definition. This is not automatic.
- **vLLM server container**: Produces verbose logs. Set `--log-level warning` in the vLLM command to reduce CloudWatch Logs ingestion cost ($0.50/GB). Debug logs only when troubleshooting.
- **Lambda functions**: Logs go to CloudWatch automatically. Enable structured logging (JSON) for easier querying.
- **Log retention**: Set CloudWatch Logs retention to 30 days for all log groups. Default is "never expire" which accumulates cost indefinitely.

---

### 4. Security Review

#### IAM Least-Privilege

The doc describes three roles (Lambda, Batch task, Batch service) at a high level. Specific gaps:

1. **Lambda execution role is too broad.** "S3 read/write on pipeline buckets" should be scoped to specific key prefixes. The upload Lambda should only write to `s3://ingest-bucket/*`, not the working or output buckets. The structural inference Lambda should only read/write `s3://working-bucket/{job_id}/*`.

2. **Missing condition keys.** Add `aws:RequestedRegion` condition to prevent cross-region resource creation. Add `s3:prefix` conditions to S3 permissions.

3. **Batch task role needs ECR pull permission.** The doc lists "ECR pull" but the Batch service role (not task role) handles instance-level ECR pulls. The task role needs only S3 access and Secrets Manager access. Clarify this.

4. **Missing: Step Functions execution role.** Step Functions needs an IAM role to invoke Lambda, submit Batch jobs, and send SNS notifications. This role should be scoped to specific Lambda function ARNs and Batch job queue ARNs.

5. **Missing: S3 bucket policies.** Add bucket policies that deny access from outside the VPC (using `aws:sourceVpc` condition) for the working and output buckets. The ingest bucket needs public PUT access via presigned URLs, but should deny direct access without presigned auth.

#### Data Encryption

- **At rest**: S3 SSE-S3 (mentioned in doc) is correct. Consider SSE-KMS if the customer requires audit trails of key usage via CloudTrail.
- **In transit**: All S3 access via HTTPS by default. Batch-to-S3 via VPC endpoint is encrypted. External API calls (Claude, GPT-4o) are HTTPS. **No gaps found.**
- **EBS encryption**: The AMI's EBS volumes should use EBS encryption (default KMS key). This is not mentioned in the doc.
- **DynamoDB encryption**: Enabled by default (AWS owned key). No action needed.

#### Network Isolation

As discussed in the cost section, the VPC design needs adjustment. The current design (private subnet + NAT Gateway) is correct from a security perspective but expensive. If moving to public subnets:

- **Security groups**: Batch instances: allow all outbound, deny all inbound. API Gateway: no security group needed (fully managed).
- **NACLs**: Default NACLs are sufficient (allow all). Custom NACLs add complexity without meaningful security benefit for this workload.
- **Missing: S3 Gateway Endpoint policy.** The S3 Gateway Endpoint should have a policy restricting access to only the pipeline's S3 buckets. This prevents Batch instances from accessing unrelated S3 data.

#### API Authentication

**The doc does not describe API authentication.** This is a significant gap. Recommendations:

1. **For the MVP**: API Gateway with API keys (simple, built-in rate limiting). Distribute API keys to authorized users.
2. **For production**: Amazon Cognito User Pool + API Gateway authorizer. Supports user registration, password reset, MFA. Or use a simpler approach with Lambda authorizer + static bearer tokens stored in Secrets Manager.
3. **Presigned URL abuse prevention**: The `/upload/init` endpoint generates presigned S3 URLs. Without authentication, anyone can call this endpoint to generate upload URLs and fill the S3 bucket with junk data. **This must be behind authentication.**

#### Secrets Management

Secrets Manager usage is correct. Add:
- **Automatic rotation** for API keys (Secrets Manager supports Lambda-based rotation).
- **Resource-based policies** on secrets to restrict access to specific IAM roles.
- **Do not log secret values.** Ensure Lambda and Batch task code never logs API keys. Use `structlog`'s value filtering or `aws-lambda-powertools` to scrub secrets from logs.

---

### 5. Scaling Concerns

#### 10 Concurrent Jobs

Works fine. 10 Fargate tasks scale instantly. 10 GPU instances may take 2-5 minutes to provision depending on Spot capacity. The S3 throughput is not a bottleneck (S3 handles 5,500 GET/s and 3,500 PUT/s per prefix partition).

**Potential issue**: 10 GPU instances x 35 GB model weights per AMI = 350 GB of EBS snapshots being restored simultaneously. EBS snapshot restoration can have throttled throughput for the first access (cold reads). Use EBS Fast Snapshot Restore ($0.75/AZ/snapshot/hour) to avoid this. At 10 concurrent jobs, this costs $0.75/hr for the snapshot -- negligible.

#### 50 Concurrent Jobs

**GPU Spot capacity becomes the bottleneck.** A single AZ in us-east-1 may have only 10-20 g6e.xlarge Spot instances available at any time. At 50 concurrent jobs:

- Spread across 6 AZs in us-east-1: ~60-120 g6e instances available
- If g6e is unavailable, fall back to g5.xlarge across AZs
- If both are unavailable, fall back to on-demand
- Set Batch compute environment `maxvCpus` to 200 (50 x 4 vCPU per g6e.xlarge)
- **Budget for 20-30% on-demand fallback at this scale**

**S3 request rate**: 50 jobs x 220 frame PUTs = 11,000 PUTs in a burst. S3 handles this easily with automatic partition scaling, but use randomized key prefixes (the `{job_id}` UUID provides this naturally).

**DynamoDB**: On-demand mode handles 50 concurrent writers without issue. No capacity concerns.

**Step Functions**: 1,000 concurrent executions per account (soft limit, raisable). 50 is well within limits.

#### 100 Concurrent Jobs

At this scale, consider:

1. **Service quotas**: Request limit increases for Batch compute environments (default: 50 compute environments, but each can have many instances), Step Functions concurrent executions, and GPU instance limits (default vCPU quota for G instances is 128 in us-east-1 -- 100 g6e.xlarge would need 400 vCPU).
2. **Multi-region deployment**: Spread jobs across us-east-1 and us-west-2 for GPU capacity diversity.
3. **Shared vLLM server**: At 100 concurrent jobs, running 100 separate vLLM instances is wasteful. Deploy a shared vLLM cluster on ECS with GPU-backed tasks, fronted by an NLB. Each pipeline job sends VLM calls to the shared cluster. This requires multi-tenant request routing but amortizes model loading across jobs.
4. **API tier becomes dominant**: At 100 jobs, the Budget tier with Claude API becomes more practical because there is no GPU capacity constraint. 100 jobs x 220 calls = 22,000 concurrent API calls. Claude's rate limits (4,000/min) would serialize this to ~5.5 minutes. GPT-4o's limits (10,000/min) are more generous.

#### GPU Instance Availability by Region

| Region | g6e.xlarge Spot Availability | g5.xlarge Spot Availability | Recommendation |
|--------|------------------------------|------------------------------|----------------|
| us-east-1 | Good (6 AZs, high capacity) | Excellent | Primary region |
| us-west-2 | Good (4 AZs) | Excellent | Secondary region |
| eu-west-1 | Moderate (3 AZs) | Good | For EU customers |
| ap-northeast-1 | Limited | Good | Avoid for GPU workloads |

---

### 6. Missing Components

#### 6.1 CI/CD Pipeline

The doc mentions CodeBuild for AMI builds and "GitHub Actions / CodePipeline" for container builds but does not describe a full CI/CD pipeline. Recommended architecture:

```
GitHub Actions (existing):
  on: push to main
  1. Run unit tests (pytest, no AWS needed)
  2. Build + push Docker images to ECR
  3. Update Batch job definitions (aws batch register-job-definition)
  4. Update Lambda function code (aws lambda update-function-code)
  5. Update Step Functions state machine (aws stepfunctions update-state-machine)

  on: schedule (monthly)
  6. Build GPU AMI via CodeBuild
  7. Update Batch compute environment AMI ID

  on: release tag
  8. Full integration test on AWS (submit synthetic video, validate output)
```

Use AWS CDK deploy (`cdk deploy`) or Terraform apply in the CI/CD pipeline rather than raw AWS CLI commands. This ensures infrastructure drift is caught.

#### 6.2 Infrastructure as Code

**Recommendation: AWS CDK (TypeScript or Python).** CDK is preferred over Terraform for this project because:
- Native Step Functions ASL generation from CDK constructs (no hand-writing JSON state machines)
- Native Batch job definition constructs
- L2 constructs for Lambda, API Gateway, S3 that handle IAM automatically
- The team is already Python-native; CDK Python is a natural fit

If the team prefers Terraform, use the `aws` provider with the `step_functions` resource. The state machine definition still requires ASL JSON but can be templated with Terraform variables.

**Do not use CloudFormation directly.** The Step Functions + Batch integration requires complex IAM role chaining that CDK handles automatically but CloudFormation requires manual wiring.

Skeleton CDK stack structure:
```
infra/cdk/
  app.py                    # CDK app entry point
  stacks/
    network_stack.py        # VPC, subnets, S3 gateway endpoint
    storage_stack.py        # S3 buckets, DynamoDB table
    compute_stack.py        # Batch compute environments, job definitions
    api_stack.py            # API Gateway, Lambda functions
    pipeline_stack.py       # Step Functions state machine
    monitoring_stack.py     # CloudWatch dashboards, alarms, SNS
    ci_stack.py             # CodeBuild for AMI, ECR lifecycle
```

#### 6.3 Disaster Recovery

For a non-critical batch processing tool, full multi-region DR is overkill. Minimum DR:

1. **S3 Cross-Region Replication** for the output bucket only (reels). Working and ingest buckets do not need replication.
2. **DynamoDB Point-in-Time Recovery** enabled (costs $0.20/GB-month on the backup storage). Allows restoring the job table to any point in the last 35 days.
3. **ECR Cross-Region Replication** for container images. Free (you pay only for the storage in the second region).
4. **CDK/Terraform state**: Store in S3 with versioning enabled. The IaC definition itself is the DR plan -- `cdk deploy` in a new region recreates everything.

**RPO**: Minutes (S3 CRR is near-real-time).
**RTO**: 30-60 minutes (deploy CDK stack in new region + build AMI).

#### 6.4 Cost Alerting and Budget Controls

**Mandatory before production**:

1. **AWS Budgets**: Create a monthly budget of $100 with an 80% threshold alert ($80). This catches runaway GPU instances.
2. **Cost Anomaly Detection**: Enable on the EC2 GPU and Lambda spend categories. Alerts when daily spend deviates from the trailing 10-day average by >$10.
3. **Batch compute environment `maxvCpus`**: Already mentioned in the doc (40 vCPUs = 10 instances). This is a hard ceiling that prevents unbounded GPU scaling.
4. **Lambda concurrency limit**: Set reserved concurrency on the VLM fan-out Lambda to 50. This prevents a bug in the Step Functions Map state from launching thousands of concurrent Lambda invocations.
5. **S3 Lifecycle policies**: Already mentioned (7-day expiry on working bucket). Add an additional rule: delete multipart upload fragments after 1 day (orphaned multipart uploads are a common silent cost leak).

---

### 7. Specific Recommendations with Configurations

#### 7.1 Revised VPC Architecture (No NAT Gateway)

```
VPC: 10.0.0.0/16
  Public Subnet AZ-a: 10.0.1.0/24  (Batch EC2, Fargate tasks)
  Public Subnet AZ-b: 10.0.2.0/24  (Batch EC2, Fargate tasks)
  Public Subnet AZ-c: 10.0.3.0/24  (Batch EC2, Fargate tasks)

  S3 Gateway Endpoint (free): attached to all route tables
  No NAT Gateway. No private subnets.

  Security Groups:
    batch-gpu-sg:   inbound: none, outbound: all
    batch-cpu-sg:   inbound: none, outbound: all

  Fargate tasks: assignPublicIp: ENABLED
  Batch EC2: auto-assign public IP via subnet setting
```

Monthly network cost: **$0** (vs $32.40+ with NAT Gateway).

If compliance requires private subnets later, add them with a NAT Gateway and move Batch to private subnets. The CDK stack should parameterize this.

#### 7.2 Graviton for CPU Stages

Switch Fargate CPU tasks to ARM64 for a 20% cost reduction:

| Stage | Current (x86) | Graviton (ARM) | Savings |
|-------|---------------|----------------|---------|
| Motion scan | $0.17/job | $0.136/job | 20% |
| Audio analysis | $0.04/job | $0.032/job | 20% |
| Frame extraction | $0.17/job | $0.136/job | 20% |
| Clip assembly | $0.06/job | $0.048/job | 20% |
| **Total CPU** | **$0.44** | **$0.352** | **$0.09/job** |

Requirements: Build the `pipeline-cpu` Docker image for `linux/arm64`. FFmpeg and Python 3.11 both support ARM64 natively. Use Docker buildx for multi-arch images.

The savings are small in absolute terms but the effort is minimal (one Dockerfile change).

#### 7.3 Recommended Batch Compute Environment

```python
# CDK pseudo-code for the GPU compute environment
gpu_compute_env = batch.ManagedEc2EcsComputeEnvironment(
    self, "GpuCompute",
    spot=True,
    spot_bid_percentage=70,  # Max 70% of on-demand price
    instance_types=[
        ec2.InstanceType("g6e.xlarge"),   # Primary: L40S 48GB
        ec2.InstanceType("g5.xlarge"),    # Fallback: A10G 24GB
        ec2.InstanceType("g5.2xlarge"),   # Fallback: A10G + more CPU
    ],
    allocation_strategy=batch.AllocationStrategy.SPOT_PRICE_CAPACITY_OPTIMIZED,
    maxv_cpus=40,
    minv_cpus=0,
    vpc=vpc,
    vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
    security_groups=[batch_gpu_sg],
    images=[batch.EcsMachineImage(
        image=ec2.MachineImage.generic_linux({"us-east-1": "ami-xxxxx"}),
        image_type=batch.EcsMachineImageType.ECS_AL2_NVIDIA,
    )],
    use_optimal_instance_classes=False,
)
```

#### 7.4 Spot Interruption Handler (add to VLM worker code)

```python
import requests
import threading

def _spot_interruption_monitor(shutdown_event: threading.Event):
    """Poll EC2 metadata for Spot interruption notice."""
    while not shutdown_event.is_set():
        try:
            r = requests.get(
                "http://169.254.169.254/latest/meta-data/spot/instance-action",
                timeout=1
            )
            if r.status_code == 200:
                log.warning("spot_interruption_detected", action=r.json())
                shutdown_event.set()  # Signal main loop to flush and exit
                return
        except requests.exceptions.RequestException:
            pass  # No interruption notice yet
        shutdown_event.wait(5)  # Check every 5 seconds
```

#### 7.5 Complete CloudWatch Dashboard Widgets

```json
{
  "widgets": [
    {"type": "metric", "properties": {"title": "Active GPU Instances",
     "metrics": [["AWS/Batch", "ActiveInstances", "ComputeEnvironment", "GpuCompute"]]}},
    {"type": "metric", "properties": {"title": "VLM Calls Per Minute",
     "metrics": [["SoccerPipeline", "VLMCallsPerMinute"]]}},
    {"type": "metric", "properties": {"title": "VLM Latency P50/P99",
     "metrics": [["SoccerPipeline", "VLMLatency", {"stat": "p50"}],
                 ["SoccerPipeline", "VLMLatency", {"stat": "p99"}]]}},
    {"type": "metric", "properties": {"title": "Step Functions Executions",
     "metrics": [["AWS/States", "ExecutionsStarted"],
                 ["AWS/States", "ExecutionsSucceeded"],
                 ["AWS/States", "ExecutionsFailed"]]}},
    {"type": "metric", "properties": {"title": "S3 Working Bucket Size",
     "metrics": [["AWS/S3", "BucketSizeBytes", "BucketName", "pipeline-working"]]}},
    {"type": "metric", "properties": {"title": "Estimated Cost Per Job",
     "metrics": [["SoccerPipeline", "JobCostEstimate"]]}}
  ]
}
```

---

### 8. Overall Assessment

This is a well-thought-out architecture design. The author clearly understands AWS Batch, Step Functions, and the tradeoffs between GPU self-hosting and API calls. The cost analysis is unusually thorough for a design doc, and the three-tier model is a genuinely good idea.

The most impactful changes to make before implementation:

1. **Eliminate NAT Gateway** (saves $32+/month, biggest single cost item for low-volume usage)
2. **Add API authentication** (security gap that blocks production deployment)
3. **Add AWS Budgets and cost alerting** (safety net against runaway charges)
4. **Add Spot interruption handler in worker code** (improves checkpoint fidelity)
5. **Add readiness probe in VLM worker** (don't rely solely on Batch health checks)
6. **Set CloudWatch Logs retention to 30 days** (prevents unbounded log storage cost)
7. **Use CDK for IaC** (the Step Functions + Batch + IAM wiring is complex enough to justify it)

The architecture is production-ready after addressing findings #1 (NAT Gateway), #2 (WAF/auth), and #10 (cost alerting). Everything else is optimization that can be done iteratively.
