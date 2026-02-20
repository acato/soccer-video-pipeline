# Soccer Video Processing Pipeline — Agent Workspace

## Project Overview

Agent-based video processing pipeline that ingests large 4K soccer match recordings from a NAS and produces:
1. **Goalkeeper Reel** — Every GK action: saves, catches, punches, distribution (short/long pass, goal kicks)
2. **Highlights Reel** — Core match events: shots, goals, near-misses, key defensive actions, great individual plays
3. **Player Reel** (on-demand) — Personalized reel for any outfield player (Phase 2)

## Input Constraints
- Format: MP4, H.264 or H.265
- Resolution/Frame rate: 4K (3840×2160) at 30 or 60 fps
- Size: Tens of GB per file
- Storage: Network-attached storage (NAS), mounted or accessible via SMB/NFS
- Multiple camera angles possible per game

## Agent Roster

| Agent | Directory | Role |
|-------|-----------|------|
| Architect | `agents/architect/` | System design, ADRs, interface contracts |
| Developer | `agents/developer/` | Implementation across all modules |
| SDET | `agents/sdet/` | Test strategy, test suites, CI integration |
| Video Analyst | `agents/video-analyst/` | CV model selection, tuning, event detection logic |
| Pipeline Operator | `agents/pipeline-operator/` | Deployment, monitoring, NAS integration, job management |

## Source Layout
```
src/
  ingestion/      # NAS watcher, video intake, metadata extraction
  detection/      # CV-based event detection (shots, saves, distribution)
  tracking/       # Player/ball tracking across frames
  segmentation/   # Clip boundary determination & trimming
  assembly/       # Reel composition, encoding, output
  api/            # REST API for job submission and status
tests/
infra/            # Docker, compose, deployment configs
docs/             # ADRs, runbooks
```

## Key Design Principles
- **Streaming-first**: Never load full video into RAM. Use FFmpeg segment processing.
- **Async job queue**: Each match is a job; processing is async with status tracking.
- **Modular detection**: Each event type (save, shot, goal, distribution) is an independent detector.
- **NAS-aware**: Read directly from NAS mount; write intermediate artifacts to local fast storage; final output back to NAS.
- **GPU-optional**: Detection pipeline degrades gracefully to CPU (slower) when no GPU present.

## Technology Stack (baseline — Architect may revise)
- **Runtime**: Python 3.11+
- **Video I/O**: FFmpeg (via ffmpeg-python), OpenCV
- **ML/CV**: YOLOv8 (ultralytics) for player/ball detection, ByteTrack for tracking
- **Event Classification**: fine-tuned action recognition (SlowFast or VideoMAE)
- **Orchestration**: Celery + Redis for job queue
- **API**: FastAPI
- **Storage**: Local SSD for working files, NAS for source + output
- **Infra**: Docker Compose (single node), extensible to K8s

## Agent Invocation Guide

The root context acts as **orchestrator**.
Sub-agents are invoked by navigating to their directory:
```
cd agents/architect       && claude   # Architecture decisions
cd agents/developer       && claude   # Implementation work  
cd agents/sdet            && claude   # Testing
cd agents/video-analyst   && claude   # CV/ML decisions
cd agents/pipeline-operator && claude # Ops/deployment
```

## Current Phase
**Phase 0 — Scaffolding complete. Begin with Architect agent.**

Execution order:
1. Architect → `docs/architecture.md` + interface contracts
2. Video Analyst → model selection + event taxonomy
3. Developer → implement modules (ingestion → detection → segmentation → assembly → api)
4. SDET → test harnesses per module
5. Pipeline Operator → Docker Compose, NAS config, monitoring
