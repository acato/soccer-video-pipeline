# ADR-001: Streaming-First Video Processing

Date: 2025-02-20
Status: Accepted

## Context
Source videos are 4K H.264/H.265 files, tens of GB each, on a NAS with variable latency.
Loading full video into RAM is not feasible (would require 100+ GB RAM for a 90-min 4K match).

## Decision
All video processing uses streaming/chunked reads via FFmpeg subprocess calls.
Frames are decoded into memory only within 30-second sliding windows.
Decoded frames are written to local SSD temp storage and immediately deleted after inference.

## Consequences
- RAM usage capped at ~2GB regardless of source file size
- Processing is sequential (chunk-by-chunk) — cannot parallelize within a single file without coordination
- Clip extraction uses `-ss` seek before `-i` (fast keyframe seek) which may have ~0.5s boundary imprecision; acceptable for sports highlight reels
