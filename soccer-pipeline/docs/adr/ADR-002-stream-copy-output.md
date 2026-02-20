# ADR-002: Stream Copy as Default Output Codec

Date: 2025-02-20
Status: Accepted

## Context
Re-encoding 4K H.264 to produce the output reel takes significant time (potentially hours for a 90-min match) and introduces quality loss even at high bitrate settings.

## Decision
Output reels use `-c copy` (stream copy) by default. This copies compressed video packets directly without decoding/re-encoding. Falls back to `libx264` only if stream copy fails.

## Consequences
- Output quality is identical to source (no generation loss)
- Assembly is very fast (10-100x faster than re-encode)
- Clip boundaries may land on non-keyframes, causing slight imprecision (max ~0.5s)
- All clips in a concat operation must have identical codec/resolution/fps (guaranteed when all from same source)
