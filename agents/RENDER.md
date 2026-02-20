# Agent: RENDER

## Role
You own all FFmpeg operations: clip extraction, concatenation, title card generation,
burn-in overlays, and final encoding. You are the only agent that writes to the NAS
output directory.

## Core Operations

### 1. Clip Extraction
```python
def extract_clip(
    source: Path,
    start_ts: float,
    end_ts: float,
    output: Path,
    *,
    codec: Literal["copy", "h264", "hevc"] = "copy",
) -> None:
    """
    Use stream copy for speed where possible.
    Seek with -ss BEFORE -i for fast H.265 seek.
    Add -avoid_negative_ts make_zero for concat compatibility.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_ts),
        "-to", str(end_ts),
        "-i", str(source),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output)
    ]
```

**Critical H.265 seeking note**: Use `-ss` before `-i` (input seeking) for speed,
but this may land on a keyframe before the target. Add `-vf trim=start=0` if
frame-accurate cutting is required (re-encode only affected clips).

### 2. Concatenation
```python
def concat_clips(clip_paths: list[Path], output: Path) -> None:
    """
    Use FFmpeg concat demuxer (not filter) for stream-copy speed.
    Generates temporary concat.txt file in scratch dir.
    All input clips must have identical codec/resolution — validate first.
    """
```

### 3. Title Card Generation
```python
def generate_title_card(
    text: str,
    duration_s: float,
    resolution: tuple[int, int],  # e.g. (3840, 2160)
    output: Path,
) -> None:
    """
    Use FFmpeg lavfi + drawtext.
    Background: black. Font: Liberation Sans Bold, white, centered.
    Fade in 0.3s, hold, fade out 0.3s.
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=black:size={w}x{h}:rate=30:duration={duration_s}",
        "-vf", f"drawtext=text='{text}':fontsize=96:fontcolor=white:x=(w-tw)/2:y=(h-th)/2,fade=in:0:9,fade=out:{frames-9}:9",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        str(output)
    ]
```

### 4. Burn-In Overlays
```python
def add_overlay(
    clip: Path,
    text: str,
    position: Literal["top-left", "top-right", "bottom-left", "bottom-right"],
    output: Path,
) -> None:
    """Optional: burn event label (e.g. 'Save #3') into clip."""
```

### 5. Final Encode
```python
ENCODE_PROFILES = {
    "delivery_h264": {
        # Good quality, universal compatibility
        "vcodec": "libx264", "crf": 18, "preset": "slow",
        "acodec": "aac", "ab": "192k",
    },
    "delivery_hevc": {
        # Smaller file, same quality — for NAS archival
        "vcodec": "libx265", "crf": 22, "preset": "medium",
        "acodec": "aac", "ab": "192k",
    },
    "preview_h264": {
        # Fast, lower quality — for review before final encode
        "vcodec": "libx264", "crf": 28, "preset": "veryfast",
        "acodec": "aac", "ab": "128k",
    },
}
```

## Reel Assembly Pipeline

```
For each ClipSpec in manifest.clips:
  1. extract_clip(source, start_ts, end_ts, scratch/clip_NNN.mp4, codec="copy")
  2. validate_clip(path) → check duration, resolution match
  3. If resolution mismatch: re-encode to canonical resolution

Optionally prepend title card.
concat_clips([title_card, *clip_paths], scratch/reel_raw.mp4)
final_encode(scratch/reel_raw.mp4, output, profile=delivery_h264)
nas_client.write_output(job_id, output, manifest.output_filename)
nas_client.purge_scratch(job_id)
```

## Failure Handling

| Failure | Action |
|---|---|
| FFmpeg non-zero exit | Log stderr, raise `RenderError`, mark job FAILED |
| Clip duration mismatch >0.5s | Re-extract with re-encode |
| Scratch full mid-render | Abort, purge scratch, raise `ScratchBudgetError` |
| Output write fails | Retry 3x, then alert operator |

## Output
- Final MP4 written to `nas.output_dir/{job_id}/{manifest.output_filename}`
- Job status updated to DONE
- Emit `qa.review_reel` Celery task
