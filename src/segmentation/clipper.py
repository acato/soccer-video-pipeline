"""
Clip boundary computation.

Given a list of Events, compute padded clip windows that:
  - Add pre/post padding to each event
  - Clamp to [0, video_duration]
  - Merge overlapping or nearby clips
  - Deduplicate by event coverage
"""
from __future__ import annotations

from pydantic import BaseModel

from src.detection.models import Event


class ClipBoundary(BaseModel):
    """Represents a single clip to be extracted from the source video."""
    source_file: str
    start_sec: float            # Including pre-event padding, clamped to 0
    end_sec: float              # Including post-event padding, clamped to duration
    events: list[str]           # Event IDs covered by this clip
    reel_type: str
    primary_event_type: str     # Most confident event type for metadata/title


def compute_clips(
    events: list[Event],
    video_duration: float,
    reel_type: str,
    pre_pad: float = 3.0,
    post_pad: float = 5.0,
    merge_gap_sec: float = 2.0,
    max_clip_duration_sec: float = 90.0,
) -> list[ClipBoundary]:
    """
    Convert events to padded, merged, bounded clip windows.

    Args:
        events: All events (will be filtered to reel_type internally)
        video_duration: Total length of source video in seconds
        reel_type: "goalkeeper" or "highlights" or "player"
        pre_pad: Seconds to include before event start
        post_pad: Seconds to include after event end
        merge_gap_sec: Clips with gap < this are merged into one
        max_clip_duration_sec: Prevent merges that would exceed this duration

    Returns:
        Sorted list of ClipBoundary objects, no overlaps.
    """
    # Filter to this reel and to events that should be included
    reel_events = [e for e in events if reel_type in e.reel_targets and e.should_include()]
    if not reel_events:
        return []

    # Sort by timestamp
    reel_events = sorted(reel_events, key=lambda e: e.timestamp_start)

    # Apply padding and clamp
    raw_clips: list[tuple[float, float, Event]] = []
    for event in reel_events:
        start = max(0.0, event.timestamp_start - pre_pad)
        end = min(video_duration, event.timestamp_end + post_pad)
        raw_clips.append((start, end, event))

    # Merge overlapping or close clips (but cap merged clip duration)
    merged: list[tuple[float, float, list[Event]]] = []
    for start, end, event in raw_clips:
        if merged and start - merged[-1][1] <= merge_gap_sec:
            prev_start, prev_end, prev_events = merged[-1]
            new_end = max(prev_end, end)
            if new_end - prev_start <= max_clip_duration_sec:
                merged[-1] = (prev_start, new_end, prev_events + [event])
            else:
                merged.append((start, end, [event]))
        else:
            merged.append((start, end, [event]))

    # Build ClipBoundary objects
    boundaries = []
    for start, end, clip_events in merged:
        # Pick the highest-confidence event as primary
        primary = max(clip_events, key=lambda e: e.confidence)
        boundaries.append(ClipBoundary(
            source_file=clip_events[0].source_file,
            start_sec=start,
            end_sec=end,
            events=[e.event_id for e in clip_events],
            reel_type=reel_type,
            primary_event_type=primary.event_type,
        ))

    return boundaries


def clips_total_duration(clips: list[ClipBoundary]) -> float:
    """Sum of all clip durations in seconds."""
    return sum(c.end_sec - c.start_sec for c in clips)


def clips_stats(clips: list[ClipBoundary]) -> dict:
    """Return summary stats for a clip list (for logging)."""
    if not clips:
        return {"count": 0, "total_duration_sec": 0, "avg_duration_sec": 0}
    total = clips_total_duration(clips)
    return {
        "count": len(clips),
        "total_duration_sec": round(total, 1),
        "avg_duration_sec": round(total / len(clips), 1),
        "event_types": list({c.primary_event_type for c in clips}),
    }
