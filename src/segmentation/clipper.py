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

from src.detection.models import Event, EVENT_TYPE_CONFIG


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
    # Filter to events that should be included.
    # Note: reel_targets filtering is deprecated — callers now pre-filter
    # by event type via ReelSpec. For backward compat, if any event has
    # reel_targets set, we still filter by it; otherwise include all.
    reel_events = [
        e for e in events
        if e.should_include()
    ]
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
                # Duration cap prevented merge — trim start to avoid
                # overlapping with the previous clip.
                safe_start = max(start, prev_end)
                if safe_start < end:
                    merged.append((safe_start, end, [event]))
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


def compute_clips_v2(
    events: list[Event],
    video_duration: float,
    reel_name: str = "reel",
    merge_gap_sec: float = 2.0,
) -> list[ClipBoundary]:
    """
    Compute clips using per-event-type padding from EVENT_TYPE_CONFIG.

    Unlike compute_clips, this function does NOT filter by reel_targets.
    The caller is responsible for passing only the wanted events.

    Padding is looked up per event from EVENT_TYPE_CONFIG. Events whose type
    is not in the config fall back to 3.0/5.0 pre/post and 90s max.

    Args:
        events: Pre-filtered events (only the ones wanted for this reel)
        video_duration: Total length of source video in seconds
        reel_name: Name for ClipBoundary.reel_type field
        merge_gap_sec: Clips with gap < this are merged into one

    Returns:
        Sorted list of ClipBoundary objects, no overlaps.
    """
    includable = [e for e in events if e.should_include()]
    if not includable:
        return []

    includable = sorted(includable, key=lambda e: e.timestamp_start)

    # Apply per-event padding and clamp
    raw_clips: list[tuple[float, float, float, Event]] = []
    for event in includable:
        cfg = EVENT_TYPE_CONFIG.get(event.event_type)
        pre = cfg.pre_pad_sec if cfg else 3.0
        post = cfg.post_pad_sec if cfg else 5.0
        max_dur = cfg.max_clip_sec if cfg else 90.0

        start = max(0.0, event.timestamp_start - pre)
        end = min(video_duration, event.timestamp_end + post)
        raw_clips.append((start, end, max_dur, event))

    # Merge overlapping or close clips (cap merged clip by smallest max_dur)
    merged: list[tuple[float, float, float, list[Event]]] = []
    for start, end, max_dur, event in raw_clips:
        if merged and start - merged[-1][1] <= merge_gap_sec:
            prev_start, prev_end, prev_max, prev_events = merged[-1]
            effective_max = min(prev_max, max_dur)
            new_end = max(prev_end, end)
            if new_end - prev_start <= effective_max:
                merged[-1] = (prev_start, new_end, effective_max, prev_events + [event])
            else:
                safe_start = max(start, prev_end)
                if safe_start < end:
                    merged.append((safe_start, end, max_dur, [event]))
        else:
            merged.append((start, end, max_dur, [event]))

    # Build ClipBoundary objects
    boundaries = []
    for start, end, _max_dur, clip_events in merged:
        primary = max(clip_events, key=lambda e: e.confidence)
        boundaries.append(ClipBoundary(
            source_file=clip_events[0].source_file,
            start_sec=start,
            end_sec=end,
            events=[e.event_id for e in clip_events],
            reel_type=reel_name,
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
