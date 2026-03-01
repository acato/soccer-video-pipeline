"""
Post-processing for clip lists:
  - Temporal IoU deduplication (chunk-boundary near-duplicates)
  - Minimum clip duration enforcement
  - Maximum reel duration cap (with confidence-based pruning)

These run after clipper.py and before assembly.
"""
from __future__ import annotations

import structlog

from src.segmentation.clipper import ClipBoundary

log = structlog.get_logger(__name__)

# Default max reel durations (configurable)
MAX_REEL_DURATIONS: dict[str, float] = {
    "goalkeeper": 20 * 60,   # 20 min max for legacy GK reel
    "keeper": 20 * 60,       # 20 min max for keeper reel (matches keeper_a/b sub-roles)
    "keeper_a": 20 * 60,     # 20 min max for keeper A (left half)
    "keeper_b": 20 * 60,     # 20 min max for keeper B (right half)
    "highlights": 15 * 60,   # 15 min max for highlights reel
    "player": 15 * 60,
}


def deduplicate_clips(
    clips: list[ClipBoundary],
    overlap_threshold: float = 0.3,
) -> list[ClipBoundary]:
    """
    Remove near-duplicate clips based on temporal overlap (IoU).

    When two clips overlap above the threshold, the one covering more events
    is kept (tie-broken by longer duration). Assumes clips may come from
    overlapping detection chunks or multiple plugins with different padding.

    Returns deduplicated list sorted by start_sec.
    """
    if len(clips) <= 1:
        return clips

    # Sort by number of events desc so we prefer richer clips when deduplicating
    sorted_clips = sorted(clips, key=lambda c: (-len(c.events), c.start_sec))
    kept: list[ClipBoundary] = []

    for clip in sorted_clips:
        is_dup = any(
            _temporal_iou(clip, k) >= overlap_threshold
            for k in kept
        )
        if not is_dup:
            kept.append(clip)

    result = sorted(kept, key=lambda c: c.start_sec)
    removed = len(clips) - len(result)
    if removed:
        log.info("deduplicator.removed_duplicates", removed=removed, kept=len(result))
    return result


def enforce_min_duration(
    clips: list[ClipBoundary],
    min_duration: float = 2.0,
) -> list[ClipBoundary]:
    """
    Remove clips shorter than min_duration seconds.

    Very short clips (< 2s) are usually detection artifacts and look bad in reels.
    """
    result = [c for c in clips if (c.end_sec - c.start_sec) >= min_duration]
    removed = len(clips) - len(result)
    if removed:
        log.info("deduplicator.removed_short_clips", removed=removed, min_duration=min_duration)
    return result


def enforce_max_reel_duration(
    clips: list[ClipBoundary],
    reel_type: str,
    max_duration_sec: float = None,
    event_confidence_map: dict[str, float] = None,
) -> list[ClipBoundary]:
    """
    Prune clips to fit within a maximum total reel duration.

    When the total duration exceeds the cap, clips are pruned in ascending
    confidence order (lowest confidence dropped first) until under the cap.

    Args:
        clips: Input clip list (sorted by time)
        reel_type: Used to look up default max duration if max_duration_sec not given
        max_duration_sec: Override max duration in seconds
        event_confidence_map: {event_id: confidence} for pruning priority.
            If None, all clips treated as equal priority.

    Returns:
        Time-sorted clip list within the duration cap.
    """
    if not clips:
        return []

    cap = max_duration_sec if max_duration_sec is not None else MAX_REEL_DURATIONS.get(reel_type, 3600)
    total = sum(c.end_sec - c.start_sec for c in clips)

    if total <= cap:
        return sorted(clips, key=lambda c: c.start_sec)

    log.info(
        "deduplicator.pruning_for_cap",
        reel_type=reel_type,
        total_sec=round(total),
        cap_sec=round(cap),
    )

    # Assign a score to each clip: max confidence of its events
    def clip_score(clip: ClipBoundary) -> float:
        if not event_confidence_map or not clip.events:
            return 0.5
        confidences = [event_confidence_map.get(eid, 0.5) for eid in clip.events]
        return max(confidences)

    # Sort by score asc (lowest confidence first for dropping)
    scored = sorted(clips, key=clip_score)
    kept = list(clips)  # Start with all, remove from lowest first

    for clip in scored:
        if sum(c.end_sec - c.start_sec for c in kept) <= cap:
            break
        kept.remove(clip)
        log.debug(
            "deduplicator.dropped_clip",
            start=clip.start_sec,
            end=clip.end_sec,
            score=round(clip_score(clip), 3),
        )

    return sorted(kept, key=lambda c: c.start_sec)


def postprocess_clips(
    clips: list[ClipBoundary],
    reel_type: str,
    min_duration: float = 2.0,
    max_reel_duration_sec: float = None,
    event_confidence_map: dict[str, float] = None,
    dedup_threshold: float = 0.3,
) -> list[ClipBoundary]:
    """
    Full post-processing pipeline: dedup → min duration → max reel cap.

    This is the single function to call after compute_clips().
    Returns the final clip list ready for assembly.
    """
    clips = deduplicate_clips(clips, overlap_threshold=dedup_threshold)
    clips = enforce_min_duration(clips, min_duration=min_duration)
    clips = enforce_max_reel_duration(
        clips,
        reel_type=reel_type,
        max_duration_sec=max_reel_duration_sec,
        event_confidence_map=event_confidence_map,
    )
    log.info(
        "deduplicator.postprocess_done",
        reel_type=reel_type,
        final_clips=len(clips),
        total_duration_sec=round(sum(c.end_sec - c.start_sec for c in clips), 1),
    )
    return clips


def _temporal_iou(a: ClipBoundary, b: ClipBoundary) -> float:
    """Compute temporal Intersection over Union of two clips."""
    inter_start = max(a.start_sec, b.start_sec)
    inter_end = min(a.end_sec, b.end_sec)
    intersection = max(0.0, inter_end - inter_start)
    if intersection == 0.0:
        return 0.0
    union = (a.end_sec - a.start_sec) + (b.end_sec - b.start_sec) - intersection
    return intersection / union if union > 0 else 0.0
