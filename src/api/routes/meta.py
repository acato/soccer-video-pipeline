"""
Metadata endpoints for the UI — event types and jersey colors.

GET /event-types   — all event types with label, category, clip params
GET /jersey-colors — all palette entries
"""
from __future__ import annotations

from fastapi import APIRouter

from src.detection.jersey_classifier import JERSEY_COLOR_PALETTE
from src.detection.models import EVENT_TYPE_CONFIG, EventType

router = APIRouter()


@router.get("/event-types")
def list_event_types():
    """Return all event types with their config (label, category, padding, etc.)."""
    result = []
    for et in EventType:
        cfg = EVENT_TYPE_CONFIG.get(et)
        if cfg is None:
            continue
        result.append({
            "value": et.value,
            "label": cfg.label,
            "category": cfg.category,
            "pre_pad_sec": cfg.pre_pad_sec,
            "post_pad_sec": cfg.post_pad_sec,
            "max_clip_sec": cfg.max_clip_sec,
            "min_confidence": cfg.min_confidence,
            "is_gk_event": cfg.is_gk_event,
        })
    return result


@router.get("/jersey-colors")
def list_jersey_colors():
    """Return all available jersey color names."""
    return sorted(JERSEY_COLOR_PALETTE.keys())
