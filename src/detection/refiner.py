"""Pass-2 refinement: per-event-type targeted confirmation queries.

Phase A measurement (scripts/analyze_confidence_signal.py on Run 49 + 52)
showed VLM-emitted confidence is essentially noise — the model emits 0.8
for almost everything regardless of TP/FP status, and confidence-based
thresholding gives ~0.000 F1 improvement on the reel-weighted metric.

So Pass 2 cannot be a cheap conf-threshold filter. It must actually re-
examine each Pass 1 candidate with a sharper, type-specific prompt that
asks ONE question with ONE evidence ruleset, rather than Pass 1's generic
9-type decision tree.

Hypothesis: prompt fragmentation. When the VLM has to evaluate every type
in one call, it carries cognitive load that produces inconsistent type
discrimination (e.g. shot-with-restart-following gets tagged as goal_kick
because the prompt's "STEP 1 — restart" check fires first). A targeted
"is this specifically a {type}?" call removes that ambiguity.

Approach:
  - For each Pass 1 candidate, sample denser frames (8 @ 1.5s spacing
    centered on event) and apply the same field_crop the parent used.
  - Build a confirmation prompt with the evidence clause for that type.
  - VLM returns {confirmed, confidence, reasoning}.
  - Only confirmed events pass through.

Token budget per call: ~5–6k (8 frames × ~600–900 visual tokens after
field_crop + ~400 prompt + ~150 output). Fits 8192 max_model_len easily.
"""
from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import structlog
from PIL import Image

from src.detection.models import Event
from src.detection.frame_sampler import FrameSampler

log = structlog.get_logger(__name__)


# ── Per-type evidence clauses ────────────────────────────────────────────
# Each clause is a sharp, ONE-event-type question. Lifted from the existing
# _DIRECT_CLASSIFY_PROMPT decision tree but stripped of cross-type noise.
_PER_TYPE_EVIDENCE: dict[str, str] = {
    "throw_in": (
        "Is there a player near the SIDELINE/TOUCHLINE (top or bottom edge of "
        "the pitch view) holding a ball with BOTH HANDS, picking up a ball, "
        "walking toward the touchline with a ball, or mid-throw motion? Any "
        "of these poses across the frames qualifies. Reject if the apparent "
        "ball-handler is mid-pitch (not at the touchline) or if no ball is "
        "visible in any hand."
    ),
    "catch": (
        "Is the GOALKEEPER (typically in a distinct kit, often near the goal "
        "or 6-yard box) visible HOLDING a ball in their hands or arms — even "
        "briefly, including walking with it, bouncing it, or preparing to "
        "throw/punt it? The ball must be IN the GK's hands, not bouncing "
        "free. Reject if the player holding the ball is at midfield (a "
        "throw-in) or off-pitch."
    ),
    "corner_kick": (
        "Is the ball STATIONARY at or very near a CORNER FLAG / corner arc "
        "(one of the four corners of the pitch), with a player standing over "
        "it ready to kick? Reject if the ball is at any other location, "
        "including along the touchline (that would be a throw-in) or at the "
        "6-yard box (that would be a goal kick)."
    ),
    "goal_kick": (
        "Is the ball STATIONARY in the 6-YARD BOX / GOAL AREA (the small "
        "rectangle directly in front of the goal), with the GOALKEEPER or a "
        "defender standing over it about to kick upfield, and outfield "
        "opponents RETREATED away from the box? Reject if the ball is at "
        "midfield, at a corner, or near a touchline."
    ),
    "free_kick_shot": (
        "Both of the following must be visible: (a) the ball is STATIONARY "
        "on the pitch outside the goal area and away from the corners, with "
        "a kicker standing over it, AND (b) direct free-kick context — "
        "either a visible DEFENSIVE WALL of 3+ opponents lined up within "
        "~10 m of the ball, OR an obvious pre-event foul (player on the "
        "ground, referee gesturing, cluster of protesting players). Reject "
        "if (b) is missing — a stationary ball without a wall or foul is "
        "more likely a paused play, throw-in setup, or goal kick."
    ),
    "shot_on_target": (
        "Did a player STRIKE the ball toward the goal? Look for: a player's "
        "leg/foot in contact with the ball, the ball moving toward or "
        "arriving at the goal mouth (ball in flight near goal posts/net), "
        "or an immediate post-strike scene (GK reacting, defenders chasing "
        "back). The shot counts even if saved by the GK, blocked, or going "
        "wide of goal. Reject if no kick action is visible and the ball is "
        "in normal mid-pitch passing motion."
    ),
    "shot_stop_diving": (
        "Did the GOALKEEPER DIVE or LUNGE laterally/forward, with their body "
        "going to the ground, AND did the ball deflect or rebound away from "
        "the GK (not held)? Look for the GK's body horizontal or low-angled, "
        "ball traveling away from their hands. Reject if the GK is standing "
        "and catching the ball (that's a catch), or if there is no visible "
        "diving motion."
    ),
    "goal": (
        "Did a shot toward goal lead to UNAMBIGUOUS CELEBRATION — players "
        "with arms raised in joy, sprinting to teammates, group hugs/piles "
        "— OR teams walking back toward the center circle to set up a "
        "kickoff? The negation 'no celebration' or 'players continue play' "
        "rules out goal. Be strict: ball-near-net alone is NOT a goal."
    ),
    "penalty": (
        "Is there a player standing over the ball at the PENALTY SPOT inside "
        "the 18-yard box, with the goalkeeper alone on the goal line and "
        "all other outfield players outside the penalty area? Reject if the "
        "ball is anywhere else or the player setup doesn't match a penalty "
        "ceremony."
    ),
}


_REFINEMENT_PROMPT = """\
You are reviewing a candidate {event_type} event detected in a soccer match \
at {start:.0f}s – {end:.0f}s. Examine these {n_frames} frames closely.

CONFIRMATION CRITERION for {event_type}:
{evidence_clause}

Reply as a single JSON object on one line:
  {{"confirmed": true|false, "confidence": 0.0-1.0, "reasoning": "brief"}}

Rules:
- "confirmed": true ONLY if the criterion above is clearly visible in at \
least one frame.
- "confirmed": false if the frames show open play, a different event, or \
ambiguous/insufficient evidence.
- Default is false. Be strict — over-confirming defeats the purpose of \
this second-pass review.
- Do not output anything other than the JSON object.
"""


@dataclass
class RefinementResult:
    event: Event
    confirmed: bool
    new_confidence: float
    reasoning: str
    latency_sec: float


@dataclass
class RefinerStats:
    total: int = 0
    confirmed: int = 0
    rejected: int = 0
    api_errors: int = 0
    no_evidence_clause: int = 0  # event_type missing from _PER_TYPE_EVIDENCE
    by_type_in: dict[str, int] = None
    by_type_confirmed: dict[str, int] = None

    def __post_init__(self):
        if self.by_type_in is None:
            self.by_type_in = {}
        if self.by_type_confirmed is None:
            self.by_type_confirmed = {}


def _crop_to_field(jpeg_bytes: bytes, field_bbox: Optional[tuple]) -> bytes:
    """Apply the cached field bbox to a frame, matching parent detector behavior."""
    if field_bbox is None:
        return jpeg_bytes
    x1, y1, x2, y2 = field_bbox
    if (x2 - x1) * (y2 - y1) >= 0.98:
        return jpeg_bytes
    img = Image.open(io.BytesIO(jpeg_bytes))
    W, H = img.size
    cropped = img.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def refine_events(
    events: list[Event],
    *,
    sampler: FrameSampler,
    video_duration: float,
    vllm_url: str,
    model: str,
    field_bbox: Optional[tuple] = None,
    n_frames: int = 8,
    span_sec: float = 12.0,
    timeout_sec: int = 60,
    job_id: Optional[str] = None,
) -> tuple[list[Event], RefinerStats, list[RefinementResult]]:
    """Re-examine each candidate with a type-specific confirmation prompt.

    Sequential per the QL1 spec — vLLM is a single stream, so fan-out
    doesn't actually parallelize. Returns the confirmed-only events plus
    aggregate stats and per-event verdicts (for diagnostics).
    """
    stats = RefinerStats()
    results: list[RefinementResult] = []
    confirmed: list[Event] = []

    for event in events:
        stats.total += 1
        etype = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
        stats.by_type_in[etype] = stats.by_type_in.get(etype, 0) + 1

        # Sample denser frames around the candidate
        center = (event.timestamp_start + event.timestamp_end) / 2
        half_span = span_sec / 2
        interval = max(1.0, span_sec / max(1, n_frames))
        frames = sampler.sample_range(
            center_sec=center,
            window_sec=half_span,
            interval_sec=interval,
            duration_sec=video_duration,
        )
        if not frames:
            stats.api_errors += 1
            continue
        if len(frames) > n_frames:
            s = len(frames) / n_frames
            indices = [int(i * s) for i in range(n_frames)]
            frames = [frames[i] for i in indices]

        # Apply field_crop to match parent's wide-view normalization
        if field_bbox is not None:
            frames = [
                type(f)(timestamp_sec=f.timestamp_sec,
                        jpeg_bytes=_crop_to_field(f.jpeg_bytes, field_bbox))
                for f in frames
            ]

        evidence = _PER_TYPE_EVIDENCE.get(etype)
        if evidence is None:
            # Unknown type — confirm by default to avoid silent drops
            stats.no_evidence_clause += 1
            stats.confirmed += 1
            stats.by_type_confirmed[etype] = stats.by_type_confirmed.get(etype, 0) + 1
            confirmed.append(event)
            results.append(RefinementResult(
                event=event, confirmed=True, new_confidence=event.confidence,
                reasoning="no_evidence_clause_for_type", latency_sec=0.0,
            ))
            continue

        prompt = _REFINEMENT_PROMPT.format(
            event_type=etype,
            start=event.timestamp_start,
            end=event.timestamp_end,
            n_frames=len(frames),
            evidence_clause=evidence,
        )

        content: list[dict] = []
        for frame in frames:
            b64 = base64.b64encode(frame.jpeg_bytes).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
            content.append({"type": "text", "text": f"t={frame.timestamp_sec:.1f}s"})
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 200,
            "temperature": 0,
        }

        t0 = time.monotonic()
        try:
            r = httpx.post(
                f"{vllm_url}/v1/chat/completions",
                json=payload,
                timeout=timeout_sec,
            )
            elapsed = time.monotonic() - t0

            if r.status_code != 200:
                log.warning("refiner.api_error",
                            status=r.status_code, body=r.text[:200],
                            event_type=etype, job_id=job_id)
                stats.api_errors += 1
                # Fail-open: accept the event if we couldn't get a verdict
                confirmed.append(event)
                stats.confirmed += 1
                stats.by_type_confirmed[etype] = stats.by_type_confirmed.get(etype, 0) + 1
                results.append(RefinementResult(
                    event=event, confirmed=True, new_confidence=event.confidence,
                    reasoning="api_error_fail_open", latency_sec=elapsed,
                ))
                continue

            text = r.json()["choices"][0]["message"]["content"].strip()
            verdict = _parse_verdict(text)
        except Exception as exc:
            log.warning("refiner.exception", error=str(exc),
                        event_type=etype, job_id=job_id)
            stats.api_errors += 1
            elapsed = time.monotonic() - t0
            # Fail-open
            confirmed.append(event)
            stats.confirmed += 1
            stats.by_type_confirmed[etype] = stats.by_type_confirmed.get(etype, 0) + 1
            results.append(RefinementResult(
                event=event, confirmed=True, new_confidence=event.confidence,
                reasoning=f"exception_fail_open: {exc}", latency_sec=elapsed,
            ))
            continue

        is_confirmed = bool(verdict.get("confirmed", False))
        new_conf = float(verdict.get("confidence", event.confidence))
        reasoning = str(verdict.get("reasoning", ""))[:300]

        results.append(RefinementResult(
            event=event,
            confirmed=is_confirmed,
            new_confidence=new_conf,
            reasoning=reasoning,
            latency_sec=elapsed,
        ))

        if is_confirmed:
            stats.confirmed += 1
            stats.by_type_confirmed[etype] = stats.by_type_confirmed.get(etype, 0) + 1
            # Patch the event's confidence and metadata with the new info
            try:
                event.confidence = new_conf
                if event.metadata is None:
                    event.metadata = {}
                event.metadata["pass2_reasoning"] = reasoning
                event.metadata["pass2_confidence"] = new_conf
            except Exception:
                pass
            confirmed.append(event)
        else:
            stats.rejected += 1

        log.debug("refiner.verdict",
                  event_type=etype,
                  start=event.timestamp_start,
                  confirmed=is_confirmed,
                  conf=new_conf,
                  elapsed=f"{elapsed:.1f}s",
                  job_id=job_id)

    return confirmed, stats, results


def _parse_verdict(text: str) -> dict:
    """Parse the VLM's confirmation JSON. Falls back to a permissive parse
    if the model returns extra prose around the JSON."""
    text = text.strip()
    # Direct JSON?
    try:
        return json.loads(text)
    except Exception:
        pass
    # Look for a JSON object substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    # As a last resort, look for confirmed:true/false in the text
    lower = text.lower()
    if "confirmed" in lower:
        if '"confirmed": true' in lower or '"confirmed":true' in lower:
            return {"confirmed": True, "confidence": 0.7, "reasoning": text[:200]}
        if '"confirmed": false' in lower or '"confirmed":false' in lower:
            return {"confirmed": False, "confidence": 0.7, "reasoning": text[:200]}
    # Fail-open
    return {"confirmed": True, "confidence": 0.5, "reasoning": "parse_failed: " + text[:100]}
