"""
VLM (Vision Language Model) classifier for goalkeeper save verification.

Uses Claude Sonnet to analyze extracted video frames and determine whether
a candidate save event is genuine.  Runs post-detection, pre-segmentation
in the worker pipeline — keeps detection fast (no API calls during chunked
processing) and lets the VLM operate on the full event list at once.

Architecture:
    BallTouchDetector → candidate events (relaxed thresholds)
         ↓
    VLMClassifier.filter_events(candidates)
         ↓  FFmpeg extracts 3 frames per event
         ↓  Claude Sonnet analyzes each frame set
         ↓  Returns keep/reject + VLM confidence
         ↓
    Spatial filter → Assembly
"""
from __future__ import annotations

import base64
import json
import subprocess
from typing import Optional

import structlog

from src.detection.models import Event

log = structlog.get_logger(__name__)


class VLMClassifier:
    """Verify goalkeeper save events using a Vision Language Model (Claude).

    For each candidate event, extracts 3 frames from the source video
    (at touch_time - 0.5s, touch_time, touch_time + 0.5s), sends them
    to Claude with match context, and parses a structured keep/reject
    decision.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        source_file: str,
        match_config,
        frame_width: int = 640,
        min_confidence: float = 0.6,
        max_workers: int = 5,
    ):
        self._api_key = api_key
        self._model = model
        self._source_file = source_file
        self._match_config = match_config
        self._frame_width = frame_width
        self._min_confidence = min_confidence
        self._max_workers = max_workers
        self._client = None  # lazy init

    def _get_client(self):
        """Lazy-init the Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def filter_events(self, events: list[Event]) -> list[Event]:
        """Filter events through VLM classification.

        Returns only events that Claude confirms as genuine GK actions.
        Events that fail frame extraction or API calls are kept (fail-open)
        to avoid dropping real saves due to transient errors.
        """
        if not events:
            return []

        kept: list[Event] = []
        for event in events:
            try:
                frames = self._extract_frames(event)
                if not frames:
                    log.warning("vlm.frame_extraction_failed",
                                event_id=event.event_id)
                    kept.append(event)  # fail-open
                    continue

                is_save, confidence, reasoning = self._classify_event(
                    event, frames
                )

                # Store VLM verdict in event metadata
                event.metadata["vlm_is_save"] = is_save
                event.metadata["vlm_confidence"] = confidence
                event.metadata["vlm_reasoning"] = reasoning
                event.metadata["vlm_model"] = self._model

                if is_save and confidence >= self._min_confidence:
                    kept.append(event)
                    log.info("vlm.kept", event_id=event.event_id,
                             event_type=event.event_type.value,
                             confidence=confidence, reasoning=reasoning)
                else:
                    log.info("vlm.rejected", event_id=event.event_id,
                             event_type=event.event_type.value,
                             is_save=is_save, confidence=confidence,
                             reasoning=reasoning)

            except Exception as exc:
                log.error("vlm.classify_error", event_id=event.event_id,
                          error=str(exc))
                kept.append(event)  # fail-open

        log.info("vlm.filter_complete", input=len(events), output=len(kept))
        return kept

    def _extract_frames(
        self,
        event: Event,
        count: int = 3,
        offsets: tuple[float, ...] = (-0.5, 0.0, 0.5),
    ) -> list[bytes]:
        """Extract frames around the event timestamp via FFmpeg.

        Returns a list of JPEG bytes, one per offset.  Uses input-side
        seek (-ss before -i) for speed, and scales to ``frame_width``
        to keep API payload small (~50-100KB per frame).
        """
        touch_time = (event.timestamp_start + event.timestamp_end) / 2
        frames: list[bytes] = []

        for offset in offsets[:count]:
            ts = max(0.0, touch_time + offset)
            jpeg_bytes = self._extract_single_frame(ts)
            if jpeg_bytes:
                frames.append(jpeg_bytes)

        return frames

    def _extract_single_frame(self, timestamp: float) -> Optional[bytes]:
        """Extract a single JPEG frame at the given timestamp."""
        cmd = [
            "ffmpeg",
            "-ss", f"{timestamp:.3f}",
            "-i", self._source_file,
            "-vframes", "1",
            "-vf", f"scale={self._frame_width}:-1",
            "-f", "image2",
            "-c:v", "mjpeg",
            "-q:v", "5",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
            log.warning("vlm.ffmpeg_failed", ts=timestamp,
                        returncode=result.returncode)
            return None
        except subprocess.TimeoutExpired:
            log.warning("vlm.ffmpeg_timeout", ts=timestamp)
            return None

    def _build_prompt(self, event: Event) -> str:
        """Build the classification prompt with match context."""
        mc = self._match_config

        # Ball position from event bounding box (if available)
        ball_x = ball_y = "unknown"
        if event.bounding_box:
            ball_x = f"{event.bounding_box.center_x:.2f}"
            ball_y = f"{event.bounding_box.center_y:.2f}"

        # Ball speed metadata (if available)
        speed_ctx = ""
        pre_speed = event.metadata.get("ball_pre_speed")
        post_speed = event.metadata.get("ball_post_speed")
        if pre_speed is not None:
            speed_ctx = f"\nBall pre-touch speed: {pre_speed:.2f}, "
            if post_speed is not None:
                speed_ctx += f"post-touch speed: {post_speed:.2f}"

        return (
            f"You are analyzing frames from a soccer match between "
            f"{mc.team.team_name} and {mc.opponent.team_name}.\n"
            f"{mc.team.team_name}'s goalkeeper wears a {mc.team.gk_color} jersey. "
            f"{mc.opponent.team_name}'s goalkeeper wears a {mc.opponent.gk_color} jersey.\n"
            f"{mc.team.team_name}'s outfield players wear {mc.team.outfield_color}. "
            f"{mc.opponent.team_name}'s outfield players wear {mc.opponent.outfield_color}.\n"
            f"\nThese 3 frames show a moment where ball tracking detected "
            f"a possible goalkeeper save (event type: {event.event_type.value}).\n"
            f"The ball position is at approximately x={ball_x}, y={ball_y} "
            f"(normalized 0-1, where 0,0=top-left).{speed_ctx}\n"
            f"\nIs this a genuine save/catch/punch by {mc.team.team_name}'s "
            f"goalkeeper ({mc.team.gk_color} jersey)?\n"
            f"\nRespond ONLY with JSON:\n"
            f'{{"is_gk_save": true/false, "confidence": 0.0-1.0, '
            f'"reasoning": "brief explanation"}}\n'
            f"\nKey criteria:\n"
            f"- Is a player in a {mc.team.gk_color} jersey visible and "
            f"actively engaged with the ball?\n"
            f"- Is this player positioned near a goal (not in midfield)?\n"
            f"- Does the action look like a save (diving, catching, punching) "
            f"vs routine play?\n"
            f"- Could this be an outfield player ({mc.opponent.outfield_color} "
            f"jersey) being misidentified?\n"
        )

    def _classify_event(
        self,
        event: Event,
        frames: list[bytes],
    ) -> tuple[bool, float, str]:
        """Send frames to Claude and parse the structured response.

        Returns (is_save, confidence, reasoning).
        """
        client = self._get_client()
        prompt = self._build_prompt(event)

        # Build message content: frames as base64 images + text prompt
        content: list[dict] = []
        for i, frame_bytes in enumerate(frames):
            b64 = base64.b64encode(frame_bytes).decode("ascii")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": content}],
        )

        return self._parse_response(response)

    def _parse_response(self, response) -> tuple[bool, float, str]:
        """Parse Claude's response into (is_save, confidence, reasoning).

        Handles both clean JSON and JSON embedded in markdown code blocks.
        Falls back to conservative defaults on parse failure.
        """
        text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            is_save = bool(data.get("is_gk_save", False))
            confidence = float(data.get("confidence", 0.0))
            reasoning = str(data.get("reasoning", ""))
            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))
            return (is_save, confidence, reasoning)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.warning("vlm.parse_error", text=text[:200], error=str(exc))
            # Fail-open: if we can't parse, assume it might be a save
            return (True, 0.0, f"parse_error: {exc}")
