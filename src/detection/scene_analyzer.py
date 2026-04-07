"""
Scene analyzer: VLM-based scene classification for the two-pass detection pipeline.

Pass 1 (scan)   — coarse scene classification at ~3-second intervals.
Pass 2 (refine) — boundary refinement at ~1-second intervals around detected events.
"""
from __future__ import annotations

import base64
import json
from typing import Optional

import structlog

from src.detection.frame_sampler import SampledFrame
from src.detection.models import EventBoundary, GameState, SceneLabel

log = structlog.get_logger(__name__)

# Game states that map to detectable event types
_EVENT_GAME_STATES: dict[str, GameState] = {
    "corner_kick": GameState.CORNER_KICK,
    "goal_kick": GameState.GOAL_KICK,
}

# Maximum frames per VLM call in the coarse scan
COARSE_BATCH_SIZE = 15


class SceneAnalyzer:
    """VLM-based scene classification -- the core detection engine."""

    def __init__(
        self,
        api_key: str,
        model: str,
        source_file: str,
        event_types: list[str],
    ):
        self._api_key = api_key
        self._model = model
        self._source_file = source_file
        self._event_types = event_types
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # Pass 1: Coarse scan
    # ------------------------------------------------------------------

    def scan(self, frames: list[SampledFrame]) -> list[SceneLabel]:
        """Classify each frame's game state in batches.

        Returns a SceneLabel per successfully classified frame.
        """
        all_labels: list[SceneLabel] = []

        for batch_start in range(0, len(frames), COARSE_BATCH_SIZE):
            batch = frames[batch_start : batch_start + COARSE_BATCH_SIZE]
            try:
                labels = self._scan_batch(batch)
                all_labels.extend(labels)
            except Exception as exc:
                log.error("scene_analyzer.scan_batch_error", error=str(exc),
                          batch_start=batch_start, batch_size=len(batch))
                # Fail-open: mark batch frames as ACTIVE_PLAY (won't create events)
                for f in batch:
                    all_labels.append(SceneLabel(
                        timestamp_sec=f.timestamp_sec,
                        game_state=GameState.ACTIVE_PLAY,
                    ))

        log.info("scene_analyzer.scan_complete", total_labels=len(all_labels))
        return all_labels

    def _scan_batch(self, batch: list[SampledFrame]) -> list[SceneLabel]:
        """Send one batch of frames to the VLM for classification."""
        client = self._get_client()

        # Build valid game state values for the prompt
        valid_states = [s.value for s in GameState]

        prompt = (
            "You are analyzing a soccer match broadcast. "
            f"Here are {len(batch)} frames sampled every few seconds.\n"
            "For each frame, classify the game state as one of:\n"
            "- active_play: Ball in play, normal game action\n"
            "- corner_kick: Corner kick being set up or taken "
            "(player at corner flag, ball at corner arc)\n"
            "- goal_kick: Goal kick being set up or taken "
            "(ball placed in 6-yard box, GK about to kick)\n"
            "- stoppage: Other stoppage (throw-in, free kick, foul, injury)\n"
            "- replay: Broadcast replay / slow motion\n"
            "- other: Pre/post match, halftime, graphics\n\n"
            "Respond ONLY with a JSON array. Each element must have:\n"
            '  {"frame_index": <int>, "timestamp": <float>, "state": "<state>"}\n'
            f"Valid states: {valid_states}\n"
        )

        content: list[dict] = []
        for i, frame in enumerate(batch):
            b64 = base64.b64encode(frame.jpeg_bytes).decode("ascii")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
            content.append({
                "type": "text",
                "text": f"Frame {i} — timestamp {frame.timestamp_sec:.1f}s",
            })
        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        return self._parse_scan_response(response, batch)

    def _parse_scan_response(
        self, response, batch: list[SampledFrame]
    ) -> list[SceneLabel]:
        """Parse the coarse scan VLM response into SceneLabel objects."""
        text = response.content[0].text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("scene_analyzer.parse_error", error=str(exc), text=text[:300])
            # Fail-open: return active_play for all frames
            return [
                SceneLabel(timestamp_sec=f.timestamp_sec, game_state=GameState.ACTIVE_PLAY)
                for f in batch
            ]

        labels: list[SceneLabel] = []
        for item in items:
            try:
                idx = int(item.get("frame_index", -1))
                state_str = item.get("state", "active_play")
                try:
                    state = GameState(state_str)
                except ValueError:
                    state = GameState.ACTIVE_PLAY
                # Use the timestamp from our batch (more reliable than VLM echo)
                ts = batch[idx].timestamp_sec if 0 <= idx < len(batch) else item.get("timestamp", 0.0)
                labels.append(SceneLabel(timestamp_sec=ts, game_state=state))
            except (KeyError, IndexError, TypeError):
                continue

        return labels

    # ------------------------------------------------------------------
    # Pass 2: Boundary refinement
    # ------------------------------------------------------------------

    def refine_event(
        self,
        region_frames: list[SampledFrame],
        event_type: str,
        center_time: float,
    ) -> Optional[EventBoundary]:
        """Pinpoint exact clip boundaries for a detected event region.

        Args:
            region_frames: Dense frames (1-sec intervals) around the event.
            event_type: e.g. "corner_kick", "goal_kick".
            center_time: Approximate center of the detected event region.

        Returns:
            EventBoundary with confirmed=True/False and precise clip times,
            or None if the API call fails (fail-open handled by caller).
        """
        try:
            return self._refine_event_call(region_frames, event_type, center_time)
        except Exception as exc:
            log.error("scene_analyzer.refine_error", error=str(exc),
                      event_type=event_type, center_time=center_time)
            return None

    def _refine_event_call(
        self,
        region_frames: list[SampledFrame],
        event_type: str,
        center_time: float,
    ) -> EventBoundary:
        client = self._get_client()

        event_label = event_type.replace("_", " ")
        prompt = (
            f"I detected a possible {event_label} around {center_time:.1f}s in a soccer match. "
            f"Here are frames at 1-second intervals around that moment.\n\n"
            f"For a confirmed {event_label}, I expect to see:\n"
            "- The cause: the play that led to the ball going out\n"
            "- The setup: dead ball placed, players positioning\n"
            "- The execution: the kick taken\n"
            "- The aftermath: first few seconds of resulting play\n\n"
            "IMPORTANT: You MUST respond with ONLY a JSON object, no other text.\n"
            "If this is NOT a " + event_label + ", set confirmed to false.\n\n"
            '{"confirmed": true, "clip_start_sec": 100.0, '
            '"clip_end_sec": 120.0, "reasoning": "Clear goal kick at 110s"}\n'
            '{"confirmed": false, "clip_start_sec": 100.0, '
            '"clip_end_sec": 120.0, "reasoning": "This is active play, not a goal kick"}'
        )

        content: list[dict] = []
        for i, frame in enumerate(region_frames):
            b64 = base64.b64encode(frame.jpeg_bytes).decode("ascii")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
            content.append({
                "type": "text",
                "text": f"Frame {i} — timestamp {frame.timestamp_sec:.1f}s",
            })
        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": content}],
        )
        return self._parse_refine_response(response, event_type, center_time)

    def _parse_refine_response(
        self, response, event_type: str, center_time: float
    ) -> EventBoundary:
        """Parse the refinement VLM response into an EventBoundary."""
        text = response.content[0].text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            return EventBoundary(
                event_type=event_type,
                clip_start_sec=float(data.get("clip_start_sec", center_time - 10)),
                clip_end_sec=float(data.get("clip_end_sec", center_time + 10)),
                confirmed=bool(data.get("confirmed", False)),
                reasoning=str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.warning("scene_analyzer.refine_parse_error", error=str(exc), text=text[:300])
            # Fail-closed: non-JSON response usually means the VLM is explaining
            # why it can't confirm the event, so treat as unconfirmed.
            return EventBoundary(
                event_type=event_type,
                clip_start_sec=center_time - 10,
                clip_end_sec=center_time + 10,
                confirmed=False,
                reasoning=f"parse_error: {exc}",
            )
