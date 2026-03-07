"""
vLLM-based classifier for dead-ball restart events.

Two-stage pipeline:
  Stage 1 — BallTouchDetector finds trajectory gaps (ball disappears ≥1.5s)
  Stage 2 — RestartClassifier extracts a short video clip around each gap,
            sends to a vLLM-hosted vision model via OpenAI-compatible API,
            and gets a structured classification:
            goal_kick / corner_kick / throw_in / free_kick / goal / other

Architecture:
    BallTouchDetector._find_trajectory_gaps() → GapCandidate list
         ↓
    RestartClassifier.classify_gaps(candidates, fps)
         ↓  FFmpeg extracts 12-15s video clip per gap
         ↓  vLLM model classifies each clip (native video understanding)
         ↓  Returns classified Event list
         ↓
    Worker extends event list → Segmentation → Assembly
"""
from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from src.detection.models import BoundingBox, Event, EventType

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Gap candidate — output of Stage 1 (trajectory gap detection)
# ---------------------------------------------------------------------------

@dataclass
class GapCandidate:
    """A trajectory gap where the ball disappeared, suggesting a dead-ball stoppage."""
    gap_start_frame: int
    gap_end_frame: int
    gap_start_ts: float
    gap_end_ts: float
    gap_duration_sec: float
    ball_pos_before: Optional[tuple[float, float]]  # last known (x, y)
    ball_pos_after: Optional[tuple[float, float]]    # first position after gap


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    event_type: str       # "goal_kick", "corner_kick", "throw_in", "free_kick", "goal", "other"
    confidence: float
    reasoning: str
    kick_timestamp_offset_sec: Optional[float]  # seconds from clip start


# ---------------------------------------------------------------------------
# Event type mapping
# ---------------------------------------------------------------------------

_RESTART_TYPE_TO_EVENT: dict[str, EventType] = {
    "goal_kick": EventType.GOAL_KICK,
    "corner_kick": EventType.CORNER_KICK,
    "goal": EventType.GOAL,
}


class RestartClassifier:
    """Classify dead-ball restart events using a vLLM-hosted vision model.

    For each trajectory gap candidate, extracts a short video clip,
    sends to vLLM via OpenAI-compatible API with video_url content type,
    and gets a structured classification:
    goal_kick / corner_kick / throw_in / free_kick / goal / other.
    """

    def __init__(
        self,
        vllm_url: str,
        model: str,
        source_file: str,
        match_config,
        job_id: str = "",
        clip_pre_sec: float = 5.0,
        clip_post_sec: float = 8.0,
        clip_width: int = 640,
        min_confidence: float = 0.5,
        target_types: Optional[set[str]] = None,
    ):
        self._vllm_url = vllm_url.rstrip("/")
        self._model = model
        self._source_file = source_file
        self._match_config = match_config
        self._job_id = job_id
        self._clip_pre_sec = clip_pre_sec
        self._clip_post_sec = clip_post_sec
        self._clip_width = clip_width
        self._min_confidence = min_confidence
        self._target_types = target_types or {"goal_kick", "corner_kick", "goal"}

    def classify_gaps(
        self, gaps: list[GapCandidate], fps: float,
    ) -> list[Event]:
        """Classify each gap candidate via vLLM video analysis.

        Returns only events whose type is in target_types and whose
        confidence meets the minimum threshold.
        """
        if not gaps:
            return []

        events: list[Event] = []
        for gap in gaps:
            try:
                clip_path = self._extract_clip(gap)
                if clip_path is None:
                    log.warning("vllm.clip_extraction_failed",
                                gap_start=gap.gap_start_ts)
                    continue

                result = self._classify_clip(clip_path, gap)

                # Clean up temp clip
                try:
                    clip_path.unlink(missing_ok=True)
                except OSError:
                    pass

                if result is None:
                    continue

                if (result.event_type in self._target_types
                        and result.confidence >= self._min_confidence):
                    event = self._make_event(gap, result, fps)
                    events.append(event)
                    log.info("vllm.classified",
                             event_type=result.event_type,
                             confidence=result.confidence,
                             reasoning=result.reasoning,
                             gap_start=gap.gap_start_ts)
                else:
                    log.debug("vllm.filtered_out",
                              event_type=result.event_type,
                              confidence=result.confidence,
                              gap_start=gap.gap_start_ts)

            except Exception as exc:
                log.error("vllm.classify_error",
                          gap_start=gap.gap_start_ts, error=str(exc))
                # Fail-open: skip this gap, don't crash the pipeline
                continue

        log.info("vllm.classification_complete",
                 gaps=len(gaps), events=len(events))
        return events

    def _extract_clip(
        self, gap: GapCandidate,
    ) -> Optional[Path]:
        """Extract video clip around gap via FFmpeg.

        Returns path to temporary MP4 file, or None on failure.
        """
        start = max(0.0, gap.gap_start_ts - self._clip_pre_sec)
        duration = (
            self._clip_pre_sec + gap.gap_duration_sec + self._clip_post_sec
        )

        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp4", prefix="vllm_clip_", delete=False,
        )
        tmp.close()
        clip_path = Path(tmp.name)

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", self._source_file,
            "-t", f"{duration:.3f}",
            "-vf", f"scale={self._clip_width}:-2",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-an",  # strip audio to reduce size
            str(clip_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=30,
            )
            if result.returncode == 0 and clip_path.stat().st_size > 0:
                return clip_path
            log.warning("vllm.ffmpeg_failed",
                        returncode=result.returncode,
                        gap_start=gap.gap_start_ts)
            clip_path.unlink(missing_ok=True)
            return None
        except subprocess.TimeoutExpired:
            log.warning("vllm.ffmpeg_timeout", gap_start=gap.gap_start_ts)
            clip_path.unlink(missing_ok=True)
            return None

    def _classify_clip(
        self, clip_path: Path, gap: GapCandidate,
    ) -> Optional[ClassificationResult]:
        """Send video clip to vLLM via OpenAI-compatible API."""
        import httpx

        prompt = self._build_prompt(gap)

        # Base64-encode the video clip as a data URL
        video_bytes = clip_path.read_bytes()
        video_b64 = base64.b64encode(video_bytes).decode()
        video_data_url = f"data:video/mp4;base64,{video_b64}"

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": video_data_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ],
            "max_tokens": 300,
        }

        try:
            resp = httpx.post(
                f"{self._vllm_url}/v1/chat/completions",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return self._parse_response(text)
        except Exception as exc:
            log.error("vllm.api_error", error=str(exc),
                      gap_start=gap.gap_start_ts)
            return None

    def _build_prompt(self, gap: GapCandidate) -> str:
        """Build classification prompt with match context."""
        mc = self._match_config

        # Ball position context
        pos_before = "unknown"
        pos_after = "unknown"
        if gap.ball_pos_before:
            pos_before = f"x={gap.ball_pos_before[0]:.2f}, y={gap.ball_pos_before[1]:.2f}"
        if gap.ball_pos_after:
            pos_after = f"x={gap.ball_pos_after[0]:.2f}, y={gap.ball_pos_after[1]:.2f}"

        return (
            f"You are analyzing a video clip from a youth soccer match "
            f"(sideline camera, fixed position).\n"
            f"Teams: {mc.team.team_name} "
            f"(outfield: {mc.team.outfield_color}, GK: {mc.team.gk_color}) vs "
            f"{mc.opponent.team_name} "
            f"(outfield: {mc.opponent.outfield_color}, GK: {mc.opponent.gk_color}).\n"
            f"\nThis clip shows a moment where ball tracking detected the ball "
            f"disappearing for {gap.gap_duration_sec:.1f} seconds, suggesting "
            f"a dead-ball stoppage.\n"
            f"Ball position before gap: {pos_before}\n"
            f"Ball position after gap: {pos_after}\n"
            f"(Coordinates: x=0 is left goal line, x=1 is right goal line; "
            f"y=0 is top sideline, y=1 is bottom sideline)\n"
            f"\nClassify this restart event. Respond ONLY with JSON:\n"
            f'{{"event_type": "goal_kick"|"corner_kick"|"throw_in"|'
            f'"free_kick"|"goal"|"other",\n'
            f' "confidence": 0.0-1.0,\n'
            f' "reasoning": "brief explanation",\n'
            f' "kick_timestamp_offset_sec": <seconds from clip start '
            f"when the restart kick occurs>}}\n"
            f"\nKey visual cues:\n"
            f"- Goal kick: GK places ball on 6-yard box line, kicks upfield. "
            f"Few players nearby.\n"
            f"- Corner kick: Ball placed at corner flag, player kicks into "
            f"crowded penalty area.\n"
            f"- Throw-in: Player holds ball with both hands, throws from sideline.\n"
            f"- Free kick: Ball placed on ground away from goal, wall of players "
            f"may be present.\n"
            f"- Goal: Ball in the net, celebration.\n"
            f"- Other: None of the above (e.g., substitution, injury, halftime).\n"
        )

    def _parse_response(self, text: str) -> Optional[ClassificationResult]:
        """Parse model response into a ClassificationResult.

        Handles both clean JSON and JSON embedded in markdown code blocks.
        Returns None on parse failure.
        """
        if not text or not text.strip():
            log.warning("vllm.empty_response")
            return None

        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try to extract JSON from text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]

        try:
            data = json.loads(text)
            event_type = str(data.get("event_type", "other")).lower()
            confidence = float(data.get("confidence", 0.0))
            reasoning = str(data.get("reasoning", ""))
            kick_offset = data.get("kick_timestamp_offset_sec")

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            if kick_offset is not None:
                kick_offset = float(kick_offset)

            return ClassificationResult(
                event_type=event_type,
                confidence=confidence,
                reasoning=reasoning,
                kick_timestamp_offset_sec=kick_offset,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.warning("vllm.parse_error", text=text[:200], error=str(exc))
            return None

    def _make_event(
        self, gap: GapCandidate, result: ClassificationResult, fps: float,
    ) -> Event:
        """Create an Event from a gap candidate and classification result."""
        event_type = _RESTART_TYPE_TO_EVENT.get(
            result.event_type, EventType.GOAL_KICK
        )

        # Use kick timestamp if available, otherwise center of gap
        if result.kick_timestamp_offset_sec is not None:
            clip_start = gap.gap_start_ts - self._clip_pre_sec
            kick_ts = clip_start + result.kick_timestamp_offset_sec
        else:
            kick_ts = (gap.gap_start_ts + gap.gap_end_ts) / 2

        # For goal kicks, is_goalkeeper_event=True; for corners, True too
        is_gk = event_type in (
            EventType.GOAL_KICK, EventType.CORNER_KICK,
        )

        bbox = None
        if gap.ball_pos_after:
            bx, by = gap.ball_pos_after
            bbox = BoundingBox(
                x=max(0, bx - 0.02), y=max(0, by - 0.02),
                width=0.04, height=0.04,
            )

        return Event(
            event_id=str(uuid.uuid4()),
            job_id=self._job_id,
            source_file=self._source_file,
            event_type=event_type,
            timestamp_start=kick_ts - 0.5,
            timestamp_end=kick_ts + 0.5,
            confidence=result.confidence,
            reel_targets=[],
            is_goalkeeper_event=is_gk,
            frame_start=max(0, int((kick_ts - 0.5) * fps)),
            frame_end=int((kick_ts + 0.5) * fps),
            bounding_box=bbox,
            metadata={
                "vllm_event_type": result.event_type,
                "vllm_confidence": result.confidence,
                "vllm_reasoning": result.reasoning,
                "vllm_model": self._model,
                "vllm_kick_offset": result.kick_timestamp_offset_sec,
                "gap_start_ts": gap.gap_start_ts,
                "gap_end_ts": gap.gap_end_ts,
                "gap_duration_sec": gap.gap_duration_sec,
            },
        )
