"""
Chunk-based video tagger using vLLM-hosted vision model (Qwen3-VL).

Instead of classifying individual trajectory gaps, this approach:
1. Splits the full video into overlapping 2.5-minute chunks
2. Extracts each chunk at 2 FPS (source resolution) via FFmpeg
3. Sends each chunk to vLLM, asking the model to tag ALL events
4. Deduplicates events from overlapping chunk boundaries
5. Returns a unified event list

This gives the model full temporal context (build-up → event → aftermath)
and detects events that trajectory-gap analysis misses (e.g. goals).

VRAM budget (48 GB, Qwen3-VL-32B-FP8, FP8 KV cache):
    150s × 2 FPS = 300 frames × 256 tok/frame = 76,800 visual tokens
    + ~1,200 tokens prompt/response = ~78,000 total
    Fits in 90K max context with --max-model-len 90000 --kv-cache-dtype fp8
"""
from __future__ import annotations

import base64
import json
import math
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

from src.detection.models import BoundingBox, Event, EventType

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Event type mapping: model output → pipeline EventType
# ---------------------------------------------------------------------------

_TAG_TO_EVENT: dict[str, EventType] = {
    "goal_kick": EventType.GOAL_KICK,
    "corner_kick": EventType.CORNER_KICK,
    "goal": EventType.GOAL,
    "save_catch": EventType.CATCH,
    "save_parry": EventType.SHOT_STOP_DIVING,
    "punch": EventType.PUNCH,
}

# Event types that are goalkeeper events
_GK_TAG_TYPES = {"goal_kick", "corner_kick", "save_catch", "save_parry", "punch"}


# ---------------------------------------------------------------------------
# Tagged event — raw output from model before conversion to Event
# ---------------------------------------------------------------------------

@dataclass
class TaggedEvent:
    """A single event tagged by the model in one chunk."""
    event_type: str       # "goal_kick", "corner_kick", "goal", etc.
    timestamp_abs: float  # Absolute timestamp in the source video (seconds)
    confidence: float
    team: str             # team name, opponent name, or "unknown"
    reasoning: str


class ChunkTagger:
    """Tag all events in a video via chunked vLLM analysis.

    Splits the video into overlapping chunks, sends each to a
    vLLM-hosted vision model, collects and deduplicates tagged events.

    Requires vLLM server configured with sufficient context:
        --max-model-len 90000 --kv-cache-dtype fp8
    """

    def __init__(
        self,
        vllm_url: str,
        model: str,
        source_file: str,
        match_config,
        job_id: str = "",
        chunk_duration_sec: float = 150.0,
        chunk_overlap_sec: float = 15.0,
        chunk_fps: int = 2,
        min_confidence: float = 0.5,
        working_dir: Optional[str] = None,
    ):
        self._vllm_url = vllm_url.rstrip("/")
        self._model = model
        self._source_file = source_file
        self._match_config = match_config
        self._job_id = job_id
        self._chunk_duration = chunk_duration_sec
        self._chunk_overlap = chunk_overlap_sec
        self._chunk_fps = chunk_fps
        self._min_confidence = min_confidence

        # Debug output directory
        if working_dir:
            self._debug_dir = Path(working_dir) / job_id / "chunk_tagger"
        else:
            self._debug_dir = Path("/tmp/soccer-pipeline") / job_id / "chunk_tagger"
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._debug_log = self._debug_dir / "tags.jsonl"

    def tag_video(
        self,
        video_duration: float,
        fps: float,
        progress_callback=None,
    ) -> list[Event]:
        """Tag all events in the video via chunked vLLM analysis.

        Args:
            video_duration: Total video duration in seconds.
            fps: Source video frame rate.
            progress_callback: Optional callable(completed, total) for
                progress reporting.

        Returns list of deduplicated Event objects.
        """
        chunks = self._compute_chunks(video_duration)
        total = len(chunks)
        log.info("chunk_tagger.start",
                 chunks=total,
                 chunk_duration=self._chunk_duration,
                 overlap=self._chunk_overlap,
                 fps=self._chunk_fps,
                 video_duration=video_duration)

        all_events: list[Event] = []

        for i, (start, end) in enumerate(chunks):
            try:
                chunk_path = self._extract_chunk(start, end, i)
                if chunk_path is None:
                    log.warning("chunk_tagger.extract_failed",
                                chunk=i, start=start, end=end)
                    continue

                tagged, raw_response = self._tag_chunk(chunk_path, start, end, fps)

                self._save_debug(i, start, end, raw_response, tagged)

                for te in tagged:
                    if (te.event_type in _TAG_TO_EVENT
                            and te.confidence >= self._min_confidence):
                        event = self._make_event(te, fps)
                        all_events.append(event)
                        log.info("chunk_tagger.tagged",
                                 event_type=te.event_type,
                                 timestamp=te.timestamp_abs,
                                 confidence=te.confidence,
                                 team=te.team,
                                 reasoning=te.reasoning)

            except Exception as exc:
                log.error("chunk_tagger.chunk_error",
                          chunk=i, start=start, error=str(exc))
                continue
            finally:
                if progress_callback:
                    progress_callback(i + 1, total)

        deduped = self._deduplicate(all_events)
        log.info("chunk_tagger.complete",
                 chunks=total,
                 raw_events=len(all_events),
                 deduped_events=len(deduped),
                 debug_dir=str(self._debug_dir))
        return deduped

    # ----- chunk computation ------------------------------------------------

    def _compute_chunks(self, video_duration: float) -> list[tuple[float, float]]:
        """Compute chunk start/end times with overlap."""
        step = self._chunk_duration - self._chunk_overlap
        if step <= 0:
            step = self._chunk_duration  # no overlap if misconfigured

        chunks = []
        start = 0.0
        while start < video_duration:
            end = min(start + self._chunk_duration, video_duration)
            # Skip very short trailing chunks (< 10s)
            if end - start < 10.0 and chunks:
                # Extend previous chunk to cover the rest
                prev_start, _ = chunks[-1]
                chunks[-1] = (prev_start, video_duration)
                break
            chunks.append((start, end))
            start += step

        return chunks

    # ----- FFmpeg chunk extraction ------------------------------------------

    def _extract_chunk(
        self, start: float, end: float, index: int,
    ) -> Optional[Path]:
        """Extract video chunk at target FPS via FFmpeg."""
        duration = end - start
        chunk_path = self._debug_dir / f"chunk_{index:03d}_t{start:.0f}.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", self._source_file,
            "-t", f"{duration:.3f}",
            "-vf", f"fps={self._chunk_fps}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-an",
            str(chunk_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=120,
            )
            if result.returncode == 0 and chunk_path.exists() and chunk_path.stat().st_size > 0:
                log.debug("chunk_tagger.extracted",
                          chunk=index, path=str(chunk_path),
                          size_mb=chunk_path.stat().st_size / 1e6)
                return chunk_path
            log.warning("chunk_tagger.ffmpeg_failed",
                        returncode=result.returncode,
                        stderr=result.stderr[-500:] if result.stderr else "")
            chunk_path.unlink(missing_ok=True)
            return None
        except subprocess.TimeoutExpired:
            log.warning("chunk_tagger.ffmpeg_timeout", chunk=index)
            chunk_path.unlink(missing_ok=True)
            return None

    # ----- vLLM API call ----------------------------------------------------

    def _tag_chunk(
        self, chunk_path: Path, chunk_start: float, chunk_end: float, fps: float,
    ) -> tuple[list[TaggedEvent], str]:
        """Send chunk to vLLM and parse tagged events.

        Returns (tagged_events, raw_response_text).
        """
        import httpx

        prompt = self._build_prompt(chunk_start, chunk_end)

        video_bytes = chunk_path.read_bytes()
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
            "max_tokens": 2048,
        }

        try:
            resp = httpx.post(
                f"{self._vllm_url}/v1/chat/completions",
                json=payload,
                timeout=600.0,
            )
            if resp.status_code != 200:
                body = resp.text[:500]
                log.error("chunk_tagger.api_error",
                          status=resp.status_code, body=body,
                          chunk_start=chunk_start)
                return [], f"HTTP_{resp.status_code}: {body}"
            data = resp.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            events = self._parse_response(text, chunk_start)
            return events, text
        except Exception as exc:
            log.error("chunk_tagger.api_error",
                      error=str(exc), chunk_start=chunk_start)
            return [], f"API_ERROR: {exc}"

    # ----- prompt construction ----------------------------------------------

    def _build_prompt(self, chunk_start: float, chunk_end: float) -> str:
        """Build tagging prompt with match context."""
        mc = self._match_config
        duration = chunk_end - chunk_start

        # Format timestamps as mm:ss
        start_m, start_s = divmod(int(chunk_start), 60)
        end_m, end_s = divmod(int(chunk_end), 60)

        return (
            f"You are analyzing a {duration:.0f}-second video clip from a youth "
            f"soccer match. Camera: fixed sideline position. "
            f"Video: {self._chunk_fps} frames per second.\n"
            f"\n"
            f"Match: {mc.team.team_name} "
            f"(outfield: {mc.team.outfield_color}, GK: {mc.team.gk_color}) vs "
            f"{mc.opponent.team_name} "
            f"(outfield: {mc.opponent.outfield_color}, GK: {mc.opponent.gk_color}).\n"
            f"This clip covers {start_m}:{start_s:02d} – {end_m}:{end_s:02d} "
            f"of the match video.\n"
            f"\n"
            f"Tag EVERY event you see from this list:\n"
            f"\n"
            f"GOAL — Ball enters the net. Signs: shot crossing goal line, "
            f"player celebration (running, arms raised, team huddle), "
            f"dejected opponents, restart from center circle.\n"
            f"\n"
            f"GOAL_KICK — Goalkeeper places ball on or near the 6-yard box, "
            f"kicks long upfield. Usually after a missed shot or cleared cross. "
            f"Few players near goal, GK alone with ball.\n"
            f"\n"
            f"CORNER_KICK — Ball placed at corner flag, kicked into penalty "
            f"area. Players crowding the box, waiting for the cross.\n"
            f"\n"
            f"SAVE_CATCH — Goalkeeper catches and holds the ball securely with "
            f"both hands after a shot. Ball is secured, not released.\n"
            f"\n"
            f"SAVE_PARRY — Goalkeeper deflects, pushes, or tips the ball away "
            f"from the goal. Ball continues in play or goes out. GK makes a "
            f"reaching or diving motion to redirect the ball.\n"
            f"\n"
            f"PUNCH — Goalkeeper punches the ball away with fist(s), usually "
            f"from a cross or corner kick. Ball struck upward/outward by fist.\n"
            f"\n"
            f"Respond ONLY with a JSON array (no markdown, no extra text). "
            f"Each event:\n"
            f'[{{"event_type": "goal"|"goal_kick"|"corner_kick"|'
            f'"save_catch"|"save_parry"|"punch",\n'
            f'  "timestamp_sec": <seconds from START of this clip>,\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "team": "{mc.team.team_name}"|"{mc.opponent.team_name}"|"unknown",\n'
            f'  "reasoning": "brief description"}}]\n'
            f"\n"
            f"Rules:\n"
            f"- timestamp_sec = 0 means the first frame of this clip\n"
            f"- Only report events you clearly see — do not guess\n"
            f"- For saves: timestamp is when the GK contacts the ball\n"
            f"- For goals: timestamp is when the ball crosses the goal line\n"
            f"- For goal kicks and corners: timestamp is when the ball is kicked\n"
            f"- If no events, return: []\n"
        )

    # ----- response parsing -------------------------------------------------

    def _parse_response(
        self, text: str, chunk_start: float,
    ) -> list[TaggedEvent]:
        """Parse model response into TaggedEvent list.

        Handles both clean JSON and JSON embedded in markdown code blocks.
        Returns empty list on parse failure.
        """
        if not text or not text.strip():
            log.warning("chunk_tagger.empty_response")
            return []

        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Extract JSON array from text
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start:end + 1]

        try:
            items = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("chunk_tagger.parse_error", text=text[:200], error=str(exc))
            return []

        if not isinstance(items, list):
            log.warning("chunk_tagger.not_a_list", type=type(items).__name__)
            return []

        events = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                event_type = str(item.get("event_type", "")).lower()
                ts_sec = float(item.get("timestamp_sec", 0))
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
                team = str(item.get("team", "unknown"))
                reasoning = str(item.get("reasoning", ""))

                events.append(TaggedEvent(
                    event_type=event_type,
                    timestamp_abs=chunk_start + ts_sec,
                    confidence=confidence,
                    team=team,
                    reasoning=reasoning,
                ))
            except (TypeError, ValueError) as exc:
                log.warning("chunk_tagger.item_parse_error",
                            item=str(item)[:100], error=str(exc))
                continue

        return events

    # ----- event creation ---------------------------------------------------

    def _make_event(self, te: TaggedEvent, fps: float) -> Event:
        """Create an Event from a TaggedEvent."""
        event_type = _TAG_TO_EVENT[te.event_type]

        # GK events: all save types + goal kicks + corners
        is_gk = te.event_type in _GK_TAG_TYPES

        # Goal conceded (opponent scored) is also a GK event
        mc = self._match_config
        if (te.event_type == "goal"
                and te.team.lower() == mc.opponent.team_name.lower()):
            is_gk = True

        frame = max(0, int(te.timestamp_abs * fps))

        return Event(
            event_id=str(uuid.uuid4()),
            job_id=self._job_id,
            source_file=self._source_file,
            event_type=event_type,
            timestamp_start=te.timestamp_abs - 0.5,
            timestamp_end=te.timestamp_abs + 0.5,
            confidence=te.confidence,
            reel_targets=[],
            is_goalkeeper_event=is_gk,
            frame_start=max(0, frame - int(fps * 0.5)),
            frame_end=frame + int(fps * 0.5),
            metadata={
                "tagger_event_type": te.event_type,
                "tagger_confidence": te.confidence,
                "tagger_team": te.team,
                "tagger_reasoning": te.reasoning,
                "tagger_model": self._model,
            },
        )

    # ----- deduplication ----------------------------------------------------

    def _deduplicate(
        self, events: list[Event], proximity_sec: float = 10.0,
    ) -> list[Event]:
        """Remove duplicate events from overlapping chunk boundaries.

        For pairs of same-type events within proximity_sec of each other,
        keeps the one with higher confidence.
        """
        if not events:
            return []

        events.sort(key=lambda e: (e.event_type.value, e.timestamp_start))

        deduped: list[Event] = [events[0]]
        for e in events[1:]:
            prev = deduped[-1]
            if (e.event_type == prev.event_type
                    and abs(e.timestamp_start - prev.timestamp_start) < proximity_sec):
                # Same event in overlap zone — keep higher confidence
                if e.confidence > prev.confidence:
                    deduped[-1] = e
            else:
                deduped.append(e)

        return deduped

    # ----- debug logging ----------------------------------------------------

    def _save_debug(
        self,
        index: int,
        start: float,
        end: float,
        raw_response: str,
        tagged: list[TaggedEvent],
    ) -> None:
        """Append debug record to JSONL log."""
        record = {
            "chunk_index": index,
            "chunk_start": start,
            "chunk_end": end,
            "raw_response": raw_response,
            "events": [
                {
                    "event_type": te.event_type,
                    "timestamp_abs": te.timestamp_abs,
                    "confidence": te.confidence,
                    "team": te.team,
                    "reasoning": te.reasoning,
                }
                for te in tagged
            ],
        }
        try:
            with open(self._debug_log, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass
