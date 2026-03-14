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
    "goal": EventType.GOAL,
    "penalty": EventType.PENALTY,
    "free_kick": EventType.FREE_KICK_SHOT,
    "shot": EventType.SHOT_ON_TARGET,
    "corner_kick": EventType.CORNER_KICK,
    "goal_kick": EventType.GOAL_KICK,
    "catch": EventType.CATCH,
    "save": EventType.SHOT_STOP_DIVING,
}

# Event types that are goalkeeper events
_GK_TAG_TYPES = {"goal_kick", "corner_kick", "catch", "save", "penalty"}


# ---------------------------------------------------------------------------
# Tagged event — raw output from model before conversion to Event
# ---------------------------------------------------------------------------

@dataclass
class TaggedEvent:
    """A single event tagged by the model in one chunk."""
    event_type: str       # "goal_kick", "corner_kick", "goal", etc.
    timestamp_abs: float  # Absolute start timestamp in the source video (seconds)
    timestamp_end_abs: float  # Absolute end timestamp in the source video (seconds)
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
            f"Tag EVERY event you see. For each event, report BOTH a start and "
            f"end timestamp that bound the full sequence.\n"
            f"\n"
            f"EVENT TYPES AND OBSERVATION CHAINS:\n"
            f"\n"
            f"GOAL — A shot goes into the net. Confirmed by: players celebrate, "
            f"teams walk to their own halves, kickoff from center circle. The "
            f"kickoff restart is what proves it was a goal (not a save or miss).\n"
            f"  START: when the shot is taken\n"
            f"  END: when the celebration ends (do NOT include teams walking "
            f"to halves or kickoff)\n"
            f"\n"
            f"PENALTY — Everyone except the referee, the goalkeeper, and one "
            f"player leaves the penalty box. The ball is set down on the penalty "
            f"spot. Only the shooter approaches and takes the shot.\n"
            f"  START: when the box clears (only keeper and shooter remain)\n"
            f"  END: when the shot result is clear (goal, save, or miss)\n"
            f"\n"
            f"FREE_KICK — The ball is placed on the ground and kicked with no "
            f"opposing player approaching. Players may form a wall at distance.\n"
            f"  START: when the ball is placed down\n"
            f"  END: when the ball is kicked\n"
            f"\n"
            f"SHOT — The ball, touched last by a member of the attacking team, "
            f"travels towards the goal and goes out past the back line without "
            f"the goalkeeper touching it (a miss). If the keeper touches it, "
            f"tag as SAVE or CATCH instead. If it goes in the net, tag as GOAL.\n"
            f"  START: when the ball is struck towards goal\n"
            f"  END: when the ball goes out of play\n"
            f"\n"
            f"CORNER_KICK — The ball is placed on the corner arc of the field. "
            f"A player kicks it into the penalty area.\n"
            f"  START: when the ball is placed on the corner\n"
            f"  END: when the ball leaves the penalty area after being kicked\n"
            f"\n"
            f"GOAL_KICK — The ball is placed on the ground inside the six-yard "
            f"box. The goalkeeper or a defender kicks it out with no opposing "
            f"player inside the box.\n"
            f"  START: when the ball is placed in the box\n"
            f"  END: when a receiving player touches or controls the ball\n"
            f"\n"
            f"CATCH — The goalkeeper grabs the ball and holds it securely.\n"
            f"  START: when the preceding shot or cross is played\n"
            f"  END: when the goalkeeper secures the ball\n"
            f"\n"
            f"SAVE — A shot is taken, the goalkeeper touches or deflects the "
            f"ball, and the ball goes out for a corner kick.\n"
            f"  START: when the shot is taken\n"
            f"  END: when the ball goes out for a corner\n"
            f"\n"
            f"Respond ONLY with a JSON array (no markdown, no extra text):\n"
            f'[{{"event_type": "goal"|"penalty"|"free_kick"|"shot"|'
            f'"corner_kick"|"goal_kick"|"catch"|"save",\n'
            f'  "start_sec": <seconds from start of this clip when event begins>,\n'
            f'  "end_sec": <seconds from start of this clip when event ends>,\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "team": "{mc.team.team_name}"|"{mc.opponent.team_name}"|"unknown",\n'
            f'  "reasoning": "brief description"}}]\n'
            f"\n"
            f"Rules:\n"
            f"- start_sec/end_sec = 0 means the first frame of this clip\n"
            f"- Only report events you clearly see — do not guess\n"
            f"- A GOAL must be confirmed by a center-circle kickoff. No kickoff "
            f"= save or miss, not a goal\n"
            f"- A SAVE always ends with a corner kick. If the keeper holds the "
            f"ball, that is a CATCH, not a save\n"
            f"- If a shot results in a goal, catch, or save, tag the specific "
            f"outcome — do not also tag it as SHOT\n"
            f'- "team" = the team performing the action (scoring team for goals, '
            f"goalkeeper's team for saves/catches, kicking team for corners/"
            f"goal kicks/free kicks)\n"
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
                # Support both start_sec/end_sec and legacy timestamp_sec
                start_sec = float(item.get("start_sec",
                                           item.get("timestamp_sec", 0)))
                end_sec = float(item.get("end_sec", start_sec + 1.0))
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
                team = str(item.get("team", "unknown"))
                reasoning = str(item.get("reasoning", ""))

                events.append(TaggedEvent(
                    event_type=event_type,
                    timestamp_abs=chunk_start + start_sec,
                    timestamp_end_abs=chunk_start + end_sec,
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
        # Penalty by opponent = shooting at our keeper = GK event
        mc = self._match_config
        opp = mc.opponent.team_name.lower()
        if te.event_type in ("goal", "penalty") and te.team.lower() == opp:
            is_gk = True

        frame_start = max(0, int(te.timestamp_abs * fps))
        frame_end = max(frame_start, int(te.timestamp_end_abs * fps))

        return Event(
            event_id=str(uuid.uuid4()),
            job_id=self._job_id,
            source_file=self._source_file,
            event_type=event_type,
            timestamp_start=te.timestamp_abs,
            timestamp_end=te.timestamp_end_abs,
            confidence=te.confidence,
            reel_targets=[],
            is_goalkeeper_event=is_gk,
            frame_start=frame_start,
            frame_end=frame_end,
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
                    "timestamp_end_abs": te.timestamp_end_abs,
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
