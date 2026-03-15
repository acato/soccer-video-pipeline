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
        rescan_fps: int = 8,
        rescan_pre_sec: float = 30.0,
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
        self._rescan_fps = rescan_fps
        self._rescan_pre_sec = rescan_pre_sec

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
        all_kickoffs: list[TaggedEvent] = []

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
                    if te.event_type == "kickoff":
                        all_kickoffs.append(te)
                        log.info("chunk_tagger.kickoff_detected",
                                 timestamp=te.timestamp_abs,
                                 team=te.team,
                                 reasoning=te.reasoning)
                        continue
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

        # --- Pass 2: rescan orphan kickoffs at high FPS ---
        rescan_events = self._rescan_orphan_kickoffs(
            all_kickoffs, all_events, fps, video_duration,
        )
        all_events.extend(rescan_events)

        # --- Pass 3: scan event gaps for missed goals/penalties ---
        gap_events = self._scan_event_gaps(all_events, fps, video_duration)
        all_events.extend(gap_events)

        deduped = self._deduplicate(all_events)
        log.info("chunk_tagger.complete",
                 chunks=total,
                 raw_events=len(all_events),
                 kickoffs=len(all_kickoffs),
                 rescans=len(rescan_events),
                 gap_scans=len(gap_events),
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
            "temperature": 0.0,
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
            f"PENALTY — The penalty box clears: ALL players except the "
            f"goalkeeper and one shooter leave the box. The ball is placed on "
            f"the penalty spot (12 yards from goal, centered). The shooter "
            f"runs up alone and kicks. The key visual cue: an unusually empty "
            f"box with only two players — one at the goal line, one at the "
            f"ball. All other players wait behind the penalty arc.\n"
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
            f"box (the small box closest to the goal). The goalkeeper or a "
            f"defender kicks it while standing still. No opposing player is "
            f"inside the penalty area. The ball is kicked from the GROUND — "
            f"NOT thrown. Do NOT confuse with a throw-in (player holding ball "
            f"overhead at the sideline).\n"
            f"  START: when the ball is placed in the six-yard box\n"
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
            f"KICKOFF — Play restarts from the CENTER CIRCLE (the large circle "
            f"at the exact midpoint of the field). All players are standing in "
            f"their own half. Two players stand over the ball at the center "
            f"spot. This ONLY happens after a goal or at halftime — NOT after "
            f"any other stoppage. Do NOT confuse with a free kick (which "
            f"happens anywhere on the field, not at center).\n"
            f"  START: when the ball is kicked from center\n"
            f"  END: same as start (single moment)\n"
            f"\n"
            f"Respond ONLY with a JSON array (no markdown, no extra text):\n"
            f'[{{"event_type": "goal"|"penalty"|"free_kick"|"shot"|'
            f'"corner_kick"|"goal_kick"|"catch"|"save"|"kickoff",\n'
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
            f"- Do NOT tag throw-ins (player holding ball overhead at the "
            f"sideline and throwing it in) — they are not goal kicks\n"
            f"- A KICKOFF only happens from the center circle after a goal or "
            f"at halftime — any other restart from midfield is a free kick\n"
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

    # ----- kickoff rescan (pass 2) -----------------------------------------

    def _rescan_orphan_kickoffs(
        self,
        kickoffs: list[TaggedEvent],
        events: list[Event],
        fps: float,
        video_duration: float,
    ) -> list[Event]:
        """Find kickoffs not preceded by a detected goal, rescan at high FPS.

        A kickoff not preceded by a goal within 90s is an "orphan" — it implies
        a goal the first pass missed (fast play, low FPS). Halftime kickoffs
        (preceded by a >120s gap with no events) are excluded.

        For each orphan, extract the preceding 30s at 8 FPS and re-query
        the model with a focused goal-finding prompt.
        """
        if not kickoffs:
            return []

        # Dedup kickoffs within 15s of each other (overlap zones)
        kickoffs_sorted = sorted(kickoffs, key=lambda k: k.timestamp_abs)
        deduped_kickoffs: list[TaggedEvent] = [kickoffs_sorted[0]]
        for ko in kickoffs_sorted[1:]:
            if ko.timestamp_abs - deduped_kickoffs[-1].timestamp_abs > 15.0:
                deduped_kickoffs.append(ko)

        # All event timestamps for gap checking
        event_times = sorted(e.timestamp_start for e in events)

        orphans: list[TaggedEvent] = []
        for ko in deduped_kickoffs:
            ko_t = ko.timestamp_abs
            # Skip kickoffs in the first 60s (match start)
            if ko_t < 60.0:
                log.info("chunk_tagger.kickoff_skip_start", timestamp=ko_t)
                continue

            # Check if a goal was already detected within 90s before
            has_goal = any(
                e.event_type == EventType.GOAL
                and 0 < (ko_t - e.timestamp_start) < 90.0
                for e in events
            )
            if has_goal:
                log.info("chunk_tagger.kickoff_has_goal", timestamp=ko_t)
                continue

            # Check for halftime: no events in the 120s before the kickoff
            latest_before = max(
                (t for t in event_times if t < ko_t - 10.0), default=0.0,
            )
            gap = ko_t - latest_before
            if gap > 120.0:
                log.info("chunk_tagger.kickoff_skip_halftime",
                         timestamp=ko_t, gap=gap)
                continue

            orphans.append(ko)
            log.info("chunk_tagger.orphan_kickoff",
                     timestamp=ko_t, gap_to_last_event=gap)

        if not orphans:
            return []

        log.info("chunk_tagger.rescan_start", orphan_kickoffs=len(orphans),
                 rescan_fps=self._rescan_fps)

        rescan_events: list[Event] = []
        inferred_count = 0
        for ko in orphans:
            try:
                goal_events = self._rescan_for_goal(ko, fps, video_duration)
                if goal_events:
                    rescan_events.extend(goal_events)
                else:
                    # Rescan found nothing — trust the kickoff and infer a goal
                    inferred = self._infer_goal_from_kickoff(ko, fps)
                    rescan_events.append(inferred)
                    inferred_count += 1
                    log.info("chunk_tagger.goal_inferred",
                             kickoff_t=ko.timestamp_abs,
                             goal_t=inferred.timestamp_start)
            except Exception as exc:
                log.error("chunk_tagger.rescan_error",
                          kickoff_t=ko.timestamp_abs, error=str(exc))

        log.info("chunk_tagger.rescan_complete",
                 orphans=len(orphans),
                 goals_found=len(rescan_events) - inferred_count,
                 goals_inferred=inferred_count)
        return rescan_events

    def _infer_goal_from_kickoff(
        self, kickoff: TaggedEvent, fps: float,
    ) -> Event:
        """Create a synthetic goal event inferred from an orphan kickoff.

        When a kickoff is detected but the goal is invisible at any FPS,
        we trust the kickoff: a goal must have happened ~20s before it.
        """
        ko_t = kickoff.timestamp_abs
        # Goal typically happens 15-30s before the kickoff (celebration + walk)
        goal_t = ko_t - 20.0
        goal_end = ko_t - 5.0  # celebration ends ~5s before kickoff

        # Attribute to opponent by default (goal conceded = GK event),
        # unless kickoff team info suggests otherwise
        mc = self._match_config
        team = "unknown"
        is_gk = True  # assume goal conceded until proven otherwise

        frame_start = max(0, int(goal_t * fps))
        frame_end = max(frame_start, int(goal_end * fps))

        return Event(
            event_id=str(uuid.uuid4()),
            job_id=self._job_id,
            source_file=self._source_file,
            event_type=EventType.GOAL,
            timestamp_start=goal_t,
            timestamp_end=goal_end,
            confidence=0.70,  # lower confidence for inferred events
            reel_targets=[],
            is_goalkeeper_event=is_gk,
            frame_start=frame_start,
            frame_end=frame_end,
            metadata={
                "tagger_event_type": "goal",
                "tagger_confidence": 0.70,
                "tagger_team": team,
                "tagger_reasoning": (
                    f"Inferred from orphan kickoff at {ko_t:.0f}s — "
                    f"kickoff confirmed but goal not visible at any FPS"
                ),
                "tagger_model": self._model,
                "inferred_from_kickoff": True,
                "kickoff_timestamp": ko_t,
            },
        )

    # ----- event gap scan (pass 3) -------------------------------------------

    def _scan_event_gaps(
        self,
        events: list[Event],
        fps: float,
        video_duration: float,
        min_gap_sec: float = 90.0,
    ) -> list[Event]:
        """Find long gaps between events and rescan for missed goals/penalties.

        After passes 1+2, if there's a gap >90s where events exist before and
        after (active play, not halftime), it may contain a missed goal or
        penalty. Rescan each gap at high FPS with a focused prompt.
        """
        if not events:
            return []

        timestamps = sorted(e.timestamp_start for e in events)
        gaps: list[tuple[float, float]] = []

        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap >= min_gap_sec:
                gap_start = timestamps[i]
                gap_end = timestamps[i + 1]
                # Skip halftime-like gaps (>5 min = probably halftime break)
                if gap > 300.0:
                    log.info("chunk_tagger.gap_skip_halftime",
                             start=gap_start, end=gap_end, gap=gap)
                    continue
                gaps.append((gap_start, gap_end))

        if not gaps:
            return []

        log.info("chunk_tagger.gap_scan_start", gaps=len(gaps),
                 rescan_fps=self._rescan_fps)

        gap_events: list[Event] = []
        for gap_start, gap_end in gaps:
            try:
                found = self._rescan_gap(gap_start, gap_end, fps, video_duration)
                gap_events.extend(found)
            except Exception as exc:
                log.error("chunk_tagger.gap_scan_error",
                          start=gap_start, end=gap_end, error=str(exc))

        log.info("chunk_tagger.gap_scan_complete",
                 gaps=len(gaps), events_found=len(gap_events))
        return gap_events

    def _rescan_gap(
        self,
        gap_start: float,
        gap_end: float,
        fps: float,
        video_duration: float,
    ) -> list[Event]:
        """Rescan a single event gap at high FPS for goals/penalties."""
        # Scan the full gap region in 15s chunks
        chunk_dur = 15.0
        rescan_chunks: list[tuple[float, float]] = []
        t = gap_start
        while t < gap_end:
            c_end = min(t + chunk_dur, gap_end)
            if c_end - t >= 3.0:
                rescan_chunks.append((t, c_end))
            t += chunk_dur

        log.info("chunk_tagger.gap_rescan",
                 gap_start=gap_start, gap_end=gap_end,
                 gap_sec=gap_end - gap_start,
                 chunks=len(rescan_chunks),
                 fps=self._rescan_fps)

        found_events: list[Event] = []
        for i, (start, end) in enumerate(rescan_chunks):
            chunk_path = self._extract_rescan_chunk(
                start, end, kickoff_t=gap_start, index=i,
            )
            if chunk_path is None:
                continue

            prompt = self._build_gap_prompt(start, end, gap_start, gap_end)

            video_bytes = chunk_path.read_bytes()
            video_b64 = base64.b64encode(video_bytes).decode()
            video_data_url = f"data:video/mp4;base64,{video_b64}"

            import httpx
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
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
            }

            try:
                resp = httpx.post(
                    f"{self._vllm_url}/v1/chat/completions",
                    json=payload,
                    timeout=600.0,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                tagged = self._parse_response(text, start)
                self._save_debug(
                    index=f"gap_{gap_start:.0f}_{i}",
                    start=start, end=end,
                    raw_response=text, tagged=tagged,
                )

                for te in tagged:
                    if (te.event_type in ("goal", "penalty")
                            and te.confidence >= self._min_confidence):
                        event = self._make_event(te, fps)
                        event.metadata["gap_scan"] = True
                        event.metadata["gap_start"] = gap_start
                        event.metadata["gap_end"] = gap_end
                        event.metadata["gap_scan_fps"] = self._rescan_fps
                        found_events.append(event)
                        log.info("chunk_tagger.gap_event_found",
                                 event_type=te.event_type,
                                 timestamp=te.timestamp_abs,
                                 confidence=te.confidence,
                                 team=te.team,
                                 reasoning=te.reasoning)
            except Exception as exc:
                log.error("chunk_tagger.gap_api_error",
                          start=start, end=end, error=str(exc))

        return found_events

    def _build_gap_prompt(
        self,
        chunk_start: float,
        chunk_end: float,
        gap_start: float,
        gap_end: float,
    ) -> str:
        """Build a focused prompt for scanning event gaps."""
        mc = self._match_config
        duration = chunk_end - chunk_start
        gap_duration = gap_end - gap_start

        start_m, start_s = divmod(int(chunk_start), 60)
        end_m, end_s = divmod(int(chunk_end), 60)

        return (
            f"You are analyzing a {duration:.0f}-second video clip from a youth "
            f"soccer match at {self._rescan_fps} frames per second (high detail).\n"
            f"\n"
            f"Match: {mc.team.team_name} "
            f"(outfield: {mc.team.outfield_color}, GK: {mc.team.gk_color}) vs "
            f"{mc.opponent.team_name} "
            f"(outfield: {mc.opponent.outfield_color}, GK: {mc.opponent.gk_color}).\n"
            f"This clip covers {start_m}:{start_s:02d} – {end_m}:{end_s:02d}.\n"
            f"\n"
            f"CONTEXT: There is a {gap_duration:.0f}-second gap with no detected "
            f"events during active play. This gap may contain a missed goal or "
            f"penalty kick. Look carefully for:\n"
            f"\n"
            f"1. PENALTY: A single player standing over the ball at the penalty "
            f"   spot (~12 yards from goal), with only the goalkeeper defending. "
            f"   All other players stand outside the penalty area. The player "
            f"   runs up and shoots.\n"
            f"2. GOAL: A shot that goes into the net, followed by celebration. "
            f"   Could be from open play, a penalty kick, or a free kick.\n"
            f"3. KICKOFF: Teams lined up in their halves, ball kicked from the "
            f"   center circle (confirms a goal was scored before this).\n"
            f"\n"
            f"Respond ONLY with a JSON array:\n"
            f'[{{"event_type": "goal"|"penalty",\n'
            f'  "start_sec": <seconds from start of this clip>,\n'
            f'  "end_sec": <seconds from start of this clip>,\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "team": "{mc.team.team_name}"|"{mc.opponent.team_name}"|"unknown",\n'
            f'  "reasoning": "brief description"}}]\n'
            f"\n"
            f"If you see nothing significant, return: []\n"
        )

    def _rescan_for_goal(
        self,
        kickoff: TaggedEvent,
        fps: float,
        video_duration: float,
    ) -> list[Event]:
        """Extract high-FPS chunks before a kickoff and find the missed goal."""
        ko_t = kickoff.timestamp_abs
        scan_start = max(0.0, ko_t - self._rescan_pre_sec)
        scan_end = ko_t

        # Split into 15s chunks to stay within VRAM budget
        chunk_dur = 15.0
        rescan_chunks: list[tuple[float, float]] = []
        t = scan_start
        while t < scan_end:
            c_end = min(t + chunk_dur, scan_end)
            if c_end - t >= 3.0:  # skip tiny trailing chunks
                rescan_chunks.append((t, c_end))
            t += chunk_dur

        log.info("chunk_tagger.rescan_goal",
                 kickoff_t=ko_t, scan_start=scan_start,
                 scan_end=scan_end, chunks=len(rescan_chunks),
                 fps=self._rescan_fps)

        goal_events: list[Event] = []
        for i, (start, end) in enumerate(rescan_chunks):
            chunk_path = self._extract_rescan_chunk(start, end, ko_t, i)
            if chunk_path is None:
                continue

            tagged, raw_response = self._tag_rescan_chunk(
                chunk_path, start, end, ko_t, fps,
            )
            self._save_debug(
                index=f"rescan_ko{ko_t:.0f}_{i}",
                start=start, end=end,
                raw_response=raw_response, tagged=tagged,
            )

            for te in tagged:
                if te.event_type == "goal" and te.confidence >= self._min_confidence:
                    event = self._make_event(te, fps)
                    event.metadata["rescan_kickoff_t"] = ko_t
                    event.metadata["rescan_fps"] = self._rescan_fps
                    goal_events.append(event)
                    log.info("chunk_tagger.rescan_goal_found",
                             timestamp=te.timestamp_abs,
                             confidence=te.confidence,
                             team=te.team,
                             reasoning=te.reasoning)

        return goal_events

    def _extract_rescan_chunk(
        self, start: float, end: float, kickoff_t: float, index: int,
    ) -> Optional[Path]:
        """Extract a high-FPS chunk for goal rescan."""
        duration = end - start
        chunk_path = (
            self._debug_dir
            / f"rescan_ko{kickoff_t:.0f}_{index}_t{start:.0f}.mp4"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", self._source_file,
            "-t", f"{duration:.3f}",
            "-vf", f"fps={self._rescan_fps}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-an",
            str(chunk_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if (result.returncode == 0 and chunk_path.exists()
                    and chunk_path.stat().st_size > 0):
                log.debug("chunk_tagger.rescan_extracted",
                          path=str(chunk_path),
                          size_mb=chunk_path.stat().st_size / 1e6)
                return chunk_path
            chunk_path.unlink(missing_ok=True)
            return None
        except subprocess.TimeoutExpired:
            chunk_path.unlink(missing_ok=True)
            return None

    def _tag_rescan_chunk(
        self,
        chunk_path: Path,
        chunk_start: float,
        chunk_end: float,
        kickoff_t: float,
        fps: float,
    ) -> tuple[list[TaggedEvent], str]:
        """Send a high-FPS rescan chunk to vLLM with a goal-focused prompt."""
        import httpx

        prompt = self._build_rescan_prompt(chunk_start, chunk_end, kickoff_t)

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
            "max_tokens": 1024,
            "temperature": 0.0,
        }

        try:
            resp = httpx.post(
                f"{self._vllm_url}/v1/chat/completions",
                json=payload,
                timeout=600.0,
            )
            if resp.status_code != 200:
                body = resp.text[:500]
                log.error("chunk_tagger.rescan_api_error",
                          status=resp.status_code, body=body)
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
            log.error("chunk_tagger.rescan_api_error", error=str(exc))
            return [], f"API_ERROR: {exc}"

    def _build_rescan_prompt(
        self, chunk_start: float, chunk_end: float, kickoff_t: float,
    ) -> str:
        """Build a focused goal-finding prompt for high-FPS rescan."""
        mc = self._match_config
        duration = chunk_end - chunk_start

        start_m, start_s = divmod(int(chunk_start), 60)
        end_m, end_s = divmod(int(chunk_end), 60)
        ko_m, ko_s = divmod(int(kickoff_t), 60)

        return (
            f"You are analyzing a {duration:.0f}-second video clip from a youth "
            f"soccer match at {self._rescan_fps} frames per second (high detail).\n"
            f"\n"
            f"Match: {mc.team.team_name} "
            f"(outfield: {mc.team.outfield_color}, GK: {mc.team.gk_color}) vs "
            f"{mc.opponent.team_name} "
            f"(outfield: {mc.opponent.outfield_color}, GK: {mc.opponent.gk_color}).\n"
            f"This clip covers {start_m}:{start_s:02d} – {end_m}:{end_s:02d}.\n"
            f"\n"
            f"CONTEXT: A kickoff from the center circle was detected at "
            f"{ko_m}:{ko_s:02d}, which means a goal was scored shortly before. "
            f"Find the goal in this clip.\n"
            f"\n"
            f"Look for: a fast attack or breakout, a shot on goal, the ball "
            f"going into the net, and the goalkeeper failing to stop it. The "
            f"goal may happen very quickly (1-2 seconds from shot to net).\n"
            f"\n"
            f"Respond ONLY with a JSON array:\n"
            f'[{{"event_type": "goal",\n'
            f'  "start_sec": <seconds from start of this clip when the shot is taken>,\n'
            f'  "end_sec": <seconds from start of this clip when celebration starts>,\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "team": "{mc.team.team_name}"|"{mc.opponent.team_name}"|"unknown",\n'
            f'  "reasoning": "brief description of what you see"}}]\n'
            f"\n"
            f"If you do not see a goal in this clip, return: []\n"
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
        index,
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
