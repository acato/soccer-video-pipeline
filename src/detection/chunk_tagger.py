"""
Chunk-based video tagger using vLLM-hosted vision model (Qwen3-VL).

Instead of classifying individual trajectory gaps, this approach:
1. Splits the full video into overlapping chunks (default 45s, 15s overlap)
2. Extracts each chunk at 4 FPS, scaled to 1280px width, via FFmpeg
3. Sends each chunk to vLLM, asking the model to tag events
4. Deduplicates events from overlapping chunk boundaries
5. Returns a unified event list

This gives the model full temporal context (build-up → event → aftermath)
and detects events that trajectory-gap analysis misses (e.g. goals).

Token budget (Qwen3-VL-32B-FP8, --max-model-len 31488):
    Observed: 120 frames (60s @ 2 FPS) uses ~24% KV cache (~7.6K tokens).
    Current: 45s × 4 FPS = 180 frames ≈ ~11K tokens (~36% KV).
    Encoder cache budget: 16,384 visual tokens (hard ceiling).
    Prompt/response overhead: ~2K tokens.
    Fits comfortably within 31,488 context and 16,384 encoder cache.

All prompts use negative-prompt inversion: the model must list reasons
each candidate is NOT a given event type before tagging it.  This
dramatically reduces hallucinated goals on ambiguous footage (fog, fast
cuts, distant camera).
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
    "kickoff": EventType.KICKOFF,
    "throw_in": EventType.THROW_IN,
}

# Event types that are goalkeeper events
_GK_TAG_TYPES = {"goal_kick", "corner_kick", "catch", "save", "penalty", "shot"}


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
        --max-model-len 31488 --kv-cache-dtype fp8
    """

    def __init__(
        self,
        vllm_url: str,
        model: str,
        source_file: str,
        match_config,
        job_id: str = "",
        chunk_duration_sec: float = 45.0,
        chunk_overlap_sec: float = 15.0,
        chunk_fps: int = 4,
        min_confidence: float = 0.5,
        working_dir: Optional[str] = None,
        rescan_fps: int = 8,
        rescan_pre_sec: float = 30.0,
        game_start_sec: float = 0.0,
        goals_only: bool = False,
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
        self._game_start_sec = game_start_sec
        self._goals_only = goals_only

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
        """Compute chunk start/end times with overlap.

        Starts from ``game_start_sec`` (with a 30s buffer before) to skip
        pre-game warmup footage.
        """
        step = self._chunk_duration - self._chunk_overlap
        if step <= 0:
            step = self._chunk_duration  # no overlap if misconfigured

        chunks = []
        start = max(0.0, self._game_start_sec - 30.0) if self._game_start_sec > 0 else 0.0
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
        """Extract video chunk at target FPS via FFmpeg.

        Uses ``-ss`` after ``-i`` for frame-accurate seeking and scales
        to 1280px width to reduce token consumption in the VLM.
        """
        duration = end - start
        chunk_path = self._debug_dir / f"chunk_{index:03d}_t{start:.0f}.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-i", self._source_file,
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-vf", f"fps={self._chunk_fps},scale=1280:-2",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
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

        prompt = self._build_prompt(chunk_start, chunk_end, goals_only=self._goals_only)

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

    def _build_prompt(
        self, chunk_start: float, chunk_end: float,
        goals_only: bool = False,
    ) -> str:
        """Build tagging prompt with match context.

        When *goals_only* is True a much shorter prompt is used that
        focuses exclusively on goals and kickoffs, improving recall by
        reducing the classification burden on the model.
        """
        if goals_only:
            return self._build_goals_only_prompt(chunk_start, chunk_end)
        return self._build_full_prompt(chunk_start, chunk_end)

    # ----- goals-only prompt --------------------------------------------------

    def _build_goals_only_prompt(
        self, chunk_start: float, chunk_end: float,
    ) -> str:
        """Short, focused prompt for goal + kickoff detection only.

        Uses negative-prompt inversion: the model must explain why each
        shot-like moment is NOT a goal before it may tag one as a goal.
        """
        mc = self._match_config
        duration = chunk_end - chunk_start
        start_m, start_s = divmod(int(chunk_start), 60)
        end_m, end_s = divmod(int(chunk_end), 60)

        return (
            f"You are analyzing a {duration:.0f}-second clip from a youth "
            f"soccer match at {self._chunk_fps} FPS. Fixed sideline camera.\n"
            f"\n"
            f"Match: {mc.team.team_name} "
            f"(outfield: {mc.team.outfield_color}, GK: {mc.team.gk_color}) vs "
            f"{mc.opponent.team_name} "
            f"(outfield: {mc.opponent.outfield_color}, GK: {mc.opponent.gk_color}).\n"
            f"Clip: {start_m}:{start_s:02d} - {end_m}:{end_s:02d}.\n"
            f"\n"
            f"TASK: Find every GOAL and every KICKOFF in this clip.\n"
            f"\n"
            f"=== STEP 1: DESCRIBE WHAT YOU SEE ===\n"
            f"List every moment where the ball moves toward a goal or a "
            f"restart formation appears. For each moment, write:\n"
            f"- The timestamp (seconds from clip start)\n"
            f"- What actually happens to the BALL (where does it end up?)\n"
            f"- What the GOALKEEPER does (catches, deflects, watches it go in, "
            f"or is not visible)\n"
            f"- What happens AFTER: do players celebrate by running/jumping/"
            f"hugging? Or does play continue normally?\n"
            f"\n"
            f"=== STEP 2: DISQUALIFY NON-GOALS ===\n"
            f"For each moment from Step 1, check this list. If ANY apply, "
            f"it is NOT a goal:\n"
            f"- The goalkeeper catches or holds the ball\n"
            f"- The goalkeeper deflects/pushes the ball and play continues\n"
            f"- The ball goes over the crossbar or wide of the post\n"
            f"- A defender blocks the ball\n"
            f"- Play continues without interruption (no restart, no "
            f"celebration)\n"
            f"- You CANNOT see the ball actually cross the goal line and "
            f"enter the net (it just disappears from view, or the camera "
            f"angle is unclear)\n"
            f"- The ball is near the goal but you are guessing rather than "
            f"seeing it in the net\n"
            f"\n"
            f"IMPORTANT: On foggy or low-visibility days, the ball often "
            f"disappears from view near the goal. Disappearing does NOT mean "
            f"it went in. If you cannot track the ball into the net, it is "
            f"NOT a goal. A save, a miss over the bar, and a goal can all "
            f"look identical in fog. You need CONFIRMATION (celebration, "
            f"restart from center) to distinguish them.\n"
            f"\n"
            f"=== STEP 3: TAG ONLY CONFIRMED EVENTS ===\n"
            f"\n"
            f"GOAL: Tag ONLY if you can answer YES to at least TWO of:\n"
            f"  a) You see the ball cross the goal line INTO the net\n"
            f"  b) Multiple players celebrate (run toward each other, jump, "
            f"hug, raise arms) — NOT just standing or walking\n"
            f"  c) Teams walk to their own halves and set up at the center "
            f"circle for a restart\n"
            f"  Confidence guide:\n"
            f"  0.90+: ball in net AND celebration visible\n"
            f"  0.75-0.85: strong celebration but ball-in-net unclear (fog)\n"
            f"  Below 0.70: do NOT tag it — insufficient evidence\n"
            f"  START: when the shot is taken\n"
            f"  END: when celebration peaks (do NOT include walk-back or "
            f"kickoff)\n"
            f"\n"
            f"KICKOFF: You MUST see ALL of these:\n"
            f"  1. Ball on the CENTER SPOT (exact middle of the field)\n"
            f"  2. Field CLEANLY SPLIT — one team per half, nobody crossing "
            f"the halfway line\n"
            f"  3. Only 1-2 players at the ball; everyone else far away\n"
            f"  4. The field looks organized and still\n"
            f"If players are clustered near the ball (like a wall), it is a "
            f"free kick, NOT a kickoff.\n"
            f"  START: when the ball is kicked from center\n"
            f"  END: same as start (single moment)\n"
            f"\n"
            f"Respond with your Step 1-2 analysis, then a JSON array "
            f"(no markdown):\n"
            f'[{{"event_type": "goal"|"kickoff",\n'
            f'  "start_sec": <seconds from start of THIS CLIP (0 = first frame)>,\n'
            f'  "end_sec": <seconds from start of THIS CLIP>,\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "team": "{mc.team.team_name}"|"{mc.opponent.team_name}"|"unknown",\n'
            f'  "reasoning": "what SPECIFIC visual evidence confirmed this '
            f'(ball in net + celebration, or center-circle formation)"}}]\n'
            f"\n"
            f"Rules:\n"
            f"- A shot where you THINK the ball might have gone in is NOT a "
            f"goal. You need to SEE it in the net or see unambiguous "
            f"celebration.\n"
            f"- \"The ball disappears toward the goal\" is NOT evidence of "
            f"a goal. The ball disappears on saves and misses too.\n"
            f"- \"The goalkeeper dives but fails to save\" is a guess, not "
            f"an observation, unless you can see the ball behind the keeper "
            f"in the net.\n"
            f"- KICKOFF is RARE (only after goals or at halftime).\n"
            f"- Most clips will have ZERO goals. Return [] confidently when "
            f"you do not see clear evidence.\n"
        )

    # ----- full all-events prompt ---------------------------------------------

    def _build_full_prompt(
        self, chunk_start: float, chunk_end: float,
    ) -> str:
        """Build all-events tagging prompt with match context.

        Uses negative-prompt inversion: for each candidate event the model
        must first list reasons it is NOT that event type, then tag only
        if no disqualifying reason applies.
        """
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
            f"This clip covers {start_m}:{start_s:02d} - {end_m}:{end_s:02d} "
            f"of the match video.\n"
            f"\n"
            f"=== STEP 1: SCAN AND DESCRIBE ===\n"
            f"Watch the entire clip. List every moment where the ball changes "
            f"possession, goes out of play, or moves toward a goal. For each "
            f"moment, describe ONLY what you can directly observe:\n"
            f"- Timestamp (seconds from clip start)\n"
            f"- Where is the ball? (field zone: left/center/right, near which "
            f"goal?)\n"
            f"- What happens to the ball? (kicked, thrown, caught, deflected, "
            f"goes out of bounds, enters net)\n"
            f"- Body posture of key players (standing still with ball on "
            f"ground, holding ball overhead, diving, running)\n"
            f"- What happens immediately AFTER? (play continues, celebration, "
            f"restart formation, corner setup)\n"
            f"\n"
            f"=== STEP 2: CLASSIFY EACH MOMENT ===\n"
            f"For each moment from Step 1, try to DISQUALIFY it from each "
            f"event type. Tag it ONLY as the type where no disqualifying "
            f"reason applies.\n"
            f"\n"
            f"EVENT TYPES AND DISQUALIFICATION RULES:\n"
            f"\n"
            f"GOAL — NOT a goal if ANY of these apply:\n"
            f"  - The goalkeeper catches or holds the ball\n"
            f"  - The goalkeeper deflects the ball and play continues or a "
            f"corner is taken\n"
            f"  - The ball goes over the crossbar or wide of the post\n"
            f"  - A defender blocks the ball\n"
            f"  - Play continues without interruption after the shot\n"
            f"  - You cannot see the ball cross the goal line into the net "
            f"(the ball just disappears from view -- this happens on saves "
            f"and misses too, especially in fog)\n"
            f"  - No celebration: players do not run toward each other, jump, "
            f"hug, or raise arms\n"
            f"  IS a goal ONLY if: (a) ball visibly enters the net OR "
            f"(b) unambiguous celebration (multiple players running/hugging) "
            f"AND teams set up at center for kickoff. Need at least (a) or "
            f"(b). \"Ball disappears toward goal\" alone is NOT enough.\n"
            f"  Confidence: 0.90+ ball in net + celebration; 0.75-0.85 "
            f"celebration only (fog obscures ball); below 0.70 do not tag.\n"
            f"  START: when shot is taken. END: celebration peak.\n"
            f"\n"
            f"PENALTY — NOT a penalty if:\n"
            f"  - Other players are inside the penalty box (not just keeper "
            f"and shooter)\n"
            f"  - The ball is not on the penalty spot (centered, 12 yards "
            f"from goal)\n"
            f"  - It looks like a free kick (wall of defenders visible)\n"
            f"  IS a penalty: box is empty except keeper on the goal line and "
            f"one shooter at the ball. All others behind the penalty arc.\n"
            f"  START: box clears. END: shot result is clear.\n"
            f"\n"
            f"SHOT — NOT a shot if:\n"
            f"  - The goalkeeper catches it (tag CATCH instead)\n"
            f"  - The goalkeeper deflects it out (tag SAVE instead)\n"
            f"  - The ball goes into the net (tag GOAL instead)\n"
            f"  IS a shot: ball struck toward goal, misses entirely (over "
            f"bar, wide), no keeper touch, goes out of play.\n"
            f"  START: ball struck. END: ball goes out.\n"
            f"\n"
            f"SAVE — NOT a save if:\n"
            f"  - The goalkeeper catches and holds the ball (tag CATCH)\n"
            f"  - The ball misses the goal entirely without keeper contact "
            f"(tag SHOT)\n"
            f"  - The ball goes into the net (tag GOAL)\n"
            f"  IS a save: shot on target, keeper touches/deflects it, ball "
            f"goes out of play (usually for a corner).\n"
            f"  START: shot taken. END: ball goes out.\n"
            f"\n"
            f"CATCH — NOT a catch if:\n"
            f"  - The goalkeeper punches or deflects the ball away (tag SAVE "
            f"or leave untagged)\n"
            f"  - The ball was not going toward goal (routine back-pass "
            f"pickup is not a catch)\n"
            f"  IS a catch: goalkeeper grabs and holds the ball securely "
            f"after a shot or cross.\n"
            f"  START: preceding shot/cross. END: keeper holds ball.\n"
            f"\n"
            f"CORNER_KICK — NOT a corner if:\n"
            f"  - The ball is not on the corner arc\n"
            f"  - The kick comes from the sideline (that is a free kick)\n"
            f"  IS a corner: ball placed on the corner arc at a corner flag, "
            f"kicked into the penalty area.\n"
            f"  START: ball placed on corner. END: ball leaves penalty area.\n"
            f"\n"
            f"GOAL_KICK — NOT a goal kick if:\n"
            f"  - A player holds the ball overhead with two hands (that is a "
            f"THROW_IN)\n"
            f"  - The ball is at the sideline, not at the end of the field "
            f"(that is a throw-in or free kick)\n"
            f"  - The ball is not on the ground inside the six-yard box\n"
            f"  - Opposing players are inside the penalty area\n"
            f"  IS a goal kick: ball on the ground in the six-yard box "
            f"(small box nearest the goal), kicked by keeper or defender, "
            f"no opponents in the penalty area. Only happens at the END of "
            f"the field, near a goal.\n"
            f"  START: ball placed. END: receiving player touches ball.\n"
            f"\n"
            f"FREE_KICK — NOT a free kick if:\n"
            f"  - The ball is thrown (hands overhead = throw-in)\n"
            f"  - The ball is on the corner arc (= corner kick)\n"
            f"  - The ball is in the six-yard box with no opponents in the "
            f"penalty area (= goal kick)\n"
            f"  IS a free kick: ball placed on the ground, player kicks "
            f"from a standstill.\n"
            f"  START: ball placed. END: ball kicked.\n"
            f"\n"
            f"THROW_IN — NOT a throw-in if:\n"
            f"  - The ball is kicked from the ground (= free kick or goal "
            f"kick)\n"
            f"  - The player is not at the sideline/touchline\n"
            f"  IS a throw-in: player at the sideline, ball held overhead "
            f"with both hands, thrown onto the field. Very common.\n"
            f"  START: ball held overhead. END: ball released.\n"
            f"\n"
            f"KICKOFF — NOT a kickoff if:\n"
            f"  - Players are clustered near the ball (= free kick)\n"
            f"  - The ball is not at the exact center spot\n"
            f"  - Teams are not cleanly split into their own halves\n"
            f"  - The field looks chaotic or players are running\n"
            f"  IS a kickoff: ball at center spot, one team per half, only "
            f"1-2 players at ball, field organized and still. RARE -- only "
            f"after goals or at halftime.\n"
            f"  START: ball kicked. END: same as start.\n"
            f"\n"
            f"=== OUTPUT ===\n"
            f"Write your Step 1 observations, then your Step 2 "
            f"disqualification reasoning, then a JSON array:\n"
            f'[{{"event_type": "goal"|"penalty"|"free_kick"|"shot"|'
            f'"corner_kick"|"goal_kick"|"catch"|"save"|"throw_in"|"kickoff",\n'
            f'  "start_sec": <seconds from start of THIS CLIP (0 = first frame)>,\n'
            f'  "end_sec": <seconds from start of THIS CLIP>,\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "team": "{mc.team.team_name}"|"{mc.opponent.team_name}"|"unknown",\n'
            f'  "reasoning": "what specific visual evidence you saw, and '
            f'which disqualifying reasons you checked and ruled out"}}]\n'
            f"\n"
            f"Rules:\n"
            f"- start_sec/end_sec = 0 means the first frame of this clip\n"
            f"- Only report events you clearly see -- do not guess\n"
            f"- If a shot results in a goal, catch, or save, tag the specific "
            f"outcome only (do not also tag as shot)\n"
            f'- "team" = the team performing the action (scoring team for '
            f"goals, goalkeeper's team for saves/catches, kicking team for "
            f"corners/goal kicks/free kicks)\n"
            f"- Most clips contain mostly throw-ins and routine play. Do NOT "
            f"over-tag. If in doubt, skip it.\n"
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
                    inferred = self._infer_goal_from_kickoff(ko, fps, events)
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
        events: list[Event] | None = None,
    ) -> Event:
        """Create a synthetic goal event inferred from an orphan kickoff.

        When a kickoff is detected but the goal is invisible at any FPS,
        we trust the kickoff: a goal must have happened before it.

        Anchoring strategy (in priority order):
        1. Last shot_on_target within 120s before the kickoff.
        2. Last event of any type within 90s (except kickoffs).
        3. Fallback to ko_t - 30s.
        """
        ko_t = kickoff.timestamp_abs
        mc = self._match_config

        # 1. Try anchoring to last shot within 120s
        preceding_shot = None
        if events:
            shots = [
                e for e in events
                if e.event_type in (EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET)
                and 0 < (ko_t - e.timestamp_start) < 120.0
            ]
            if shots:
                preceding_shot = max(shots, key=lambda e: e.timestamp_start)

        # 2. Fallback: last event of any type within 90s
        preceding_event = None
        if not preceding_shot and events:
            candidates = [
                e for e in events
                if 0 < (ko_t - e.timestamp_start) < 90.0
                and e.event_type != EventType.KICKOFF
            ]
            if candidates:
                preceding_event = max(candidates, key=lambda e: e.timestamp_start)

        if preceding_shot:
            goal_t = preceding_shot.timestamp_start
            goal_end = preceding_shot.timestamp_end
            team = preceding_shot.metadata.get("tagger_team", "unknown")
            confidence = 0.80  # kickoff + shot = strong evidence
            reasoning = (
                f"Inferred from orphan kickoff at {ko_t:.0f}s — "
                f"anchored to shot at {goal_t:.0f}s (likely misclassified goal)"
            )
        elif preceding_event:
            goal_t = preceding_event.timestamp_start
            goal_end = preceding_event.timestamp_end
            team = preceding_event.metadata.get("tagger_team", "unknown")
            confidence = 0.65
            reasoning = (
                f"Inferred from orphan kickoff at {ko_t:.0f}s — "
                f"anchored to {preceding_event.event_type.value} at {goal_t:.0f}s"
            )
        else:
            goal_t = ko_t - 30.0
            goal_end = ko_t - 10.0
            team = "unknown"
            confidence = 0.65
            reasoning = (
                f"Inferred from orphan kickoff at {ko_t:.0f}s — "
                f"kickoff confirmed but goal not visible at any FPS"
            )

        # Team attribution: if the shot/event was by our team, it's our goal
        # (not conceded). Otherwise assume conceded (GK event).
        is_gk = True  # default: goal conceded
        if team.lower() == mc.team.team_name.lower():
            is_gk = False  # our team scored

        frame_start = max(0, int(goal_t * fps))
        frame_end = max(frame_start, int(goal_end * fps))

        return Event(
            event_id=str(uuid.uuid4()),
            job_id=self._job_id,
            source_file=self._source_file,
            event_type=EventType.GOAL,
            timestamp_start=goal_t,
            timestamp_end=goal_end,
            confidence=confidence,
            reel_targets=[],
            is_goalkeeper_event=is_gk,
            frame_start=frame_start,
            frame_end=frame_end,
            metadata={
                "tagger_event_type": "goal",
                "tagger_confidence": confidence,
                "tagger_team": team,
                "tagger_reasoning": reasoning,
                "tagger_model": self._model,
                "inferred_from_kickoff": True,
                "kickoff_timestamp": ko_t,
                "anchored_to_shot": preceding_shot.timestamp_start if preceding_shot else None,
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
        """Build a focused prompt for scanning event gaps.

        Uses negative-prompt inversion: the model must explain why each
        candidate is NOT a goal/penalty before tagging.
        """
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
            f"This clip covers {start_m}:{start_s:02d} - {end_m}:{end_s:02d}.\n"
            f"\n"
            f"CONTEXT: There is a {gap_duration:.0f}-second gap with no detected "
            f"events during active play. This gap may contain a missed goal or "
            f"penalty kick.\n"
            f"\n"
            f"=== STEP 1: DESCRIBE ===\n"
            f"List every shot-like moment or restart formation you see. For "
            f"each one, describe:\n"
            f"- Timestamp\n"
            f"- What happens to the ball (where does it end up?)\n"
            f"- What the goalkeeper does\n"
            f"- What happens after (celebration? play continues? restart?)\n"
            f"\n"
            f"=== STEP 2: DISQUALIFY ===\n"
            f"For each moment, check these disqualifying reasons:\n"
            f"- Keeper catches or deflects the ball = NOT a goal\n"
            f"- Ball goes over the bar or wide = NOT a goal\n"
            f"- Play continues without interruption = NOT a goal\n"
            f"- Ball disappears from view without clear net entry = NOT "
            f"a goal (fog/distance makes saves look like goals)\n"
            f"- No visible celebration = probably NOT a goal\n"
            f"\n"
            f"PENALTY must show: empty box (only keeper + shooter), ball on "
            f"penalty spot. If other players are in the box or there is a "
            f"wall of defenders, it is NOT a penalty.\n"
            f"\n"
            f"=== STEP 3: TAG ===\n"
            f"Tag ONLY moments where no disqualifying reason applies.\n"
            f"\n"
            f"Write your analysis, then a JSON array:\n"
            f'[{{"event_type": "goal"|"penalty",\n'
            f'  "start_sec": <seconds from start of this clip>,\n'
            f'  "end_sec": <seconds from start of this clip>,\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "team": "{mc.team.team_name}"|"{mc.opponent.team_name}"|"unknown",\n'
            f'  "reasoning": "specific visual evidence and which '
            f'disqualifying reasons were checked"}}]\n'
            f"\n"
            f"Most clips in a gap contain routine play, not goals. "
            f"Return [] if no confirmed goal or penalty.\n"
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
            "-i", self._source_file,
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-vf", f"fps={self._rescan_fps},scale=1280:-2",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
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
        """Build a focused goal-finding prompt for high-FPS rescan.

        This prompt runs AFTER a kickoff was confirmed, so a goal almost
        certainly happened.  We still use negative-prompt inversion to
        ensure the model pinpoints the actual goal moment rather than
        tagging any random shot.
        """
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
            f"This clip covers {start_m}:{start_s:02d} - {end_m}:{end_s:02d}.\n"
            f"\n"
            f"CONTEXT: A kickoff from the center circle was confirmed at "
            f"{ko_m}:{ko_s:02d}. A goal was scored before that kickoff. "
            f"Find the exact moment of the goal in this clip.\n"
            f"\n"
            f"=== STEP 1: LIST ALL SHOTS ===\n"
            f"Describe every moment where the ball moves toward a goal. "
            f"For each, state:\n"
            f"- Timestamp\n"
            f"- What the ball does (enters net, saved, goes wide, goes over)\n"
            f"- What the goalkeeper does\n"
            f"- What happens after (celebration, play continues, corner)\n"
            f"\n"
            f"=== STEP 2: FIND THE GOAL ===\n"
            f"Only ONE of these shots is the actual goal. It is the one "
            f"where:\n"
            f"- The ball enters the net, OR\n"
            f"- Players celebrate (run, hug, jump) immediately after, OR\n"
            f"- Play stops and does not restart with a goal kick or corner "
            f"(because the restart is the kickoff at center)\n"
            f"\n"
            f"Disqualify shots where:\n"
            f"- The keeper saves it and play continues or a corner follows\n"
            f"- The ball goes over or wide and a goal kick follows\n"
            f"- Play continues without interruption\n"
            f"\n"
            f"Write your analysis, then a JSON array with at most ONE goal:\n"
            f'[{{"event_type": "goal",\n'
            f'  "start_sec": <seconds from start of this clip when shot is taken>,\n'
            f'  "end_sec": <seconds when celebration starts or play stops>,\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "team": "{mc.team.team_name}"|"{mc.opponent.team_name}"|"unknown",\n'
            f'  "reasoning": "which shot, what happened to the ball, what '
            f'disqualified the other shots"}}]\n'
            f"\n"
            f"If you see no clear goal moment in this clip, return: []\n"
            f"The goal may be in an adjacent clip -- do not force a tag.\n"
        )

    # ----- deduplication ----------------------------------------------------

    def _deduplicate(
        self, events: list[Event], proximity_sec: float = 10.0,
    ) -> list[Event]:
        """Remove duplicate events from overlapping chunk boundaries.

        For pairs of same-type events within proximity_sec of each other,
        keeps the one with higher confidence.  Then removes shots that
        are superseded by a nearby goal (cross-type dedup).
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

        return self._cross_type_dedup(deduped)

    @staticmethod
    def _cross_type_dedup(
        events: list[Event], proximity_sec: float = 15.0,
    ) -> list[Event]:
        """Remove shots superseded by a nearby goal.

        When a goal is inferred from a kickoff and anchored to a preceding
        shot, both the shot and the goal exist at the same timestamp.
        Keep only the goal.
        """
        goal_times = {
            e.timestamp_start for e in events
            if e.event_type == EventType.GOAL
        }
        if not goal_times:
            return events
        return [
            e for e in events
            if not (
                e.event_type in (EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET)
                and any(abs(e.timestamp_start - g) < proximity_sec for g in goal_times)
            )
        ]

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
