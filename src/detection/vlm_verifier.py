"""
VLM classification (Phase 3) — two-pass architecture.

Pass 1 (Observe): "Describe what you see" — free-form observation
Pass 2 (Classify): Feed observation back + classification question

Extended clip windows: 5s pre / 15s post to capture celebrations
and post-event context that distinguishes goals from saves.

Dependencies: httpx (vLLM), anthropic (Claude fallback), FFmpeg.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import structlog

from src.detection.models import EventType
from src.detection.visual_candidate import EventCandidate, CandidateSource
from src.ingestion.models import MatchConfig

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class VerificationResult(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class VLMVerdict:
    """VLM verification result for one event candidate."""
    candidate: EventCandidate
    result: VerificationResult
    event_type: Optional[EventType]      # Classified type, or None if rejected
    confidence: float                     # VLM confidence 0-1
    reasoning: str                        # Model's explanation
    model_used: str                       # Which model produced this


# ---------------------------------------------------------------------------
# Prompts — two-pass architecture
# ---------------------------------------------------------------------------

_OBSERVE_PROMPT = """\
You are analyzing frames from a soccer match recorded by a sideline camera.
{match_context}

IMPORTANT: These frames are sampled at 2 FPS from a {clip_duration:.0f}-second clip. \
The action happens between players who may be small in frame due to the wide-angle sideline camera.

Describe what you observe in these frames. Focus on:
1. Where is the ball? Is it moving toward/away from a goal?
2. What are the players doing? (running, celebrating, standing still, clustered)
3. Is there a goalkeeper making a save or picking up the ball?
4. Are players celebrating (arms raised, running toward each other, group hugs)?
5. Is there a restart (kick-off from center, goal kick, corner, free kick)?
6. Is there a penalty kick setup (one player vs goalkeeper, others outside box)?

Be specific about what you SEE, not what you infer."""

_CLASSIFY_PROMPT = """\
Based on your observation of the soccer clip:

"{observation}"

Now classify this moment. Choose the SINGLE best match:
- "goal": A goal was scored (see STRICT GOAL RULES below)
- "save": Goalkeeper stopped a shot — GK touched/blocked the ball, ball did NOT enter the net
- "shot": Shot toward goal that missed or was blocked (NOT by GK hands)
- "corner_kick": Ball placed at corner arc and kicked into the box
- "goal_kick": Goalkeeper or defender kicks from the six-yard box after ball went out over goal line
- "free_kick": Ball placed on ground, kicked from a stoppage
- "penalty": Penalty kick — one shooter vs goalkeeper from the penalty spot
- "throw_in": Player throws ball in from the sideline
- "kickoff": Kick-off from center circle (start of half or after a goal)
- "none": Normal play, nothing significant, or cannot determine

STRICT GOAL RULES — you MUST see at least ONE of these confirmations to classify as "goal":
1. Players CELEBRATING: arms raised, running to teammates, group hugs, sliding on knees
2. Teams walking back to center circle for a KICKOFF restart
3. Ball CLEARLY inside the net (not just near the goal line)
If you only see a shot toward goal but NO celebration, NO kickoff setup, and the ball is NOT clearly in the net, classify as "shot" or "save" instead. "Ball near the goal line" alone is NEVER enough for a goal.

Other distinctions:
- SAVE: GK touches/blocks ball, then play continues or restarts from goal kick/corner
- SHOT: ball kicked toward goal but goes wide, over, or is blocked by a defender (not GK)
- Penalty: single shooter facing GK from penalty spot, all others outside the box

Respond with EXACTLY this JSON (no other text):
{{"event": "<type>", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

_CLASSIFY_SINGLE_PASS_PROMPT = """\
You are analyzing {num_frames} frames sampled at 2 FPS from a {clip_duration:.0f}-second clip \
of a soccer match recorded by a SIDELINE CAMERA at ~50 metres from the field.
{match_context}

CAMERA LIMITATIONS — at this distance:
- You CANNOT see whether the goalkeeper's hands touched the ball
- You CANNOT see the ball cross the goal line with certainty
- You CAN see player positions, formations, celebrations, and restart patterns

CLASSIFICATION STRATEGY — what happens AFTER the action is the best signal:
- Kickoff restart (players at center circle) → classify as "goal" (a goal was scored)
- Throw-in restart (player holding ball at sideline) → classify as "throw_in"
- Goal kick or corner kick restart → could be "save", "shot", or "goal_kick" (see rules below)
- Play continues normally → "none"

DISTINGUISHING "save" vs "shot" vs "goal_kick":
- "save" (CATCH): After a shot, the goalkeeper is HOLDING or CARRYING the ball — \
cradling it to their chest, picking it up off the ground, or standing with the ball \
in their hands. This is the result state, not the action. Look for the GK with the \
ball in hand in any frame AFTER the shot.
- "save" (PARRY): The goalkeeper DIVES, JUMPS, or REACHES and the ball changes \
direction near the GK. A corner kick restart after a shot is strong parry evidence, \
but classify as "shot" here — the pipeline handles parry inference structurally.
- "shot": A shot was taken toward goal and a goal kick follows, but you do NOT see \
the goalkeeper holding the ball or making a clear save action. The ball likely went \
WIDE or OVER the goal.
- "goal_kick": NO shot preceded the goal kick — a wayward cross or backpass drifted \
over the goal line with no shot attempt.

When uncertain between "save" and "shot", classify as "shot".

Classify the MAIN event. Choose ONE:
- "goal": GOAL — ONLY if you see celebration (arms raised, group hugs, sliding) \
OR kickoff restart at center circle. "Ball near goal" alone is NEVER enough.
- "save": SAVE (catch) — after a shot, the goalkeeper is HOLDING the ball (cradling, \
carrying, picking up). Look for the GK with ball in hands in frames AFTER the shot.
- "shot": SHOT — ball kicked toward goal. If followed by a goal kick but you see \
NO clear goalkeeper save action, classify as "shot" (ball went wide/over). \
If unsure whether it was a save or a shot, classify as "shot".
- "corner_kick": ball placed at CORNER FLAG arc, kicked into the penalty box.
- "goal_kick": ball kicked from six-yard box with NO preceding shot attempt — \
a wayward cross or backpass drifted out. If a shot was taken first, classify as \
"shot" or "save" instead (depending on whether GK made a save action).
- "free_kick": ball placed on ground mid-field, kicked from a stoppage after a foul.
- "penalty": ONE player facing goalkeeper from the penalty spot, all others outside the box.
- "throw_in": player holding ball OVERHEAD at the sideline, throws it in.
- "kickoff": kick-off from CENTER CIRCLE — two players over the ball at the center spot.
- "none": normal play, nothing significant, or cannot determine.

Respond with EXACTLY this JSON (no other text):
{{"event": "<type>", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


_SET_PIECE_CHECK_PROMPT = """\
You are analyzing frames from a soccer match recorded by a sideline camera.
{match_context}

These frames are sampled at 2 FPS from a {clip_duration:.0f}-second clip.

A shot or save just happened. What restart follows? Choose ONE:

CORNER KICK:
- Ball placed at the corner arc (near the corner flag at the edge of the field)
- One player standing by the corner flag, about to kick
- Many players gathered in or near the penalty area, waiting for the cross
- Ball kicked into the box (high, curving delivery)

GOAL KICK:
- Ball placed on the ground inside the six-yard box (small rectangle near the goal)
- Goalkeeper or defender standing over the ball, about to kick it
- Opposing players far from the goal, outside the penalty area
- Ball kicked long upfield or short to a defender

THROW-IN:
- Player holding the ball overhead at the sideline
- Both feet on/behind the touchline (side of the field)
- Ball thrown into play

NONE:
- Play continues without a restart
- Cannot determine the restart type
- Players are actively running/contesting

Respond with EXACTLY this JSON (no other text):
{{"restart": "corner_kick" or "goal_kick" or "throw_in" or "none", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

_CORNER_CHECK_PROMPT = """\
You are analyzing frames from a soccer match recorded by a sideline camera.
{match_context}

These frames are sampled at 2 FPS from a {clip_duration:.0f}-second clip.

Is a CORNER KICK being taken in these frames?

Signs of a corner kick:
- A player standing at or near the CORNER FLAG / CORNER ARC at the edge of the field
- The ball placed on the ground at the corner arc
- Multiple players from both teams gathered inside or around the PENALTY AREA, waiting for the cross
- The ball is kicked high into the box (curving/lofted delivery)
- The corner flag is clearly visible near the player taking the kick

This is NOT a corner kick if:
- Players are running/contesting the ball in open play
- The ball is in the middle of the field or near the sideline (not corner)
- A player is taking a goal kick (ball in the six-yard box, not corner)
- A player is taking a throw-in (holding ball overhead at the sideline)
- A free kick from a central or wide position (not at the corner arc)

Respond with EXACTLY this JSON (no other text):
{{"is_corner": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

_KICKOFF_CHECK_PROMPT = """\
You are analyzing frames from a soccer match recorded by a sideline camera.
{match_context}

These frames are sampled at 2 FPS from a {clip_duration:.0f}-second clip.

Is this a CENTER CIRCLE KICKOFF restart (the kind that happens after a goal is scored)?

Signs of a post-goal kickoff:
- Players from both teams lined up on their own half of the field
- Ball placed at the center spot (middle of the field)
- One or two players standing near the center circle ready to kick
- Players NOT actively running or contesting — they are stationary, waiting

This is NOT a kickoff if:
- Players are running/contesting the ball in the midfield (that's normal play)
- The ball is near a goal or sideline (that's a different restart)
- Players are spread across the field in active play

Respond with EXACTLY this JSON (no other text):
{{"is_kickoff": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


_SHOT_CHECK_PROMPT = """\
You are analyzing {num_frames} frames sampled at 2 FPS from a {clip_duration:.0f}-second clip \
of a soccer match recorded by a SIDELINE CAMERA at ~50 metres from the field.
{match_context}

Was a SHOT taken toward goal in these frames?

Signs of a shot:
- A player strikes/kicks the ball forcefully TOWARD THE GOAL
- The ball travels at speed in the direction of the goal
- Players react — ducking, jumping, turning to watch the ball
- The goalkeeper dives, jumps, or moves laterally
- Defenders throw themselves at the ball to block
- A brief crowd reaction (excitement or groaning)

Signs this is NOT a shot:
- Ball is passed sideways or backwards between teammates
- A clearance kicked AWAY from goal (upfield)
- A goal kick or free kick taken from the team's own half
- Players jogging or walking — no sudden burst of activity
- Normal midfield play with no clear strike on the ball

Respond with EXACTLY this JSON (no other text):
{{"is_shot": true, "shot_type": "on_target" or "off_target" or "blocked", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
or
{{"is_shot": false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


_CATCH_CHECK_PROMPT = """\
You are analyzing {num_frames} frames sampled at 2 FPS from a {clip_duration:.0f}-second clip \
of a soccer match recorded by a SIDELINE CAMERA at ~50 metres from the field.
{match_context}

CONTEXT: A shot toward goal was just taken. These frames show the AFTERMATH \
(3-8 seconds after the shot). Did the goalkeeper CATCH the ball?

Signs of a CATCH (goalkeeper holding the ball):
- Goalkeeper CRADLING the ball against their chest or stomach
- Goalkeeper STANDING or KNEELING with the ball in both hands
- Goalkeeper getting up from the ground with the ball SECURED in their arms
- Goalkeeper WALKING or MOVING while holding the ball (about to distribute)
- Ball clearly visible IN the goalkeeper's hands/arms (not on the ground)
- After catching, goalkeeper may look upfield to distribute

Signs this is NOT a catch:
- Ball is on the ground near the goalkeeper (not in their hands)
- Ball is bouncing or rolling away from the goalkeeper
- Ball is in the air or being played by other players
- Goalkeeper is diving but the ball is loose or deflected
- Normal play — ball is far from the goalkeeper
- Players are taking a goal kick (ball on the ground in the 6-yard box)

Respond with EXACTLY this JSON (no other text):
{{"is_catch": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


def _match_context(match_config: Optional[MatchConfig]) -> str:
    if not match_config:
        return ""
    return (
        f"Match: {match_config.team.team_name} vs {match_config.opponent.team_name}. "
        f"{match_config.team.team_name} wears {match_config.team.outfield_color} "
        f"(GK: {match_config.team.gk_color}). "
        f"{match_config.opponent.team_name} wears {match_config.opponent.outfield_color} "
        f"(GK: {match_config.opponent.gk_color})."
    )


# Map VLM response strings to EventType
_EVENT_TYPE_MAP = {
    "goal": EventType.GOAL,
    "save": EventType.SHOT_STOP_DIVING,
    "shot": EventType.SHOT_ON_TARGET,
    "corner_kick": EventType.CORNER_KICK,
    "goal_kick": EventType.GOAL_KICK,
    "free_kick": EventType.FREE_KICK_SHOT,
    "penalty": EventType.PENALTY,
    "throw_in": EventType.THROW_IN,
    "kickoff": EventType.KICKOFF,
}


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class VLMVerifier:
    """Classify event candidates using a vision-language model.

    Two-pass architecture:
      Pass 1: "Describe what you see" (open-ended observation)
      Pass 2: Feed observation + classification prompt

    Falls back to single-pass if two-pass fails or times out.

    Supports two backends:
      - vLLM (Qwen3-VL via OpenAI-compatible API) — primary
      - Claude API — fallback

    Usage::

        verifier = VLMVerifier(vllm_url="http://10.10.2.222:8000")
        verdicts = verifier.verify(candidates, match_config=mc)
    """

    _CLIP_FPS = 2         # 2 FPS — longer clips need fewer frames per second
    _CLIP_WIDTH = 768     # Frame width (768px ≈ 200 tok/frame)
    _MAX_FRAMES = 24      # 12 seconds × 2 FPS = 24 frames
    _TWO_PASS = False     # Single-pass: 2x candidates at same GPU cost

    def __init__(
        self,
        *,
        vllm_url: Optional[str] = None,
        vllm_model: str = "Qwen/Qwen3-VL-32B-Instruct-FP8",
        anthropic_api_key: Optional[str] = None,
        anthropic_model: str = "claude-sonnet-4-20250514",
        min_confidence: float = 0.5,
        source_file: Optional[str | Path] = None,
        working_dir: Optional[str | Path] = None,
    ):
        self._vllm_url = vllm_url
        self._vllm_model = vllm_model
        self._anthropic_key = anthropic_api_key
        self._anthropic_model = anthropic_model
        self._min_conf = min_confidence
        self._source = Path(source_file) if source_file else None

        if working_dir:
            self._work = Path(working_dir)
        else:
            self._work = Path("/tmp/soccer-pipeline")
        self._work.mkdir(parents=True, exist_ok=True)

        # Resolve ffmpeg path once — Celery workers sometimes lose PATH
        self._ffmpeg = (
            shutil.which("ffmpeg")
            or "/opt/homebrew/bin/ffmpeg"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        candidates: list[EventCandidate],
        *,
        match_config: Optional[MatchConfig] = None,
        source_file: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[VLMVerdict]:
        """Classify each candidate with the VLM. Returns a verdict per candidate."""
        src = Path(source_file) if source_file else self._source
        if src is None:
            log.error("vlm_verifier.no_source_file")
            return [self._passthrough(c) for c in candidates]

        verdicts: list[VLMVerdict] = []
        total = len(candidates)

        for i, candidate in enumerate(candidates):
            mm = int(candidate.timestamp // 60)
            ss = candidate.timestamp % 60
            log.info("vlm_verifier.classifying",
                     idx=i + 1, total=total,
                     time=f"{mm:02d}:{ss:05.2f}",
                     confidence=round(candidate.confidence, 3))

            verdict = self._classify_one(candidate, src, match_config)
            verdicts.append(verdict)

            log.info("vlm_verifier.verdict",
                     time=f"{mm:02d}:{ss:05.2f}",
                     result=verdict.result.value,
                     event_type=verdict.event_type.value if verdict.event_type else None,
                     vlm_conf=round(verdict.confidence, 3),
                     reasoning=verdict.reasoning[:100])

            if progress_callback:
                progress_callback((i + 1) / total)

        confirmed = sum(1 for v in verdicts if v.result == VerificationResult.CONFIRMED)
        log.info("vlm_verifier.complete",
                 total=total, confirmed=confirmed,
                 rejected=total - confirmed)
        return verdicts

    def is_available(self) -> bool:
        """Check if at least one VLM backend is reachable."""
        if self._vllm_url:
            try:
                import httpx
                r = httpx.get(f"{self._vllm_url}/v1/models", timeout=5)
                return r.status_code == 200
            except Exception:
                pass
        if self._anthropic_key:
            return True
        return False

    # ------------------------------------------------------------------
    # Internal — single candidate classification
    # ------------------------------------------------------------------

    def _classify_one(
        self,
        candidate: EventCandidate,
        source_file: Path,
        match_config: Optional[MatchConfig],
    ) -> VLMVerdict:
        """Two-pass classification: observe then classify."""
        frames = self._extract_clip_frames(
            source_file,
            candidate.clip_start,
            candidate.clip_end,
        )
        if not frames:
            log.warning("vlm_verifier.no_frames",
                        timestamp=candidate.timestamp)
            return self._passthrough(candidate)

        clip_duration = candidate.clip_end - candidate.clip_start
        ctx = _match_context(match_config)

        if self._TWO_PASS:
            verdict = self._two_pass_classify(
                candidate, frames, ctx, clip_duration,
            )
            if verdict is not None:
                return verdict
            log.warning("vlm_verifier.two_pass_failed, falling back",
                        timestamp=candidate.timestamp)

        # Single-pass fallback (or primary when _TWO_PASS is False)
        prompt = _CLASSIFY_SINGLE_PASS_PROMPT.format(
            match_context=ctx,
            clip_duration=clip_duration,
            num_frames=len(frames),
        )
        return self._single_pass_classify(candidate, frames, prompt)

    def _two_pass_classify(
        self,
        candidate: EventCandidate,
        frames: list[bytes],
        match_context: str,
        clip_duration: float,
    ) -> Optional[VLMVerdict]:
        """Pass 1: observe. Pass 2: classify based on observation."""
        # Pass 1: Observe
        observe_prompt = _OBSERVE_PROMPT.format(
            match_context=match_context,
            clip_duration=clip_duration,
        )

        observation = None
        model_used = "none"

        if self._vllm_url:
            observation = self._call_vllm(observe_prompt, frames)
            model_used = self._vllm_model

        if observation is None and self._anthropic_key:
            observation = self._call_claude(observe_prompt, frames)
            model_used = self._anthropic_model

        if observation is None:
            return None

        mm = int(candidate.timestamp // 60)
        ss = candidate.timestamp % 60
        log.debug("vlm_verifier.observation",
                  time=f"{mm:02d}:{ss:05.2f}",
                  observation=observation[:200])

        # Pass 2: Classify based on observation (text-only, no images)
        classify_prompt = _CLASSIFY_PROMPT.format(observation=observation)

        response = None
        if self._vllm_url:
            response = self._call_vllm_text(classify_prompt)
            model_used = self._vllm_model

        if response is None and self._anthropic_key:
            response = self._call_claude_text(classify_prompt)
            model_used = self._anthropic_model

        if response is None:
            return None

        return self._parse_response(response, candidate, model_used)

    def _single_pass_classify(
        self,
        candidate: EventCandidate,
        frames: list[bytes],
        prompt: str,
    ) -> VLMVerdict:
        """Single-pass classification with images."""
        response = None
        model_used = "none"

        if self._vllm_url:
            response = self._call_vllm(prompt, frames)
            model_used = self._vllm_model

        if response is None and self._anthropic_key:
            response = self._call_claude(prompt, frames)
            model_used = self._anthropic_model

        if response is None:
            return self._passthrough(candidate)

        return self._parse_response(response, candidate, model_used)

    # ------------------------------------------------------------------
    # Internal — frame extraction
    # ------------------------------------------------------------------

    def _extract_clip_frames(
        self,
        source_file: Path,
        start_sec: float,
        end_sec: float,
    ) -> list[bytes]:
        """Extract frames as JPEG bytes from the clip window."""
        clip_duration = end_sec - start_sec
        # Cap at MAX_FRAMES worth of content
        max_duration = self._MAX_FRAMES / self._CLIP_FPS
        if clip_duration > max_duration:
            # Centre the extraction window around the midpoint
            mid = (start_sec + end_sec) / 2
            start_sec = mid - max_duration / 2
            end_sec = mid + max_duration / 2
            clip_duration = max_duration

        out_dir = self._work / "vlm_frames"
        out_dir.mkdir(exist_ok=True)

        pattern = str(out_dir / "frame_%05d.jpg")
        cmd = [
            self._ffmpeg, "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(source_file),
            "-t", f"{clip_duration:.3f}",
            "-vf", f"fps={self._CLIP_FPS},scale={self._CLIP_WIDTH}:-2",
            "-q:v", "5",
            pattern,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                log.warning("vlm_verifier.ffmpeg_failed",
                            stderr=result.stderr[-200:].decode(errors="replace"))
                return []
        except subprocess.TimeoutExpired:
            return []

        frame_files = sorted(out_dir.glob("frame_*.jpg"))
        if len(frame_files) > self._MAX_FRAMES:
            step = len(frame_files) / self._MAX_FRAMES
            indices = [int(i * step) for i in range(self._MAX_FRAMES)]
            frame_files = [frame_files[i] for i in indices]

        frames: list[bytes] = []
        for f in frame_files:
            frames.append(f.read_bytes())
            f.unlink()

        # Clean up any remaining frames (from subsampling)
        for f in out_dir.glob("frame_*.jpg"):
            f.unlink()

        return frames

    # ------------------------------------------------------------------
    # Internal — VLM backends (image + text)
    # ------------------------------------------------------------------

    def _call_vllm(self, prompt: str, frames: list[bytes]) -> Optional[str]:
        """Call vLLM OpenAI-compatible API with images."""
        try:
            import httpx
        except ImportError:
            return None

        content: list[dict] = []

        selected = frames
        if len(frames) > self._MAX_FRAMES:
            step = len(frames) / self._MAX_FRAMES
            indices = [int(i * step) for i in range(self._MAX_FRAMES)]
            selected = [frames[i] for i in indices]

        for frame_bytes in selected:
            b64 = base64.b64encode(frame_bytes).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self._vllm_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 500,
            "temperature": 0,
        }

        try:
            r = httpx.post(
                f"{self._vllm_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            log.warning("vlm_verifier.vllm_error",
                        status=r.status_code,
                        body=r.text[:200])
            return None
        except Exception as exc:
            log.warning("vlm_verifier.vllm_exception", error=str(exc))
            return None

    def _call_vllm_text(self, prompt: str) -> Optional[str]:
        """Call vLLM with text-only prompt (Pass 2 — no images)."""
        try:
            import httpx
        except ImportError:
            return None

        payload = {
            "model": self._vllm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0,
        }

        try:
            r = httpx.post(
                f"{self._vllm_url}/v1/chat/completions",
                json=payload,
                timeout=60,
            )
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            log.warning("vlm_verifier.vllm_text_error",
                        status=r.status_code,
                        body=r.text[:200])
            return None
        except Exception as exc:
            log.warning("vlm_verifier.vllm_text_exception", error=str(exc))
            return None

    def _call_claude(self, prompt: str, frames: list[bytes]) -> Optional[str]:
        """Call Claude API with images."""
        try:
            import anthropic
        except ImportError:
            return None

        max_images = 20
        if len(frames) > max_images:
            step = len(frames) / max_images
            indices = [int(i * step) for i in range(max_images)]
            selected = [frames[i] for i in indices]
        else:
            selected = frames

        content: list[dict] = []
        for frame_bytes in selected:
            b64 = base64.b64encode(frame_bytes).decode()
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        try:
            client = anthropic.Anthropic(api_key=self._anthropic_key)
            msg = client.messages.create(
                model=self._anthropic_model,
                max_tokens=500,
                messages=[{"role": "user", "content": content}],
            )
            return msg.content[0].text
        except Exception as exc:
            log.warning("vlm_verifier.claude_exception", error=str(exc))
            return None

    def _call_claude_text(self, prompt: str) -> Optional[str]:
        """Call Claude API with text-only prompt (Pass 2)."""
        try:
            import anthropic
        except ImportError:
            return None

        try:
            client = anthropic.Anthropic(api_key=self._anthropic_key)
            msg = client.messages.create(
                model=self._anthropic_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as exc:
            log.warning("vlm_verifier.claude_text_exception", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Internal — response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: str,
        candidate: EventCandidate,
        model_used: str,
    ) -> VLMVerdict:
        """Parse VLM JSON response into a VLMVerdict."""
        try:
            text = response.strip()
            # Strip markdown code fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()

            # Handle Qwen3 <think>...</think> blocks
            if "<think>" in text:
                import re
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            # Try to find JSON in the response
            if not text.startswith("{"):
                # Look for JSON object in the text
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]

            data = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            log.warning("vlm_verifier.parse_failed",
                        response=response[:300],
                        timestamp=candidate.timestamp)
            return self._passthrough(candidate)

        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning", "")

        # Handle event classification response
        if "event" in data:
            event_str = data["event"]
            if event_str in ("none", "neither"):
                return VLMVerdict(
                    candidate=candidate,
                    result=VerificationResult.REJECTED,
                    event_type=None,
                    confidence=confidence,
                    reasoning=reasoning,
                    model_used=model_used,
                )
            event_type = _EVENT_TYPE_MAP.get(event_str)
            if event_type is None:
                log.warning("vlm_verifier.unknown_event",
                            event=event_str, timestamp=candidate.timestamp)
                return self._passthrough(candidate)

            result = (
                VerificationResult.CONFIRMED
                if confidence >= self._min_conf
                else VerificationResult.UNCERTAIN
            )
            return VLMVerdict(
                candidate=candidate,
                result=result,
                event_type=event_type,
                confidence=confidence,
                reasoning=reasoning,
                model_used=model_used,
            )

        # Handle verify-event response (legacy)
        if "is_correct" in data:
            is_correct = data["is_correct"]
            actual = data.get("actual_event", "")
            event_type = _EVENT_TYPE_MAP.get(actual)
            result = (
                VerificationResult.CONFIRMED
                if is_correct and confidence >= self._min_conf
                else VerificationResult.REJECTED
            )
            return VLMVerdict(
                candidate=candidate,
                result=result,
                event_type=event_type,
                confidence=confidence,
                reasoning=reasoning,
                model_used=model_used,
            )

        return self._passthrough(candidate)

    def verify_kickoff(
        self,
        candidates: list[EventCandidate],
        *,
        match_config: Optional[MatchConfig] = None,
        source_file: Optional[str | Path] = None,
    ) -> list[VLMVerdict]:
        """Focused kickoff check — single-pass, binary question.

        Used by goal inference to detect center-circle restarts after shots.
        Much cheaper than full two-pass classification.
        """
        src = Path(source_file) if source_file else self._source
        if src is None:
            return [self._passthrough(c) for c in candidates]

        ctx = _match_context(match_config)
        verdicts: list[VLMVerdict] = []

        for candidate in candidates:
            mm = int(candidate.timestamp // 60)
            ss = candidate.timestamp % 60
            log.info("vlm_verifier.kickoff_check",
                     time=f"{mm:02d}:{ss:05.2f}")

            frames = self._extract_clip_frames(
                src, candidate.clip_start, candidate.clip_end,
            )
            if not frames:
                verdicts.append(self._passthrough(candidate))
                continue

            clip_duration = candidate.clip_end - candidate.clip_start
            prompt = _KICKOFF_CHECK_PROMPT.format(
                match_context=ctx,
                clip_duration=clip_duration,
            )

            response = None
            model_used = "none"
            if self._vllm_url:
                response = self._call_vllm(prompt, frames)
                model_used = self._vllm_model
            if response is None and self._anthropic_key:
                response = self._call_claude(prompt, frames)
                model_used = self._anthropic_model

            if response is None:
                verdicts.append(self._passthrough(candidate))
                continue

            verdict = self._parse_kickoff_response(
                response, candidate, model_used,
            )
            verdicts.append(verdict)

            log.info("vlm_verifier.kickoff_result",
                     time=f"{mm:02d}:{ss:05.2f}",
                     is_kickoff=verdict.event_type == EventType.KICKOFF,
                     reasoning=verdict.reasoning[:100])

        return verdicts

    def verify_set_piece(
        self,
        candidates: list[EventCandidate],
        *,
        match_config: Optional[MatchConfig] = None,
        source_file: Optional[str | Path] = None,
    ) -> list[VLMVerdict]:
        """Focused set-piece check after shots — corner, goal kick, or throw-in.

        Used by set-piece inference to detect restarts after shots/saves.
        """
        src = Path(source_file) if source_file else self._source
        if src is None:
            return [self._passthrough(c) for c in candidates]

        ctx = _match_context(match_config)
        verdicts: list[VLMVerdict] = []

        for candidate in candidates:
            mm = int(candidate.timestamp // 60)
            ss = candidate.timestamp % 60
            log.info("vlm_verifier.set_piece_check",
                     time=f"{mm:02d}:{ss:05.2f}")

            frames = self._extract_clip_frames(
                src, candidate.clip_start, candidate.clip_end,
            )
            if not frames:
                verdicts.append(self._passthrough(candidate))
                continue

            clip_duration = candidate.clip_end - candidate.clip_start
            prompt = _SET_PIECE_CHECK_PROMPT.format(
                match_context=ctx,
                clip_duration=clip_duration,
            )

            response = None
            model_used = "none"
            if self._vllm_url:
                response = self._call_vllm(prompt, frames)
                model_used = self._vllm_model
            if response is None and self._anthropic_key:
                response = self._call_claude(prompt, frames)
                model_used = self._anthropic_model

            if response is None:
                verdicts.append(self._passthrough(candidate))
                continue

            verdict = self._parse_set_piece_response(
                response, candidate, model_used,
            )
            verdicts.append(verdict)

            log.info("vlm_verifier.set_piece_result",
                     time=f"{mm:02d}:{ss:05.2f}",
                     restart=verdict.event_type.value if verdict.event_type else "none",
                     reasoning=verdict.reasoning[:100])

        return verdicts

    _SET_PIECE_TYPE_MAP = {
        "corner_kick": EventType.CORNER_KICK,
        "goal_kick": EventType.GOAL_KICK,
        "throw_in": EventType.THROW_IN,
    }

    def _parse_set_piece_response(
        self,
        response: str,
        candidate: EventCandidate,
        model_used: str,
    ) -> VLMVerdict:
        """Parse set-piece check response."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            if "<think>" in text:
                import re
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
            data = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return self._passthrough(candidate)

        restart = data.get("restart", "none")
        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning", "")

        event_type = self._SET_PIECE_TYPE_MAP.get(restart)
        if event_type and confidence >= self._min_conf:
            return VLMVerdict(
                candidate=candidate,
                result=VerificationResult.CONFIRMED,
                event_type=event_type,
                confidence=confidence,
                reasoning=reasoning,
                model_used=model_used,
            )
        return VLMVerdict(
            candidate=candidate,
            result=VerificationResult.REJECTED,
            event_type=None,
            confidence=confidence,
            reasoning=reasoning,
            model_used=model_used,
        )

    def verify_corner(
        self,
        candidates: list[EventCandidate],
        *,
        match_config: Optional[MatchConfig] = None,
        source_file: Optional[str | Path] = None,
    ) -> list[VLMVerdict]:
        """Focused corner kick check — binary question.

        Used by the independent corner scan to detect corner kicks
        at motion candidates that the main VLM pass missed.
        """
        src = Path(source_file) if source_file else self._source
        if src is None:
            return [self._passthrough(c) for c in candidates]

        ctx = _match_context(match_config)
        verdicts: list[VLMVerdict] = []

        for candidate in candidates:
            mm = int(candidate.timestamp // 60)
            ss = candidate.timestamp % 60
            log.info("vlm_verifier.corner_check",
                     time=f"{mm:02d}:{ss:05.2f}")

            frames = self._extract_clip_frames(
                src, candidate.clip_start, candidate.clip_end,
            )
            if not frames:
                verdicts.append(self._passthrough(candidate))
                continue

            clip_duration = candidate.clip_end - candidate.clip_start
            prompt = _CORNER_CHECK_PROMPT.format(
                match_context=ctx,
                clip_duration=clip_duration,
            )

            response = None
            model_used = "none"
            if self._vllm_url:
                response = self._call_vllm(prompt, frames)
                model_used = self._vllm_model
            if response is None and self._anthropic_key:
                response = self._call_claude(prompt, frames)
                model_used = self._anthropic_model

            if response is None:
                verdicts.append(self._passthrough(candidate))
                continue

            verdict = self._parse_corner_response(
                response, candidate, model_used,
            )
            verdicts.append(verdict)

            log.info("vlm_verifier.corner_result",
                     time=f"{mm:02d}:{ss:05.2f}",
                     is_corner=verdict.event_type == EventType.CORNER_KICK,
                     reasoning=verdict.reasoning[:100])

        return verdicts

    def _parse_corner_response(
        self,
        response: str,
        candidate: EventCandidate,
        model_used: str,
    ) -> VLMVerdict:
        """Parse corner check response."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            if "<think>" in text:
                import re
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
            data = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return self._passthrough(candidate)

        is_corner = data.get("is_corner", False)
        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning", "")

        if is_corner and confidence >= self._min_conf:
            return VLMVerdict(
                candidate=candidate,
                result=VerificationResult.CONFIRMED,
                event_type=EventType.CORNER_KICK,
                confidence=confidence,
                reasoning=reasoning,
                model_used=model_used,
            )
        return VLMVerdict(
            candidate=candidate,
            result=VerificationResult.REJECTED,
            event_type=None,
            confidence=confidence,
            reasoning=reasoning,
            model_used=model_used,
        )

    # ------------------------------------------------------------------
    # Shot scan — binary "was a shot taken?" check
    # ------------------------------------------------------------------

    def verify_shot(
        self,
        candidates: list[EventCandidate],
        *,
        match_config: Optional[MatchConfig] = None,
        source_file: Optional[str | Path] = None,
    ) -> list[VLMVerdict]:
        """Focused shot check — binary question with shot_type sub-class.

        Used by the shot scan phase to recover shots from rejected
        candidates that the multi-class prompt missed.
        """
        src = Path(source_file) if source_file else self._source
        if src is None:
            return [self._passthrough(c) for c in candidates]

        ctx = _match_context(match_config)
        verdicts: list[VLMVerdict] = []

        for candidate in candidates:
            mm = int(candidate.timestamp // 60)
            ss = candidate.timestamp % 60
            log.info("vlm_verifier.shot_check",
                     time=f"{mm:02d}:{ss:05.2f}")

            # Use a tighter clip window: 3s pre / 5s post to focus
            # on the shot moment, not the aftermath
            tight_start = max(0, candidate.timestamp - 3.0)
            tight_end = candidate.timestamp + 5.0
            frames = self._extract_clip_frames(
                src, tight_start, tight_end,
            )
            if not frames:
                verdicts.append(self._passthrough(candidate))
                continue

            clip_duration = tight_end - tight_start
            prompt = _SHOT_CHECK_PROMPT.format(
                match_context=ctx,
                clip_duration=clip_duration,
                num_frames=len(frames),
            )

            response = None
            model_used = "none"
            if self._vllm_url:
                response = self._call_vllm(prompt, frames)
                model_used = self._vllm_model
            if response is None and self._anthropic_key:
                response = self._call_claude(prompt, frames)
                model_used = self._anthropic_model

            if response is None:
                verdicts.append(self._passthrough(candidate))
                continue

            verdict = self._parse_shot_response(
                response, candidate, model_used,
            )
            verdicts.append(verdict)

            log.info("vlm_verifier.shot_result",
                     time=f"{mm:02d}:{ss:05.2f}",
                     is_shot=verdict.result == VerificationResult.CONFIRMED,
                     event_type=(verdict.event_type.value
                                 if verdict.event_type else None),
                     reasoning=verdict.reasoning[:100])

        return verdicts

    _SHOT_TYPE_MAP = {
        "on_target": EventType.SHOT_ON_TARGET,
        "off_target": EventType.SHOT_OFF_TARGET,
        "blocked": EventType.SHOT_OFF_TARGET,  # Blocked = off target
    }

    def _parse_shot_response(
        self,
        response: str,
        candidate: EventCandidate,
        model_used: str,
    ) -> VLMVerdict:
        """Parse shot check response."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            if "<think>" in text:
                import re
                text = re.sub(
                    r"<think>.*?</think>", "", text, flags=re.DOTALL,
                ).strip()
            if not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
            data = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return self._passthrough(candidate)

        is_shot = data.get("is_shot", False)
        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning", "")
        shot_type = data.get("shot_type", "on_target")

        if is_shot and confidence >= self._min_conf:
            event_type = self._SHOT_TYPE_MAP.get(
                shot_type, EventType.SHOT_ON_TARGET,
            )
            return VLMVerdict(
                candidate=candidate,
                result=VerificationResult.CONFIRMED,
                event_type=event_type,
                confidence=confidence,
                reasoning=f"SHOT_SCAN: {reasoning}",
                model_used=model_used,
            )
        return VLMVerdict(
            candidate=candidate,
            result=VerificationResult.REJECTED,
            event_type=None,
            confidence=confidence,
            reasoning=reasoning,
            model_used=model_used,
        )

    # ------------------------------------------------------------------
    # Catch check (Phase 3g)
    # ------------------------------------------------------------------

    def verify_catch(
        self,
        candidates: list[EventCandidate],
        *,
        match_config: Optional[MatchConfig] = None,
        source_file: Optional[str | Path] = None,
    ) -> list[VLMVerdict]:
        """Focused catch check — probe aftermath of a shot for GK holding ball.

        Uses a shifted clip window: shot+3s to shot+8s to focus on the
        result state (GK holding ball) rather than the shot itself.
        """
        src = Path(source_file) if source_file else self._source
        if src is None:
            return [self._passthrough(c) for c in candidates]

        ctx = _match_context(match_config)
        verdicts: list[VLMVerdict] = []

        for candidate in candidates:
            mm = int(candidate.timestamp // 60)
            ss = candidate.timestamp % 60
            log.info("vlm_verifier.catch_check",
                     time=f"{mm:02d}:{ss:05.2f}")

            # Shifted window: 3-8s AFTER the shot to see the result state
            catch_start = candidate.timestamp + 3.0
            catch_end = candidate.timestamp + 8.0
            frames = self._extract_clip_frames(
                src, catch_start, catch_end,
            )
            if not frames:
                verdicts.append(self._passthrough(candidate))
                continue

            clip_duration = catch_end - catch_start
            prompt = _CATCH_CHECK_PROMPT.format(
                match_context=ctx,
                clip_duration=clip_duration,
                num_frames=len(frames),
            )

            response = None
            model_used = "none"
            if self._vllm_url:
                response = self._call_vllm(prompt, frames)
                model_used = self._vllm_model
            if response is None and self._anthropic_key:
                response = self._call_claude(prompt, frames)
                model_used = self._anthropic_model

            if response is None:
                verdicts.append(self._passthrough(candidate))
                continue

            verdict = self._parse_catch_response(
                response, candidate, model_used,
            )
            verdicts.append(verdict)

            log.info("vlm_verifier.catch_result",
                     time=f"{mm:02d}:{ss:05.2f}",
                     is_catch=verdict.result == VerificationResult.CONFIRMED,
                     reasoning=verdict.reasoning[:100])

        return verdicts

    def _parse_catch_response(
        self,
        response: str,
        candidate: EventCandidate,
        model_used: str,
    ) -> VLMVerdict:
        """Parse catch check response."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            if "<think>" in text:
                import re
                text = re.sub(
                    r"<think>.*?</think>", "", text, flags=re.DOTALL,
                ).strip()
            if not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
            data = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return self._passthrough(candidate)

        is_catch = data.get("is_catch", False)
        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning", "")

        if is_catch and confidence >= self._min_conf:
            return VLMVerdict(
                candidate=candidate,
                result=VerificationResult.CONFIRMED,
                event_type=EventType.CATCH,
                confidence=confidence,
                reasoning=f"CATCH_SCAN: {reasoning}",
                model_used=model_used,
            )
        return VLMVerdict(
            candidate=candidate,
            result=VerificationResult.REJECTED,
            event_type=None,
            confidence=confidence,
            reasoning=reasoning,
            model_used=model_used,
        )

    # ------------------------------------------------------------------
    # Kickoff response parsing
    # ------------------------------------------------------------------

    def _parse_kickoff_response(
        self,
        response: str,
        candidate: EventCandidate,
        model_used: str,
    ) -> VLMVerdict:
        """Parse kickoff check response."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            if "<think>" in text:
                import re
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if not text.startswith("{"):
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]
            data = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return self._passthrough(candidate)

        is_kickoff = data.get("is_kickoff", False)
        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning", "")

        if is_kickoff and confidence >= self._min_conf:
            return VLMVerdict(
                candidate=candidate,
                result=VerificationResult.CONFIRMED,
                event_type=EventType.KICKOFF,
                confidence=confidence,
                reasoning=reasoning,
                model_used=model_used,
            )
        return VLMVerdict(
            candidate=candidate,
            result=VerificationResult.REJECTED,
            event_type=None,
            confidence=confidence,
            reasoning=reasoning,
            model_used=model_used,
        )

    def _passthrough(self, candidate: EventCandidate) -> VLMVerdict:
        """Pass candidate through unverified (fail-open)."""
        return VLMVerdict(
            candidate=candidate,
            result=VerificationResult.UNCERTAIN,
            event_type=None,
            confidence=candidate.confidence,
            reasoning="VLM unavailable — passed through unverified",
            model_used="none",
        )
