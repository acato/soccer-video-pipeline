"""
VLM event detector with two modes:

SINGLE-PASS MODE (single_pass=True, recommended with 32B + NVLink):
  Slides a window across the entire game, sends frames directly to the 32B
  with the full classification prompt.  No triage, no observe, no text
  intermediary.  The 32B sees images and classifies events in one step.

DUAL-PASS MODE (single_pass=False, legacy for 8B+32B with model swap):
  Phase 1: 8B triage scan (coarse labels)
  Phase 2: 8B observe (images → text descriptions)
  Phase 3: Model swap to 32B
  Phase 4: 32B classify (text-only, no images)
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import structlog

from src.detection.frame_sampler import FrameSampler, SampledFrame
from src.detection.models import Event, EventType
from src.detection.triage_scanner import (
    ACTIVE_LABELS,
    CandidateWindow,
    TriageScanResult,
    TriageScanner,
    merge_flags,
)

log = structlog.get_logger(__name__)


class CanaryFailure(RuntimeError):
    """Raised when a canary check detects a likely pipeline regression."""
    pass

# ── 8B Observe prompt ──────────────────────────────────────────────────
# Sent to 8B WITH images.  The 8B handles many frames + long prompts fine.
# Asks for free-text description focused on the signals that matter for
# event classification downstream.
# Run #16 lesson: structured yes/no checklist made 8B ultra-conservative —
# it answered "no" to ALL restart questions for every GT corner, free kick,
# and throw-in.  Free-form narrative (Run #15) produced richer descriptions.
#
# Run #16 fix: revert to free-form but add explicit DEAD BALL detection.
# Merge corner_kick + free_kick_shot → "set_piece" since the 8B can't
# distinguish them visually (corner flag too small at 960px).
_OBSERVE_PROMPT = """\
You are analyzing frames from a soccer match recorded by a sideline camera.
These {n_frames} frames span a {duration:.0f}-second window ({start:.0f}s – {end:.0f}s).
Triage flagged this window as: {triage_labels}.

Describe EXACTLY what you see. Cover ALL of these topics:

1. DEAD BALL / RESTART — This is the MOST IMPORTANT question. \
Is play STOPPED at any point? Look for these signs:
   - The ball is STATIONARY on the ground while players stand around it
   - Players are LINED UP or GATHERED in a formation, waiting
   - A player is STANDING OVER the ball, about to kick it
   - A player at the SIDELINE holds the ball OVERHEAD (= throw-in)
   - The ball is on the ground near a CORNER of the field
   - The ball is on the ground in the GOAL AREA with the GK about to kick
   - Players form a WALL (3+ standing shoulder to shoulder)
   If you see ANY of these, describe the LOCATION (which part of the \
   field — corner, sideline, goal area, center, or elsewhere) and \
   what the players are doing. Even if you are unsure what type of \
   restart it is, describe the scene.

2. GOALKEEPER — Is the GK:
   - HOLDING the ball in their hands? (= catch)
   - DIVING and the ball REBOUNDING away? (= save/parry)
   - PUNCHING the ball with a fist?

3. GOAL SIGNALS — after a shot, look carefully for these:
   - Do players CELEBRATE — arms raised, group hugs, running to teammates?
   - Do teams WALK BACK / RETURN toward the CENTER CIRCLE? \
     (Even subtle movement — players jogging away from the goal toward \
     midfield is a strong signal)
   - Do players from BOTH teams gather near the CENTER SPOT? \
     (= kickoff setup, always follows a goal)
   - Is the ball IN THE NET (net visibly disturbed)?
   The camera is far away — celebrations may look like small figures \
   with arms up. Look for the PATTERN: action near goal, then general \
   movement toward the center of the field.

4. SHOTS — Does a player STRIKE the ball toward goal?

5. SEQUENCE — What happens AFTER the main action? This is CRITICAL. \
   Describe what players do in the 20-30 seconds after a shot: \
   do they set up for a goal kick? Walk to center? Continue playing? \
   (A shot followed by a restart is TWO events. Describe BOTH.)

Include timestamps (e.g., "at t=45s play stops, ball is stationary \
near the corner, a player stands over it ready to kick at t=48s").
Be specific about what you SEE, not what you infer."""

# ── 32B Classify prompt — TEXT-ONLY, no images ─────────────────────────
# Sent to 32B with ONLY the 8B observation text.  No image constraints means
# we can use the full detailed prompt with all disambiguation rules.
# CRITICAL: Do NOT mention "[]" or "empty list" anywhere — 32B latches on it.
#
# Run #16 lesson: merged corner_kick + free_kick_shot → set_piece.
# The 8B can't distinguish them visually.  We detect the dead-ball pattern
# (stationary ball, players gathered/lined up) and classify as set_piece.
# Also: free-form observe is back — structured yes/no made 8B too conservative.
_CLASSIFY_PROMPT = """\
You are a soccer event classifier. An observation model analyzed video frames \
from a {duration:.0f}s window ({start:.0f}s – {end:.0f}s) and produced this description:

---
{observation}
---

Classify events using this DECISION TREE. Work through each step IN ORDER. \
Return ALL events that apply (a window can contain a shot AND a restart).

STEP 1 — DEAD BALL / SET PIECE (check first — these are distinctive):
- throw_in: Player at SIDELINE/TOUCHLINE, ball held OVERHEAD with both hands. \
  Very common (~1 per 2 minutes of play).
- goal_kick: Ball STATIONARY in GOAL AREA / near the GOAL. GK or defender \
  about to kick upfield. Key: the GOALKEEPER is the one kicking, and \
  opponents are far away / retreating. Check BEFORE set_piece.
- set_piece: Ball STATIONARY on the ground with players gathered around it, \
  OR a player standing over the ball ready to kick, OR a defensive wall \
  of 3+ players. This covers corner kicks AND free kicks — do not try to \
  distinguish between them. Location can be anywhere: corner of field, \
  edge of penalty area, midfield, etc. Key signal: PLAY IS STOPPED, \
  ball on the ground, players waiting. \
  NOTE: if the GK is the one kicking from the goal area → goal_kick, not set_piece.
- kickoff: Ball at CENTER SPOT, both teams in own halves.

If the description mentions play stopping, ball stationary, or players \
lining up — it is likely a set piece. INCLUDE it.

STEP 2 — GOALKEEPER ACTIONS:
- catch: GK HOLDS/SECURES ball in hands, then distributes. Ball IN hands. \
  If a shot preceded it, classify ONLY as catch (not shot + catch).
- shot_stop_diving: GK DIVES, ball REBOUNDS/DEFLECTS away. Ball NOT held. \
  Often followed by a set piece — include both if described.
- punch: GK PUNCHES ball with FIST. Ball goes up/outward, not caught.

STEP 3 — GOAL (look for post-goal signals):
- goal: A shot toward goal FOLLOWED by any of these post-goal signals:
  (a) Players CELEBRATING — arms raised, group hugs, sliding, fist pumps
  (b) Teams WALKING BACK / RETURNING toward the center circle
  (c) Players from BOTH TEAMS gathering near the CENTER SPOT / center circle \
      (this is the KICKOFF setup that always follows a goal)
  (d) Ball CLEARLY IN NET with net disturbed
  (e) A KICKOFF restart — ball at center spot, teams in own halves
  ANY of these signals after a shot = GOAL. The camera is far away, so \
  celebrations may look subtle (small figures with arms up, players jogging \
  toward center). Look for the overall pattern: action near goal THEN \
  movement toward center of the field.

STEP 4 — SHOT (fallback only):
- shot_on_target: A player strikes ball toward goal. Use ONLY if Steps 1-3 \
  did not produce a more specific classification. If a restart follows \
  (goal kick, set piece, throw-in), classify the restart TOO.

For each event: start_sec = moment of action, end_sec = start + 3-5s.

Reply as a JSON array. Each element needs: event_type, start_sec, \
end_sec, confidence, reasoning. Example element: \
{{"event_type": "set_piece", "start_sec": 30.0, "end_sec": 35.0, \
"confidence": 0.85, "reasoning": "play stopped, ball stationary near corner"}}

If the observation describes any notable action, you MUST return at \
least one event. Do NOT return empty when the description contains events.
"""


# ── Single-pass prompt — frames + classification in one shot ──────────
# Sent to 32B WITH images.  Combines the observe and classify steps into
# a single VLM call.  Requires a model that handles images well (32B FP8
# with TP=2 NVLink).
_DIRECT_CLASSIFY_PROMPT = """\
You are analyzing {n_frames} frames from a soccer match ({start:.0f}s – {end:.0f}s).

For each DISTINCT event you see, classify it. Work through this decision tree:

STEP 1 — DEAD BALL / RESTART (check first — these are the most common events):
- throw_in: Player standing near the SIDELINE/TOUCHLINE with ball held \
  OVERHEAD or at chest height preparing to throw. Both hands on ball. \
  This is the MOST COMMON event (~1 every 2 minutes). Look for ANY player \
  near the sideline holding or about to hold the ball — even the setup \
  (walking to the line, picking up the ball) counts. Flag aggressively.
- goal_kick: Ball STATIONARY in the 6-YARD BOX or GOAL AREA, GK or \
  defender about to kick upfield, opponents retreated. The kicker is \
  INSIDE or very near the small box closest to the goal.
- corner_kick: Ball STATIONARY at or near a CORNER FLAG / corner arc. \
  A player is standing at the corner of the field ready to kick. Look \
  for the distinctive corner flag and the curved corner arc marking.
- free_kick_shot: Ball STATIONARY anywhere on the pitch OUTSIDE the \
  goal area and away from corners, with a player standing over it ready \
  to kick. Often a defensive WALL of 3+ players forms nearby. Can be \
  anywhere: midfield, edge of penalty area, defensive third.
- kickoff: Ball at CENTER SPOT, both teams in own halves.

STEP 2 — GOALKEEPER ACTIONS (look for the GK specifically):
- catch: GK HOLDS or SECURES ball in their HANDS. Ball is IN the GK's \
  hands/arms, not bouncing away. GK then typically distributes (throws \
  or kicks) the ball. This is VERY COMMON (~12 per game). If you see \
  the GK standing with the ball in their hands, that is a catch.
- shot_stop_diving: GK DIVES or LUNGES and the ball REBOUNDS/deflects \
  AWAY. The key difference from catch: the ball is NOT held — it bounces \
  off the GK or goes wide. The GK's body hits the ground.
- punch: GK PUNCHES ball with a closed FIST, usually on a cross/corner.

STEP 3 — GOAL (requires post-goal signals):
- goal: Shot toward goal FOLLOWED BY celebration (arms raised, group \
  hugs, running), OR teams walking back toward center circle, OR \
  kickoff setup. Requires POSITIVE evidence — "ball near goal" alone \
  is NOT a goal.

STEP 4 — SHOT (first-class event — emit alongside related events):
- shot_on_target: A player STRIKES the ball toward the goal. This is \
  NOT a fallback — it is a primary event class. Emit shot_on_target \
  WHENEVER a shot is struck, INCLUDING when: \
  (a) the shot is SAVED by the GK (emit BOTH shot_on_target AND \
      catch/shot_stop_diving — they are separate events); \
  (b) the shot results in a GOAL (emit BOTH shot_on_target AND goal); \
  (c) a RESTART follows the shot — goal kick, throw-in, corner (emit \
      BOTH shot_on_target AND the restart); \
  (d) the shot goes wide, hits the post, or is blocked by an outfield \
      player. \
  The ground truth treats every shot as a separate event from its \
  outcome, so DO NOT collapse "shot + save" into a single catch/save \
  event — emit both. Ball MOVING toward the goal after being struck by \
  a player = shot_on_target, regardless of what happens after.

A window can contain MULTIPLE events (e.g., shot_on_target + goal_kick).
Throw-ins and goal kicks are the MOST COMMON events — flag aggressively.
Catches are common too — whenever the GK has the ball in their hands.

For each event: start_sec and end_sec should be the actual timestamps.

Reply as a JSON array. Each element: {{"event_type": "...", "start_sec": N, \
"end_sec": N, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

BEFORE you conclude "open play" and return none, CHECK EACH OF THESE — \
these events are routinely missed when the pose is only briefly visible:

  (a) THROW-IN check — scan the SIDELINES (top and bottom edges of the \
      frame) in EVERY frame: is any player near the touchline holding a \
      ball, reaching down to pick one up, walking toward the line with a \
      ball, or mid-throw? The throw-in pose (ball overhead, both hands) \
      is brief — you may only see the pre- or post-throw posture. Any of \
      these = throw_in. Do NOT require a clear "ball overhead" pose.

  (b) CATCH check — is the goalkeeper visible with a ball in their \
      hands/arms, even briefly? Even if they are walking, bouncing the \
      ball, or preparing to distribute — that is catch. If a shot or \
      save preceded and the GK now has the ball = catch.

  (c) CORNER check — is the ball anywhere near a CORNER FLAG, or is a \
      player standing at the corner arc? Corner kicks are often shot \
      from wide, so look at the four corners of the pitch specifically.

Only return "none" if ALL THREE checks are clearly negative AND the \
frames show continuous open play (ball in motion mid-field, no stoppage). \
If in doubt on any check, emit the event with confidence 0.5-0.7 rather \
than skipping.

If you see only normal open play with no notable event, return: \
{{"event_type": "none", "start_sec": {start}, "end_sec": {end}, \
"confidence": 0.9, "reasoning": "open play (throw-in/catch/corner checks negative)"}}
"""


import re

# ── Goal evidence checker ──────────────────────────────────────────────
# Run #13 showed 79/93 "goal" detections match "celebrat" because the 32B
# writes "no celebration is seen, but ball is in the net" — the word
# appears in NEGATION.  This helper requires POSITIVE evidence.

# Patterns that indicate real celebration (not negated)
_POSITIVE_CELEB_RE = re.compile(
    r"(?:players?|team(?:mates)?|(?:he|she|they))\s+"
    r"(?:are |is |were |was |begin |start )?"
    r"(?:celebrat|hugging|embracing|sliding)",
    re.IGNORECASE,
)
_ARMS_RAISED_RE = re.compile(
    r"(?:with |raising |raises? |raised )"
    r"(?:his |her |their )?"
    r"arms?\s*(?:raised|up|in the air|aloft)",
    re.IGNORECASE,
)
_KICKOFF_RESTART_RE = re.compile(
    r"(?:walking|walk|return|returning|headed|heading|moving)\s+"
    r"(?:back\s+)?(?:to|toward)\s+(?:the\s+)?center\s+circle",
    re.IGNORECASE,
)
_CENTER_CIRCLE_KICKOFF_RE = re.compile(
    r"center\s+circle.*(?:kickoff|kick-off|kick off|restart)",
    re.IGNORECASE,
)

# ── Real-kickoff evidence ──────────────────────────────────────────────
# Distinguish true post-goal / post-halftime kickoffs from the VLM's
# tendency to tag any center-field activity as "kickoff". A real kickoff
# has visually distinctive markers: ball exactly on center spot, both
# teams in own halves, players in formation lines.
_CENTER_SPOT_RE = re.compile(r"center\s+spot", re.IGNORECASE)
_BOTH_TEAMS_OWN_HALVES_RE = re.compile(
    r"(?:both\s+teams|each\s+team|teams\s+are)\s+.{0,30}?"
    r"(?:own\s+hal(?:f|ves)|their\s+hal(?:f|ves)|respective\s+hal)",
    re.IGNORECASE,
)
_KICKOFF_SETUP_RE = re.compile(
    r"kickoff\s+(?:setup|formation|restart|position)"
    r"|(?:setup|formation|position)\s+for\s+(?:the\s+)?kickoff",
    re.IGNORECASE,
)


def _has_real_kickoff_evidence(reasoning: str) -> bool:
    """Distinguish real kickoffs from false VLM 'kickoff' tags.

    Real kickoff markers (any one suffices):
    - Ball at center spot
    - Both teams in own halves
    - Explicit kickoff setup/formation/restart language
    - Walking/returning toward center circle (post-goal ceremony)

    Reject if reasoning is vague ("kickoff", "center circle" alone) —
    those fire on many non-kickoff scenes (e.g. free kicks at midfield).
    """
    if _CENTER_SPOT_RE.search(reasoning):
        return True
    if _BOTH_TEAMS_OWN_HALVES_RE.search(reasoning):
        return True
    if _KICKOFF_SETUP_RE.search(reasoning):
        return True
    if _KICKOFF_RESTART_RE.search(reasoning):
        return True
    if _CENTER_CIRCLE_KICKOFF_RE.search(reasoning):
        return True
    return False


def _has_positive_goal_evidence(reasoning: str) -> bool:
    """Check if the 32B reasoning contains POSITIVE goal evidence.

    Returns True only if celebration, kickoff restart, or similar
    post-goal signals are mentioned affirmatively (not negated).
    "Ball in the net" alone is NOT sufficient — the 8B/32B describe
    many saves as ball near/in goal.
    """
    if _POSITIVE_CELEB_RE.search(reasoning):
        return True
    if _ARMS_RAISED_RE.search(reasoning):
        return True
    if _KICKOFF_RESTART_RE.search(reasoning):
        return True
    if _CENTER_CIRCLE_KICKOFF_RE.search(reasoning):
        return True
    # Explicit positive phrases
    for phrase in ["fist pump", "group hug", "sliding on knees",
                   "running to teammates", "jumping in celebration"]:
        if phrase in reasoning:
            return True
    return False


# ── Save-language goal gate ────────────────────────────────────────────
# Run #22: 32B single-pass labels saves as "goal" because it sees a shot
# toward goal.  The reasoning text itself describes the save — goalkeeper
# diving, ball rebounding away, "does not enter the net" — contradicting
# the goal label.  Demote these to shot_on_target.

_SAVE_ACTION_RE = re.compile(
    r"(?:goalkeeper|gk|keeper)\s+"
    r"(?:dives?|saves?|parr(?:ies|y|ied)|stops?|deflects?"
    r"|catches|punches|tips|blocks?)",
    re.IGNORECASE,
)
_BALL_REBOUND_RE = re.compile(
    r"(?:ball|it)\s+"
    r"(?:deflects?|rebounds?|goes?\s+wide|bounces?)\s+away",
    re.IGNORECASE,
)
_NO_GOAL_RE = re.compile(
    r"does not enter the net|"
    r"ball is deflected away from the goal|"
    r"saved|cleared off the line",
    re.IGNORECASE,
)


def _has_save_language(reasoning: str) -> bool:
    """Return True if the VLM reasoning describes a save, not a goal.

    When the reasoning says the keeper dived/saved/deflected or the ball
    rebounded away, the event is a shot that was stopped — not a goal.
    """
    return bool(
        _SAVE_ACTION_RE.search(reasoning)
        or _BALL_REBOUND_RE.search(reasoning)
        or _NO_GOAL_RE.search(reasoning)
    )


@dataclass
class DualPassConfig:
    """Configuration for the dual-pass detector."""
    # vLLM server
    vllm_url: str = "http://10.10.2.222:8000"

    # Single-pass mode: 32B classifies directly from frames (no triage/observe split)
    single_pass: bool = False

    # Single-pass settings
    single_pass_step_sec: float = 10.0   # Slide step (10s = ~684 windows for 114 min game)
    single_pass_window_sec: float = 15.0  # Each window spans 15s (5s overlap)
    single_pass_frames: int = 5           # Frames per window (one every 3s)
    single_pass_timeout_sec: int = 60     # Per-window timeout

    # 8B triage model
    tier1_model_name: str = "qwen3-vl-8b"
    tier1_model_path: str = "Qwen/Qwen3-VL-8B-Instruct"

    # 32B classification model
    tier2_model_name: str = "qwen3-vl-32b-fp8"
    tier2_model_path: str = "Qwen/Qwen3-VL-32B-Instruct-FP8"

    # Triage scanner settings
    frame_width: int = 1280  # Run #16: up from 960 — need to see celebrations/kickoffs
    frames_per_window: int = 7
    window_span_sec: float = 10.0
    step_sec: float = 4.0  # Tighter step = more overlap = fewer boundary misses

    # Candidate merging (tight — ball_zone + gap splitting)
    merge_gap_sec: float = 4.0
    merge_pad_sec: float = 30.0  # Run #16: up from 6 — must capture post-goal kickoff
    max_window_sec: float = 90.0  # Run #16: up from 60 — longer windows for goal context

    # 8B observe — frames → text description (while 8B still loaded)
    observe_max_frames: int = 15  # Run #16: up from 10 — more temporal coverage
    observe_timeout_sec: int = 120  # More frames = slightly longer

    # 32B classification — text-only (no images sent to 32B)
    classify_max_frames: int = 2  # Legacy — only used if observe is skipped
    sub_window_sec: float = 20.0  # Sub-window size for chunking
    sub_window_overlap_sec: float = 5.0  # Overlap between sub-windows
    max_candidates: int = 9999  # Effectively uncapped — user trades time for quality

    # Model swap
    swap_script: str = ""  # Path to swap_vllm_model.sh
    swap_timeout_sec: int = 180

    # ── Canary system ───────────────────────────────────────────────────
    canary_enabled: bool = True
    canary_action: str = "fail"             # "fail" or "warn"

    # Canary 1: Classify empty-response detector
    canary_window_count: int = 20           # Check after this many sub-windows
    canary_min_nonempty_fraction: float = 0.05  # Fail if < 5% produced events

    # Canary 2: Triage distribution sanity check
    triage_canary_max_single_label_pct: float = 0.92  # Fail if any label > 92%
    triage_canary_min_active_pct: float = 0.03        # Fail if < 3% active flags

    # Canary 3: Model swap health ping
    swap_canary_enabled: bool = True  # Send test prompt after 32B swap

    # Canary 4: Event type diversity (mid-classify, 50% mark)
    diversity_canary_min_types: int = 2     # Expect ≥2 distinct event types
    diversity_canary_check_after: int = 50  # Check after this many events

    # Canary 5: vLLM latency tracking
    latency_canary_max_p95_sec: float = 180.0  # Warn if p95 > 3 min

    # ── YOLO spatial grounding (Run #33 breakthrough) ──────────────────
    yolo_grounding_enabled: bool = False
    yolo_grounding_fail_open: bool = True
    yolo_grounding_frames: int = 5
    yolo_grounding_frame_span_sec: float = 2.0
    yolo_grounding_inference_size: int = 640
    yolo_grounding_ball_conf: float = 0.15
    yolo_model_path: str = ""              # Path to YOLOv8 weights
    yolo_use_gpu: bool = True


class DualPassDetector:
    """Orchestrates the 8B triage → swap → 32B classification pipeline."""

    def __init__(
        self,
        config: DualPassConfig,
        source_file: str,
        video_duration: float,
        job_id: str,
        working_dir: str,
    ):
        self._cfg = config
        self._source_file = source_file
        self._video_duration = video_duration
        self._job_id = job_id
        self._working_dir = Path(working_dir)
        self._sampler = FrameSampler(source_file, frame_width=config.frame_width)

    def detect(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[Event]:
        """Run the detection pipeline.

        Routes to single-pass (32B direct) or dual-pass (triage→observe→classify)
        based on config.single_pass.
        """
        if self._cfg.single_pass:
            return self._single_pass_detect(progress_callback)
        return self._dual_pass_detect(progress_callback)

    def _single_pass_detect(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[Event]:
        """Single-pass: slide frames across the game, 32B classifies directly.

        Progress allocation:
            0.00 - 0.95: Sliding window classification
            0.95 - 1.00: Post-processing
        """
        import httpx
        import time

        def _progress(pct: float):
            if progress_callback:
                progress_callback(min(1.0, pct))

        _progress(0.0)

        step = self._cfg.single_pass_step_sec
        window = self._cfg.single_pass_window_sec
        n_frames = self._cfg.single_pass_frames
        model = self._cfg.tier2_model_name  # 32B

        total_windows = int(self._video_duration / step) + 1
        log.info("single_pass.start",
                 job_id=self._job_id,
                 total_windows=total_windows,
                 step=step, window=window, n_frames=n_frames,
                 model=model)

        all_events: list[Event] = []
        latencies: list[float] = []
        n_nonempty = 0
        diag_dir = self._working_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag_file = open(diag_dir / "single_pass_windows.jsonl", "w")

        t = 0.0
        win_idx = 0
        while t < self._video_duration:
            win_start = max(0.0, t - (window - step) / 2)
            win_end = min(self._video_duration, win_start + window)

            # Sample frames
            center = (win_start + win_end) / 2
            half_span = (win_end - win_start) / 2
            interval = max(1.0, (win_end - win_start) / n_frames)
            frames = self._sampler.sample_range(
                center_sec=center,
                window_sec=half_span,
                interval_sec=interval,
                duration_sec=self._video_duration,
            )
            if not frames:
                t += step
                win_idx += 1
                continue

            # Cap frames
            if len(frames) > n_frames:
                s = len(frames) / n_frames
                indices = [int(i * s) for i in range(n_frames)]
                frames = [frames[i] for i in indices]

            # Build prompt with images
            prompt = _DIRECT_CLASSIFY_PROMPT.format(
                n_frames=len(frames),
                start=win_start,
                end=win_end,
            )

            content: list[dict] = []
            for frame in frames:
                b64 = base64.b64encode(frame.jpeg_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
                content.append({
                    "type": "text",
                    "text": f"t={frame.timestamp_sec:.1f}s",
                })
            content.append({"type": "text", "text": prompt})

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 800,
                "temperature": 0,
            }

            events_this_window: list[Event] = []
            try:
                t0 = time.monotonic()
                r = httpx.post(
                    f"{self._cfg.vllm_url}/v1/chat/completions",
                    json=payload,
                    timeout=self._cfg.single_pass_timeout_sec,
                )
                elapsed = time.monotonic() - t0
                latencies.append(elapsed)

                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"]["content"].strip()

                    # Parse into events
                    tmp_window = CandidateWindow(
                        start_sec=win_start, end_sec=win_end,
                        labels=["DIRECT"], flags=[],
                    )
                    events_this_window = self._parse_classify_response(
                        text, tmp_window
                    )
                    # Filter out "none" events
                    events_this_window = [
                        e for e in events_this_window
                        if e.event_type.value != "none"
                    ]
                    if events_this_window:
                        n_nonempty += 1
                    all_events.extend(events_this_window)

                    log.info("single_pass.window",
                             idx=win_idx, start=win_start, end=win_end,
                             elapsed=f"{elapsed:.1f}s",
                             events=len(events_this_window),
                             text=text[:200])
                else:
                    log.warning("single_pass.api_error",
                                status=r.status_code, body=r.text[:200])

            except Exception as exc:
                log.warning("single_pass.exception",
                            error=str(exc), start=win_start)

            # Diagnostics
            json.dump({
                "window_idx": win_idx,
                "start_sec": win_start,
                "end_sec": win_end,
                "n_frames": len(frames),
                "n_events": len(events_this_window),
                "event_types": [e.event_type.value for e in events_this_window],
                "latency": latencies[-1] if latencies else None,
            }, diag_file)
            diag_file.write("\n")

            # Progress
            _progress(0.95 * (win_idx + 1) / total_windows)

            # Log latency stats periodically
            if win_idx > 0 and win_idx % 50 == 0:
                p50 = sorted(latencies)[len(latencies) // 2]
                log.info("single_pass.progress",
                         windows=win_idx, total=total_windows,
                         events=len(all_events), nonempty=n_nonempty,
                         p50_latency=f"{p50:.1f}s")

            t += step
            win_idx += 1

        diag_file.close()

        log.info("single_pass.scan_complete",
                 job_id=self._job_id,
                 windows=win_idx, events_raw=len(all_events),
                 nonempty_windows=n_nonempty)

        # Deduplicate and post-filter (reuse existing methods)
        all_events = self._deduplicate_events(all_events)

        # YOLO spatial grounding (Run #33) — reject events whose spatial
        # prerequisites are contradicted by ball/person positions.
        if self._cfg.yolo_grounding_enabled:
            all_events = self._apply_yolo_grounding(all_events)

        all_events = self._post_filter_events(all_events)

        # Save final events
        self._save_classify_diagnostics([], all_events)

        log.info("single_pass.complete",
                 job_id=self._job_id, events=len(all_events))

        _progress(1.0)
        return all_events

    def _dual_pass_detect(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[Event]:
        """Legacy dual-pass: triage → observe → classify.

        Progress allocation:
            0.00 - 0.02: Model swap to 8B
            0.02 - 0.40: 8B triage scan
            0.40 - 0.55: 8B observe (images → text descriptions)
            0.55 - 0.60: Model swap to 32B
            0.60 - 0.95: 32B classification (text-only)
            0.95 - 1.00: Post-processing

        Returns:
            List of Event objects.
        """
        def _progress(pct: float):
            if progress_callback:
                progress_callback(min(1.0, pct))

        _progress(0.0)

        # ── Phase 1: Ensure 8B model is loaded ─────────────────────────
        log.info("dual_pass.phase1_swap_to_8b", job_id=self._job_id)
        self._swap_model(
            model_name=self._cfg.tier1_model_name,
            model_path=self._cfg.tier1_model_path,
            tier="tier1",
        )
        _progress(0.02)

        # ── Phase 2: 8B triage scan ────────────────────────────────────
        log.info("dual_pass.phase2_triage", job_id=self._job_id)
        scanner = TriageScanner(
            vllm_url=self._cfg.vllm_url,
            vllm_model=self._cfg.tier1_model_name,
            source_file=self._source_file,
            video_duration=self._video_duration,
            frame_width=self._cfg.frame_width,
            frames_per_window=self._cfg.frames_per_window,
            window_span_sec=self._cfg.window_span_sec,
            step_sec=self._cfg.step_sec,
        )

        def on_triage_progress(pct: float):
            _progress(0.02 + pct * 0.38)

        scan_result = scanner.scan(progress_callback=on_triage_progress)
        flags = scan_result.flags

        # ── Canary 2: Triage distribution sanity check ────────────────
        if self._cfg.canary_enabled:
            self._check_triage_canary(scan_result)

        # Merge nearby flags into candidate windows
        candidates = merge_flags(
            flags,
            merge_gap_sec=self._cfg.merge_gap_sec,
            pad_sec=self._cfg.merge_pad_sec,
            max_window_sec=self._cfg.max_window_sec,
        )

        # Cap candidates to stay within time budget.  Rank by triage label
        # priority so we keep SHOT_SAVE/GOAL/SET_PIECE over generic ATTACK.
        _LABEL_PRIORITY = {"GOAL": 0, "SHOT_SAVE": 1, "SET_PIECE": 2, "ATTACK": 3}
        if len(candidates) > self._cfg.max_candidates:
            def _score(w):
                best = min(_LABEL_PRIORITY.get(l, 99) for l in w.labels) if w.labels else 99
                return (best, w.start_sec)
            candidates.sort(key=_score)
            dropped = len(candidates) - self._cfg.max_candidates
            candidates = candidates[:self._cfg.max_candidates]
            # Re-sort by time for sequential processing
            candidates.sort(key=lambda w: w.start_sec)
            log.info("dual_pass.candidates_capped",
                     kept=len(candidates), dropped=dropped)

        # Save diagnostics
        self._save_triage_diagnostics(flags, candidates)

        log.info("dual_pass.triage_complete",
                 job_id=self._job_id,
                 flags=len(flags),
                 candidates=len(candidates),
                 total_review_sec=sum(w.duration_sec for w in candidates))

        if not candidates:
            _progress(1.0)
            return []

        _progress(0.40)

        # ── Phase 3: 8B observe — images → text descriptions ──────────
        # Still on the 8B model.  Send frames and collect free-text
        # descriptions that the 32B will classify without seeing images.
        log.info("dual_pass.phase3_observe",
                 job_id=self._job_id, windows=len(candidates))

        observations: dict[int, str] = {}  # candidate index → description
        for idx, window in enumerate(candidates):
            obs_text = self._observe_window(window)
            observations[idx] = obs_text
            _progress(0.40 + 0.15 * (idx + 1) / len(candidates))

        log.info("dual_pass.observe_complete",
                 job_id=self._job_id,
                 windows=len(candidates),
                 nonempty=sum(1 for v in observations.values() if v))

        # Save observation diagnostics for debugging
        self._save_observation_diagnostics(candidates, observations)

        _progress(0.55)

        # ── Phase 4: Model swap to 32B ─────────────────────────────────
        log.info("dual_pass.phase4_swap_to_32b", job_id=self._job_id)
        self._swap_model(
            model_name=self._cfg.tier2_model_name,
            model_path=self._cfg.tier2_model_path,
            tier="tier2",
        )

        # ── Canary 3: Model swap health ping ──────────────────────────
        if self._cfg.canary_enabled and self._cfg.swap_canary_enabled:
            self._check_swap_canary()

        _progress(0.60)

        # ── Phase 5: 32B classification — text-only ────────────────────
        log.info("dual_pass.phase5_classify",
                 job_id=self._job_id, windows=len(candidates))

        all_events: list[Event] = []
        self._canary_sub_total = 0
        self._canary_sub_nonempty = 0
        self._canary_checked = False
        self._classify_latencies: list[float] = []
        self._diversity_checked = False

        for idx, window in enumerate(candidates):
            observation = observations.get(idx, "")
            if not observation:
                log.info("dual_pass.skip_empty_observation",
                         window_start=window.start_sec)
                self._canary_sub_total += 1
                continue

            events = self._classify_window(window, observation)
            all_events.extend(events)

            # Canary 1: Classify empty-response check after N sub-windows
            if (self._cfg.canary_enabled
                    and not self._canary_checked
                    and self._canary_sub_total >= self._cfg.canary_window_count):
                self._check_classify_canary()

            # Canary 4: Event type diversity (after enough events detected)
            if (self._cfg.canary_enabled
                    and not self._diversity_checked
                    and len(all_events) >= self._cfg.diversity_canary_check_after):
                self._check_diversity_canary(all_events)

            # Canary 5: vLLM latency tracking (log every 50 sub-windows)
            if (self._cfg.canary_enabled
                    and self._canary_sub_total > 0
                    and self._canary_sub_total % 50 == 0):
                self._check_latency_canary()

            _progress(0.60 + 0.35 * (idx + 1) / len(candidates))

        # Deduplicate events that overlap significantly
        all_events = self._deduplicate_events(all_events)

        # G2-7: Post-filter — suppress contextually invalid events
        all_events = self._post_filter_events(all_events)

        # Save classification diagnostics
        self._save_classify_diagnostics(candidates, all_events)

        log.info("dual_pass.complete",
                 job_id=self._job_id,
                 events=len(all_events))

        _progress(1.0)
        return all_events

    def _swap_model(self, model_name: str, model_path: str, tier: str) -> None:
        """Swap the vLLM model via the swap script."""
        swap_script = self._cfg.swap_script
        if not swap_script:
            # Look for default location
            repo_root = Path(__file__).parent.parent.parent
            swap_script = str(repo_root / "scripts" / "swap_vllm_model.sh")

        if not Path(swap_script).exists():
            log.warning("dual_pass.no_swap_script", path=swap_script)
            return

        env = os.environ.copy()
        env["SWAP_TARGET_NAME"] = model_name
        env["SWAP_TARGET_PATH"] = model_path
        env["SWAP_TARGET_TIER"] = tier
        env["SWAP_VLLM_URL"] = self._cfg.vllm_url

        log.info("dual_pass.swapping_model",
                 model=model_name, tier=tier, script=swap_script)

        try:
            result = subprocess.run(
                ["bash", swap_script],
                env=env,
                capture_output=True,
                text=True,
                timeout=self._cfg.swap_timeout_sec,
            )
            if result.returncode != 0:
                log.error("dual_pass.swap_failed",
                          rc=result.returncode,
                          stdout=result.stdout[-500:],
                          stderr=result.stderr[-500:])
                raise RuntimeError(f"Model swap failed (rc={result.returncode})")
            log.info("dual_pass.swap_complete", model=model_name,
                     output=result.stdout[-200:])
        except subprocess.TimeoutExpired:
            log.error("dual_pass.swap_timeout", timeout=self._cfg.swap_timeout_sec)
            raise RuntimeError("Model swap timed out")

    # ── 8B Observe methods ──────────────────────────────────────────────

    def _observe_window(self, window: CandidateWindow) -> str:
        """Send frames from a candidate window to the 8B and get a text description.

        For long windows (>sub_window_sec), splits into sub-windows and
        concatenates descriptions.  Uses more frames than classify since
        the 8B handles them well.
        """
        sub_size = self._cfg.sub_window_sec
        overlap = self._cfg.sub_window_overlap_sec

        if window.duration_sec <= sub_size + overlap:
            return self._observe_sub_window(
                start_sec=window.start_sec,
                end_sec=window.end_sec,
                triage_labels=list(set(window.labels)),
            )

        # Split into overlapping sub-windows
        step = sub_size - overlap
        parts: list[str] = []
        t = window.start_sec
        while t < window.end_sec:
            end = min(t + sub_size, window.end_sec)
            obs = self._observe_sub_window(
                start_sec=t,
                end_sec=end,
                triage_labels=list(set(window.labels)),
            )
            if obs:
                parts.append(f"[{t:.0f}s–{end:.0f}s]: {obs}")
            t += step
            if end >= window.end_sec:
                break

        return "\n\n".join(parts)

    def _observe_sub_window(
        self,
        start_sec: float,
        end_sec: float,
        triage_labels: list[str],
    ) -> str:
        """Send frames from a sub-window to the 8B for observation."""
        import httpx
        import time

        duration = end_sec - start_sec
        # Sample more frames — 8B handles 8-10 well
        interval = max(1.0, duration / self._cfg.observe_max_frames)

        frames = self._sampler.sample_range(
            center_sec=(start_sec + end_sec) / 2,
            window_sec=duration / 2,
            interval_sec=interval,
            duration_sec=self._video_duration,
        )

        if not frames:
            return ""

        # Cap to observe_max_frames
        if len(frames) > self._cfg.observe_max_frames:
            step = len(frames) / self._cfg.observe_max_frames
            indices = [int(i * step) for i in range(self._cfg.observe_max_frames)]
            frames = [frames[i] for i in indices]

        prompt = _OBSERVE_PROMPT.format(
            n_frames=len(frames),
            duration=duration,
            start=start_sec,
            end=end_sec,
            triage_labels=", ".join(triage_labels),
        )

        content: list[dict] = []
        for frame in frames:
            b64 = base64.b64encode(frame.jpeg_bytes).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
            content.append({
                "type": "text",
                "text": f"t={frame.timestamp_sec:.1f}s",
            })
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self._cfg.tier1_model_name,  # 8B — handles images well
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 600,  # Free-form with dead-ball focus
            "temperature": 0,
        }

        try:
            t0 = time.monotonic()
            r = httpx.post(
                f"{self._cfg.vllm_url}/v1/chat/completions",
                json=payload,
                timeout=self._cfg.observe_timeout_sec,
            )
            elapsed = time.monotonic() - t0

            if r.status_code != 200:
                log.warning("dual_pass.observe_error",
                            status=r.status_code, body=r.text[:200],
                            sub_start=start_sec)
                return ""

            text = r.json()["choices"][0]["message"]["content"].strip()
            log.info("dual_pass.observe_result",
                     sub_start=start_sec, elapsed=f"{elapsed:.1f}s",
                     n_frames=len(frames), text_len=len(text),
                     text=text[:300])
            return text

        except Exception as exc:
            log.warning("dual_pass.observe_exception",
                        error=str(exc), sub_start=start_sec)
            return ""

    # ── 32B Classify methods (text-only) ───────────────────────────────

    def _classify_window(
        self, window: CandidateWindow, observation: str
    ) -> list[Event]:
        """Classify a candidate window using the 32B with text-only input.

        The observation text from the 8B is sent to the 32B along with the
        full detailed classify prompt.  No images are sent to the 32B.
        """
        events = self._classify_from_observation(
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            observation=observation,
            triage_labels=list(set(window.labels)),
        )
        return events

    def _classify_from_observation(
        self,
        start_sec: float,
        end_sec: float,
        observation: str,
        triage_labels: list[str],
    ) -> list[Event]:
        """Send observation text to the 32B model for classification (no images)."""
        import httpx
        import time

        duration = end_sec - start_sec

        prompt = _CLASSIFY_PROMPT.format(
            duration=duration,
            start=start_sec,
            end=end_sec,
            observation=observation,
        )

        # Text-only — no images, no length constraints
        payload = {
            "model": self._cfg.tier2_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0,
        }

        try:
            t0 = time.monotonic()
            r = httpx.post(
                f"{self._cfg.vllm_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            elapsed = time.monotonic() - t0
            self._classify_latencies.append(elapsed)

            if r.status_code != 200:
                log.warning("dual_pass.classify_error",
                            status=r.status_code, body=r.text[:200],
                            sub_start=start_sec)
                self._canary_sub_total += 1
                return []

            text = r.json()["choices"][0]["message"]["content"].strip()
            log.info("dual_pass.raw_32b_response",
                     sub_start=start_sec, elapsed=f"{elapsed:.1f}s",
                     text=text[:300])

            tmp_window = CandidateWindow(
                start_sec=start_sec, end_sec=end_sec,
                labels=triage_labels, flags=[],
            )
            events = self._parse_classify_response(text, tmp_window)

            # Track canary counters
            self._canary_sub_total += 1
            if events:
                self._canary_sub_nonempty += 1

            return events

        except Exception as exc:
            log.warning("dual_pass.classify_exception",
                        error=str(exc), sub_start=start_sec)
            self._canary_sub_total += 1
            return []

    def _parse_classify_response(
        self, text: str, window: CandidateWindow
    ) -> list[Event]:
        """Parse 32B classification response into Event objects."""
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            log.warning("dual_pass.classify_parse_error",
                        text=text[:300], window_start=window.start_sec)
            return []

        if not isinstance(items, list):
            items = [items]

        events: list[Event] = []
        for item in items:
            try:
                event_type_str = item.get("event_type", "")
                try:
                    event_type = EventType(event_type_str)
                except ValueError:
                    log.debug("dual_pass.unknown_event_type",
                              event_type=event_type_str)
                    continue

                start = float(item.get("start_sec", window.start_sec))
                end = float(item.get("end_sec", start + 5))
                conf = float(item.get("confidence", 0.7))
                reasoning = str(item.get("reasoning", ""))

                # G2-5: Confidence floor — drop low-confidence events
                # Run #13: raised from 0.4 → 0.65 (32B is confidently wrong
                # on many false positives, but this still helps at the margin)
                if conf < 0.65:
                    log.info("dual_pass.low_confidence_dropped",
                             event_type=event_type_str, conf=conf,
                             start=start)
                    continue

                # Clamp to video bounds
                start = max(0.0, start)
                end = min(self._video_duration, end)
                if end <= start:
                    continue

                # Determine if this is a GK event
                from src.detection.models import is_gk_event_type
                is_gk = is_gk_event_type(event_type)

                events.append(Event(
                    event_id=str(uuid.uuid4()),
                    job_id=self._job_id,
                    source_file=self._source_file,
                    event_type=event_type,
                    timestamp_start=start,
                    timestamp_end=end,
                    confidence=conf,
                    reel_targets=[],  # Assigned by pipeline based on reel specs
                    is_goalkeeper_event=is_gk,
                    frame_start=int(start * 30),
                    frame_end=int(end * 30),
                    reviewed=False,
                    review_override=None,
                    metadata={
                        "detection_method": "dual_pass",
                        "triage_labels": list(set(window.labels)),
                        "vlm_reasoning": reasoning,
                        "vlm_model": self._cfg.tier2_model_name,
                        "vlm_confidence": conf,
                    },
                ))
            except (KeyError, TypeError, ValueError) as exc:
                log.debug("dual_pass.event_parse_error", error=str(exc))
                continue

        return events

    def _deduplicate_events(
        self, events: list[Event], cluster_gap_sec: float = 30.0
    ) -> list[Event]:
        """Temporal clustering: merge same-type events within cluster_gap_sec.

        Groups events of the same type that are within cluster_gap_sec of
        each other, then keeps only the highest-confidence event per cluster.
        This collapses overlapping sub-window detections of the same real
        event into a single detection.

        Run #13 analysis: 93 goals → 61 clusters (30s gap).  The old
        overlap-based dedup only caught exact overlaps, missing events
        from adjacent sub-windows that describe the same moment.
        """
        if len(events) <= 1:
            return events

        # Group by event type
        by_type: dict[EventType, list[Event]] = {}
        for e in events:
            by_type.setdefault(e.event_type, []).append(e)

        kept: list[Event] = []
        for etype, typed_events in by_type.items():
            # Sort by start time
            typed_events.sort(key=lambda e: e.timestamp_start)

            # Cluster events within gap
            clusters: list[list[Event]] = []
            for e in typed_events:
                if not clusters or e.timestamp_start - clusters[-1][-1].timestamp_start > cluster_gap_sec:
                    clusters.append([])
                clusters[-1].append(e)

            # Keep highest-confidence event per cluster, stamping the
            # cluster size so downstream gates can reason about density
            # (e.g. kickoff-inference needs ≥2 merged tags to trust the
            # signal).
            for cluster in clusters:
                best = max(cluster, key=lambda e: e.confidence)
                if len(cluster) > 1:
                    best = Event(
                        event_id=best.event_id,
                        job_id=best.job_id,
                        source_file=best.source_file,
                        event_type=best.event_type,
                        timestamp_start=best.timestamp_start,
                        timestamp_end=best.timestamp_end,
                        confidence=best.confidence,
                        reel_targets=best.reel_targets,
                        is_goalkeeper_event=best.is_goalkeeper_event,
                        frame_start=best.frame_start,
                        frame_end=best.frame_end,
                        reviewed=best.reviewed,
                        review_override=best.review_override,
                        metadata={
                            **best.metadata,
                            "cluster_size": len(cluster),
                            "cluster_span_sec": cluster[-1].timestamp_start - cluster[0].timestamp_start,
                        },
                    )
                kept.append(best)

            if len(clusters) < len(typed_events):
                log.info("dual_pass.clustered",
                         event_type=etype.value,
                         before=len(typed_events),
                         clusters=len(clusters))

        # Re-sort by time
        kept.sort(key=lambda e: e.timestamp_start)

        if len(kept) < len(events):
            log.info("dual_pass.deduplicated",
                     before=len(events), after=len(kept))
        return kept

    def _apply_yolo_grounding(self, events: list[Event]) -> list[Event]:
        """Run the YOLO spatial grounding gate over detected events.

        See src/detection/yolo_grounding.py for the per-type rules. This
        is a precision gate only — fail-open by default so YOLO outages
        don't cascade into recall loss.
        """
        from src.detection.yolo_grounding import YoloGrounder

        diag_path = self._working_dir / "diagnostics" / "yolo_grounding.jsonl"
        grounder = YoloGrounder(
            sampler=self._sampler,
            video_duration=self._video_duration,
            model_path=self._cfg.yolo_model_path or None,
            inference_size=self._cfg.yolo_grounding_inference_size,
            use_gpu=self._cfg.yolo_use_gpu,
            ball_conf_threshold=self._cfg.yolo_grounding_ball_conf,
            n_frames=self._cfg.yolo_grounding_frames,
            frame_span_sec=self._cfg.yolo_grounding_frame_span_sec,
            fail_open=self._cfg.yolo_grounding_fail_open,
            diagnostics_path=diag_path,
        )
        try:
            return grounder.filter(events)
        finally:
            grounder.close()

    def _post_filter_events(self, events: list[Event]) -> list[Event]:
        """Apply contextual post-filters.

        - save-language goal gate: goal whose reasoning describes a save
          (keeper diving, ball rebounding) → demote to shot_on_target
        - kickoff-based goal inference: kickoff within 60s after a shot → goal
        - goal keyword gate (dual-pass only): demote goal without evidence
        - kickoff suppression: GT doesn't score kickoffs.
        - shot_stop_standing → catch promotion.
        """
        # ── Pass 1: Collect + validate kickoff events for goal inference
        # Run #27 (broken): cluster-check failed because _deduplicate_events
        # had already collapsed 35 window-level kickoff tags → 16 events
        # (30s merge gap eats the very clusters we want to count).
        # Run #28: use metadata["cluster_size"] stamped during dedup, which
        # records the original merge density. cluster_size ≥ 2 means the
        # kept event represents multiple raw tags within 30s — the real
        # post-goal kickoff pattern. Isolated tags (cluster_size == 1) are
        # the midfield-free-kick / generic-restart false positives.
        raw_kickoffs = [e for e in events if e.event_type == EventType.KICKOFF]
        validated_kickoffs = [
            e for e in raw_kickoffs
            if _has_real_kickoff_evidence(e.metadata.get("vlm_reasoning", ""))
            and e.metadata.get("cluster_size", 1) >= 2
        ]
        kickoff_times = sorted(e.timestamp_start for e in validated_kickoffs)
        log.info("dual_pass.kickoff_filter",
                 raw=len(raw_kickoffs),
                 validated=len(validated_kickoffs),
                 kept_times=kickoff_times)

        # Shot times for catch-preceding-shot gate (Run 29). A catch in
        # GT requires the GK to have caught an opponent's SHOT. Without a
        # preceding shot, the "catch" is usually the GK picking up a ball
        # after a goal kick, throw-in, or clearance — not a real catch.
        # Run 31 (R10): also accept CORNER_KICK as a preceding event. A
        # corner → cross → GK-catches-the-ball sequence is a real catch;
        # Run 30 showed 3 catch TPs evaporated when shots got reclassified
        # as corners (corner detection unblocked) and the catch gate
        # dropped them for lacking a "preceding shot".
        _CATCH_SHOT_TYPES = {
            EventType.SHOT_ON_TARGET,
            EventType.SHOT_STOP_DIVING,
            EventType.SHOT_STOP_STANDING,
            EventType.GOAL,
            EventType.FREE_KICK_SHOT,
            EventType.PENALTY,
            EventType.CORNER_KICK,
        }
        shot_times = sorted(
            e.timestamp_start for e in events
            if e.event_type in _CATCH_SHOT_TYPES
        )

        # ── Pass 2: Apply filters ─────────────────────────────────────
        kept: list[Event] = []
        dropped = 0
        goal_demoted = 0
        goal_inferred = 0
        goal_suppressed = 0
        catch_dropped_no_shot = 0
        for event in events:
            # Drop kickoffs from output (GT doesn't score them)
            if event.event_type == EventType.KICKOFF:
                dropped += 1
                continue

            # ── Goal keyword gate (dual-pass only) ────────────────────
            # In dual-pass mode, the 32B only sees text from the 8B and
            # hallucinates goals from "ball near goal" descriptions.
            # In single-pass mode, the 32B sees actual frames — trust it.
            if event.event_type == EventType.GOAL and not self._cfg.single_pass:
                reasoning_lower = event.metadata.get("vlm_reasoning", "").lower()
                has_evidence = _has_positive_goal_evidence(reasoning_lower)
                if not has_evidence:
                    log.info("dual_pass.goal_demoted",
                             start=event.timestamp_start,
                             reasoning=reasoning_lower[:150])
                    event = Event(
                        event_id=event.event_id,
                        job_id=event.job_id,
                        source_file=event.source_file,
                        event_type=EventType.SHOT_ON_TARGET,
                        timestamp_start=event.timestamp_start,
                        timestamp_end=event.timestamp_end,
                        confidence=event.confidence,
                        reel_targets=event.reel_targets,
                        is_goalkeeper_event=False,
                        frame_start=event.frame_start,
                        frame_end=event.frame_end,
                        reviewed=event.reviewed,
                        review_override=event.review_override,
                        metadata={**event.metadata,
                                  "goal_demoted": True,
                                  "original_event_type": "goal"},
                    )
                    goal_demoted += 1

            # ── Kickoff-based goal inference ──────────────────────────
            # A kickoff ALWAYS follows a goal. If a shot/save is detected
            # and a kickoff appears 10-90s later, upgrade to goal.
            _SHOT_TYPES = {
                EventType.SHOT_ON_TARGET,
                EventType.SHOT_STOP_DIVING,
                EventType.SHOT_STOP_STANDING,
            }
            if event.event_type in _SHOT_TYPES:
                t = event.timestamp_start
                has_kickoff_after = any(
                    10 <= (kt - t) <= 90 for kt in kickoff_times
                )
                if has_kickoff_after:
                    log.info("dual_pass.goal_inferred_from_kickoff",
                             start=t, original_type=event.event_type.value)
                    event = Event(
                        event_id=event.event_id,
                        job_id=event.job_id,
                        source_file=event.source_file,
                        event_type=EventType.GOAL,
                        timestamp_start=event.timestamp_start,
                        timestamp_end=event.timestamp_end,
                        confidence=min(event.confidence + 0.1, 1.0),
                        reel_targets=event.reel_targets,
                        is_goalkeeper_event=False,
                        frame_start=event.frame_start,
                        frame_end=event.frame_end,
                        reviewed=event.reviewed,
                        review_override=event.review_override,
                        metadata={**event.metadata,
                                  "goal_inferred": True,
                                  "kickoff_signal": True,
                                  "original_event_type": event.event_type.value},
                    )
                    goal_inferred += 1

            # Reclassify shot_stop_standing → catch
            if event.event_type == EventType.SHOT_STOP_STANDING:
                event = Event(
                    event_id=event.event_id,
                    job_id=event.job_id,
                    source_file=event.source_file,
                    event_type=EventType.CATCH,
                    timestamp_start=event.timestamp_start,
                    timestamp_end=event.timestamp_end,
                    confidence=event.confidence,
                    reel_targets=event.reel_targets,
                    is_goalkeeper_event=True,
                    frame_start=event.frame_start,
                    frame_end=event.frame_end,
                    reviewed=event.reviewed,
                    review_override=event.review_override,
                    metadata=event.metadata,
                )

            # ── Catch preceding-shot gate (Run 29) ──────────────────────
            # Run 28 FP analysis: catch had 4 TP / 17 FP. All 4 TPs had a
            # preceding shot event within 10s (GK catching opponent's shot).
            # Most FPs were the VLM labeling GK-holding-ball-after-goal-kick
            # or GK-picking-up-ball-from-clearance as "catch" — not real
            # catches in the GT taxonomy.
            # Drop catches without a preceding shot/free-kick/penalty/goal
            # in the last 10s. Predicted cut: 12 of 17 FPs at the cost of
            # 1 TP (whose preceding shot was outside the 10s window).
            if event.event_type == EventType.CATCH:
                t = event.timestamp_start
                has_preceding_shot = any(0 < t - st <= 10 for st in shot_times)
                if not has_preceding_shot:
                    log.info("dual_pass.catch_dropped_no_preceding_shot",
                             start=t,
                             reasoning=event.metadata.get("vlm_reasoning", "")[:150])
                    catch_dropped_no_shot += 1
                    dropped += 1
                    continue

            # ── Save-language goal gate ─────────────────────────────────
            # If the VLM's own reasoning describes a save (keeper diving,
            # ball rebounding away), the event is a shot — not a goal.
            # Applies to both single-pass and dual-pass modes.
            #
            # EXCEPTION: events promoted to goal by the kickoff-inference
            # gate above carry metadata["kickoff_signal"]=True. A kickoff
            # only follows a goal (or a half-start, ruled out by timing).
            # The temporal evidence trumps in-window save language — a
            # save followed by a kickoff means the rebound went in. Without
            # this exception the two gates fight: kickoff-inference promotes
            # shot→goal, save-language demotes goal→shot_on_target, net zero.
            if event.event_type == EventType.GOAL \
                    and not event.metadata.get("kickoff_signal"):
                reasoning = event.metadata.get("vlm_reasoning", "")
                if _has_save_language(reasoning):
                    log.info("dual_pass.goal_suppressed_save_language",
                             start=event.timestamp_start,
                             reasoning=reasoning[:150])
                    event = Event(
                        event_id=event.event_id,
                        job_id=event.job_id,
                        source_file=event.source_file,
                        event_type=EventType.SHOT_ON_TARGET,
                        timestamp_start=event.timestamp_start,
                        timestamp_end=event.timestamp_end,
                        confidence=event.confidence,
                        reel_targets=event.reel_targets,
                        is_goalkeeper_event=False,
                        frame_start=event.frame_start,
                        frame_end=event.frame_end,
                        reviewed=event.reviewed,
                        review_override=event.review_override,
                        metadata={**event.metadata,
                                  "goal_suppressed": True,
                                  "suppression_reason": "save_language",
                                  "original_event_type": "goal"},
                    )
                    goal_suppressed += 1

            kept.append(event)

        if dropped or goal_demoted or goal_inferred or goal_suppressed or catch_dropped_no_shot:
            log.info("dual_pass.post_filtered",
                     dropped=dropped, goal_demoted=goal_demoted,
                     goal_inferred=goal_inferred,
                     goal_suppressed=goal_suppressed,
                     catch_dropped_no_shot=catch_dropped_no_shot,
                     kept=len(kept))
        return kept

    # ── Canary methods ──────────────────────────────────────────────────

    def _check_triage_canary(self, result: TriageScanResult) -> None:
        """Canary 2: Validate triage label distribution is sane.

        Fires after triage completes, before model swap. Detects:
        - Degenerate distributions (>92% single label = broken triage prompt)
        - Too few active flags (<3% = triage too conservative, nothing for 32B)
        """
        total = result.total_windows
        if total == 0:
            return

        counts = result.label_counts
        active_count = sum(counts.get(l, 0) for l in ACTIVE_LABELS)
        active_pct = active_count / total

        # Check single-label dominance
        for label, count in counts.items():
            pct = count / total
            if pct > self._cfg.triage_canary_max_single_label_pct:
                msg = (
                    f"Triage canary: label '{label}' accounts for "
                    f"{pct:.1%} of {total} windows (threshold: "
                    f"{self._cfg.triage_canary_max_single_label_pct:.0%}). "
                    f"Distribution: {counts}. Likely 8B prompt regression."
                )
                if self._cfg.canary_action == "fail":
                    log.critical("dual_pass.triage_canary_failed",
                                 dominant_label=label, pct=f"{pct:.1%}",
                                 counts=counts)
                    raise CanaryFailure(msg)
                else:
                    log.critical("dual_pass.triage_canary_warning", msg=msg)

        # Check minimum active fraction
        if active_pct < self._cfg.triage_canary_min_active_pct:
            msg = (
                f"Triage canary: only {active_pct:.1%} of {total} windows "
                f"are active (threshold: "
                f"{self._cfg.triage_canary_min_active_pct:.0%}). "
                f"Distribution: {counts}. Triage is too conservative."
            )
            if self._cfg.canary_action == "fail":
                log.critical("dual_pass.triage_canary_failed",
                             active_pct=f"{active_pct:.1%}", counts=counts)
                raise CanaryFailure(msg)
            else:
                log.critical("dual_pass.triage_canary_warning", msg=msg)

        log.info("dual_pass.triage_canary_passed",
                 active_pct=f"{active_pct:.1%}",
                 total_windows=total, counts=counts)

    def _check_swap_canary(self) -> None:
        """Canary 3: Verify the 32B model responds coherently after swap.

        Sends a minimal text-only prompt to the 32B and checks for a valid
        JSON response. Catches model load failures, OOM, or wrong model.
        """
        import httpx

        test_payload = {
            "model": self._cfg.tier2_model_name,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": (
                    "Respond with ONLY this exact JSON: "
                    '[{"event_type": "test", "status": "ok"}]'
                )},
            ]}],
            "max_tokens": 50,
            "temperature": 0,
        }

        try:
            r = httpx.post(
                f"{self._cfg.vllm_url}/v1/chat/completions",
                json=test_payload,
                timeout=30,
            )
            if r.status_code != 200:
                msg = (
                    f"Swap canary: 32B health ping returned HTTP {r.status_code}. "
                    f"Body: {r.text[:200]}. Model may not have loaded correctly."
                )
                log.critical("dual_pass.swap_canary_failed",
                             status=r.status_code)
                if self._cfg.canary_action == "fail":
                    raise CanaryFailure(msg)
                else:
                    log.critical("dual_pass.swap_canary_warning", msg=msg)
                return

            text = r.json()["choices"][0]["message"]["content"].strip()
            # Just verify we got some response — don't require exact JSON
            if len(text) < 2:
                msg = f"Swap canary: 32B returned empty response: '{text}'"
                log.critical("dual_pass.swap_canary_failed", text=text)
                if self._cfg.canary_action == "fail":
                    raise CanaryFailure(msg)
            else:
                model_id = r.json().get("model", "unknown")
                log.info("dual_pass.swap_canary_passed",
                         model=model_id, response=text[:100])

        except httpx.TimeoutException:
            msg = "Swap canary: 32B health ping timed out after 30s"
            log.critical("dual_pass.swap_canary_timeout")
            if self._cfg.canary_action == "fail":
                raise CanaryFailure(msg)

    def _check_diversity_canary(self, events: list[Event]) -> None:
        """Canary 4: Check that detected events aren't all the same type.

        Fires once after diversity_canary_check_after events. If the 32B
        is always returning the same event type, the classify prompt is
        likely biased or broken.
        """
        self._diversity_checked = True
        type_counts: dict[str, int] = {}
        for e in events:
            t = e.event_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        n_types = len(type_counts)
        if n_types >= self._cfg.diversity_canary_min_types:
            log.info("dual_pass.diversity_canary_passed",
                     n_types=n_types, counts=type_counts,
                     total_events=len(events))
            return

        # Check if a single type dominates >90%
        dominant = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]
        dominant_pct = type_counts[dominant] / len(events)
        msg = (
            f"Diversity canary: only {n_types} event type(s) in first "
            f"{len(events)} events. '{dominant}' = {dominant_pct:.0%}. "
            f"Distribution: {type_counts}."
        )
        # This is a warning, not a failure — some games legitimately have
        # concentrated event types early on.
        log.warning("dual_pass.diversity_canary_warning",
                    n_types=n_types, counts=type_counts, msg=msg)

    def _check_latency_canary(self) -> None:
        """Canary 5: Monitor vLLM response latencies for anomalies.

        Logs p50/p95/max latency stats every 50 sub-windows. Warns if
        p95 exceeds the threshold (GPU throttling, OOM pressure, etc.).
        """
        if not self._classify_latencies:
            return

        latencies = sorted(self._classify_latencies)
        n = len(latencies)
        p50 = latencies[n // 2]
        p95 = latencies[int(n * 0.95)]
        p_max = latencies[-1]

        log.info("dual_pass.latency_stats",
                 n_calls=n, p50=f"{p50:.1f}s", p95=f"{p95:.1f}s",
                 max=f"{p_max:.1f}s")

        if p95 > self._cfg.latency_canary_max_p95_sec:
            log.warning("dual_pass.latency_canary_warning",
                        p95=f"{p95:.1f}s",
                        threshold=f"{self._cfg.latency_canary_max_p95_sec:.0f}s",
                        msg=(
                            f"vLLM p95 latency {p95:.1f}s exceeds "
                            f"{self._cfg.latency_canary_max_p95_sec:.0f}s threshold. "
                            f"Check GPU health / memory pressure."
                        ))

    def _check_classify_canary(self) -> None:
        """Check if the 32B classify phase is producing reasonable results.

        Fires once after canary_window_count sub-windows. If the fraction
        of non-empty responses is below the threshold, the run is likely
        broken (e.g., prompt regression causing all-empty returns).
        """
        self._canary_checked = True
        total = self._canary_sub_total
        nonempty = self._canary_sub_nonempty
        frac = nonempty / total if total > 0 else 0.0
        threshold = self._cfg.canary_min_nonempty_fraction

        if frac >= threshold:
            log.info("dual_pass.canary_passed",
                     nonempty=nonempty, total=total,
                     fraction=f"{frac:.1%}",
                     threshold=f"{threshold:.1%}")
            return

        msg = (
            f"Classify canary FAILED: only {nonempty}/{total} "
            f"({frac:.1%}) sub-windows returned events "
            f"(threshold: {threshold:.1%}). "
            f"Likely 32B prompt regression — aborting to save compute."
        )
        if self._cfg.canary_action == "fail":
            log.critical("dual_pass.canary_failed",
                         nonempty=nonempty, total=total,
                         fraction=f"{frac:.1%}")
            raise CanaryFailure(msg)
        else:
            log.critical("dual_pass.canary_warning",
                         nonempty=nonempty, total=total,
                         fraction=f"{frac:.1%}", msg=msg)

    def _save_observation_diagnostics(
        self, candidates: list[CandidateWindow],
        observations: dict[int, str],
    ) -> None:
        """Save 8B observation texts for post-run debugging."""
        diag_dir = self._working_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        with open(diag_dir / "observations.jsonl", "w") as f:
            for idx, window in enumerate(candidates):
                obs = observations.get(idx, "")
                json.dump({
                    "window_idx": idx,
                    "start_sec": window.start_sec,
                    "end_sec": window.end_sec,
                    "triage_labels": list(set(window.labels)),
                    "observation": obs,
                }, f)
                f.write("\n")

    def _save_triage_diagnostics(
        self, flags: list, candidates: list[CandidateWindow]
    ) -> None:
        """Save triage scan results for analysis."""
        diag_dir = self._working_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        # Flags
        with open(diag_dir / "triage_flags.jsonl", "w") as f:
            for flag in flags:
                json.dump({
                    "center_sec": flag.center_sec,
                    "label": flag.label,
                    "ball_zone": flag.ball_zone,
                    "window_start": flag.window_start,
                    "window_end": flag.window_end,
                }, f)
                f.write("\n")

        # Candidate windows
        with open(diag_dir / "triage_candidates.jsonl", "w") as f:
            for w in candidates:
                json.dump({
                    "start_sec": w.start_sec,
                    "end_sec": w.end_sec,
                    "duration_sec": w.duration_sec,
                    "labels": w.labels,
                    "n_flags": len(w.flags),
                }, f)
                f.write("\n")

    def _save_classify_diagnostics(
        self, candidates: list[CandidateWindow], events: list[Event]
    ) -> None:
        """Save 32B classification results for analysis."""
        diag_dir = self._working_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        with open(diag_dir / "dual_pass_events.jsonl", "w") as f:
            for event in events:
                json.dump({
                    "event_type": event.event_type.value,
                    "start_sec": event.timestamp_start,
                    "end_sec": event.timestamp_end,
                    "confidence": event.confidence,
                    "reasoning": event.metadata.get("vlm_reasoning", ""),
                    "triage_labels": event.metadata.get("triage_labels", []),
                }, f)
                f.write("\n")
