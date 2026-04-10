"""
Dual-pass VLM detector: 8B triage → model swap → 32B classification.

Phase 1: Fast 8B model scans entire game with multi-frame sliding windows.
          Produces candidate time windows where events might be occurring.
Phase 2: Model swap — stop 8B, start 32B (requires both GPUs for TP=2).
Phase 3: 32B model classifies each candidate window with dense frame sampling.
          Produces precise event boundaries and types.

This replaces the audio-first candidate generation with visual-first triage,
solving the core recall problem (audio misses ~80% of set pieces).
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

# 32B classification prompt — sent with dense frames from a candidate window
_CLASSIFY_PROMPT = """\
You are analyzing a soccer match from a sideline camera. \
Here are {n_frames} frames from a {duration:.0f}-second window \
where our triage model flagged potential events: {triage_labels}.

The triage system has already filtered for likely events, so this window \
likely contains at least one event. Identify every event you can see in \
these frames — typical windows contain 1-3 events. Be specific about what \
you observe. You may return an empty list [] if the frames clearly show \
nothing but routine midfield play with no event near either goal, but do \
not default to empty — most flagged windows do contain a real event.

Valid event types (choose ONLY from this list):
{valid_types_block}

GOALKEEPER GUIDELINES — distinguishing saves from restarts:
- catch: GK ends up HOLDING the ball (ball is in hands/arms and stays there \
for 2+ frames, OR the ball is no longer visible because it is clutched to \
the GK body, OR play clearly stops for the GK to distribute). Catches are \
extremely common — roughly half of all GK events in this game are catches. \
A catch is valid whether or not a shot is clearly visible in the sampled \
frames: if the GK ends up holding the ball in play, it is a catch.
- SHOT-THEN-CATCH IS ONE EVENT — emit catch, NOT shot_on_target. \
If you see a shot and then the GK holds the ball in the same window, the \
event is catch. The shot_on_target + catch pair is wrong — pick catch.
- shot_stop_diving: GK contacts the ball but it CLEARLY rebounds away, \
continues in play, or goes out for a corner. A visible rebound must be \
present. If the ball is no longer in play (GK holding), it is a catch.
- punch: GK clearly strikes the ball with a FIST (visible punch motion).
- Split rule: ball held after GK contact → catch. Ball rebounds visibly → \
shot_stop_diving. Stationary ball on the ground in the 6-yard box with no \
GK contact yet → goal_kick.
- CATCH vs GOAL_KICK: catch means the ball is IN THE GK'S HANDS. goal_kick \
means the ball is STATIONARY ON THE GROUND and the GK is preparing to kick \
it. Ball location (hands vs ground) is the deciding factor.

SET-PIECE TIEBREAKER — when a stationary ball + kicker is visible, prefer \
the specific restart type over free_kick_shot:
- Player at the corner flag / inside the corner arc → corner_kick.
- Player near the touchline holding/gathering the ball → throw_in.
- GK or defender in their own 6-yard box or penalty area → goal_kick.
- Ball on the center circle with both teams in their own halves and no \
defensive wall → kickoff.
- Ball on the penalty spot, all players outside the area → penalty.
- Only use free_kick_shot when a defensive wall of 3+ players is clearly \
forming in front of a stationary ball OUTSIDE the penalty area and the \
other five options above do not apply.

GOAL DETECTION — goals are RARE (typically 2-6 per game), so require \
strong evidence. Emit goal ONLY when at least ONE of these is present:
- (a) the ball is clearly visible INSIDE the goal frame / touching or \
crossing the goal line / bulging the net, OR
- (b) MULTIPLE teammates celebrating together for 2+ consecutive frames \
(arms raised AND running toward teammates AND/OR hugging). A single \
player with arms raised is NOT enough — require a celebration group.
A shot alone, even a powerful one, is NOT a goal. A shot followed by a GK \
catch or defenders clearing is NOT a goal. If you are not certain, prefer \
shot_on_target over goal. When you emit goal, DO NOT also emit \
shot_on_target for the same moment.

NOT a save event: GK walking, fielding a routine back-pass to feet, \
or kicking for distribution with no shot involved.

TIMESTAMP PRECISION — mark the ACTION moment:
- Restarts (goal_kick, corner_kick, throw_in, free_kick_shot): \
start_sec = the frame where the ball is KICKED or THROWN, not the setup.
- Shots: start_sec = the frame where the player's foot contacts the ball.
- Saves/catches: start_sec = the frame where the GK makes contact with the ball.
- end_sec should be 3-5 seconds after start_sec (the immediate aftermath).

Respond with ONLY a JSON array listing every event you observe:
[{{"event_type": "catch", "start_sec": 125.0, "end_sec": 130.0, \
"confidence": 0.85, "reasoning": "GK dives right and secures ball at 127s"}}]
"""

# Full event type catalog — all types offered for every window regardless of
# triage label (Run #4 showed has_set_piece gating missed corner_kick/throw_in).
_ALL_TYPES = [
    ("shot_on_target", "Player shoots toward goal — any clear shot attempt at the goal frame, on or off target. NOT a corner_kick or free_kick_shot — shots ONLY apply to open-play shots, not set-piece restarts."),
    ("goal", "Ball crossing the goal line into the net, players celebrating with arms raised, running to teammates"),
    ("catch", "GK holds/secures the ball — ball in hands/arms for 2+ frames or disappears into GK body"),
    ("shot_stop_diving", "GK contacts ball but it CLEARLY rebounds away or continues in play"),
    ("punch", "GK strikes ball with a FIST — visible punch motion in crowded area"),
    ("corner_kick", "Player standing over a STATIONARY ball INSIDE the corner arc (the quarter-circle at a pitch corner). REQUIREMENT: the corner flag OR the quarter-circle arc MUST be visible in at least one frame of the window. If you only see players clustered in the penalty area around a crossed ball with NO corner flag/arc in frame, it is NOT a corner_kick — it is likely goal_kick aftermath, shot_on_target, or open-play cross. Do not guess corner_kick from context alone; require the visual cue."),
    ("goal_kick", "GK or defender standing near a STATIONARY ball ON THE GROUND inside their own 6-yard box or penalty area, preparing to kick it long. Typical cue: the opposing team has retreated past the halfway line. OVERRIDE: whenever you see a GK or defender in or near their own 6-yard box with a stationary ball on the ground AND opponents retreated past midfield, classify as goal_kick — this takes priority over corner_kick (corner_kick requires a visible corner flag/arc in the attacking corner, which a goal_kick never has), throw_in (which requires a touchline), free_kick_shot (which requires a defensive wall), and shot_on_target (which requires a visible kicking motion at goal). goal_kicks are among the most common restarts — expect 15-25 per game."),
    ("free_kick_shot", "Stationary ball outside the penalty area with a defensive wall of 3+ players forming; the free-kick taker is preparing to shoot directly at goal. NOT a corner_kick (corner is inside the corner arc) and NOT a goal_kick (that is the GK in the 6-yard box)."),
    ("throw_in", "Player at the sideline/touchline holding the ball with BOTH HANDS above or behind the head, about to throw it back in. Also count the setup frames where a player is standing at the sideline gathering a ball. If ANY player is near the sideline with a ball and the throw motion is visible or imminent, it is a throw_in."),
    ("penalty", "All players outside the penalty area except the kicker and GK; ball on the penalty spot, GK on the goal line"),
    ("kickoff", "Ball STATIONARY on the center circle at the halfway line, players from BOTH teams standing in their own halves, one player about to tap the ball forward. Typical at match start, after a goal, or at the start of the second half."),
]


def _build_valid_types_block(triage_labels: list[str]) -> str:
    """Build the valid event types section — always offer the full catalog."""
    lines = [f"- {name}: {desc}" for name, desc in _ALL_TYPES]
    return "\n".join(lines)


@dataclass
class DualPassConfig:
    """Configuration for the dual-pass detector."""
    # vLLM server
    vllm_url: str = "http://10.10.2.222:8000"

    # 8B triage model
    tier1_model_name: str = "qwen3-vl-8b"
    tier1_model_path: str = "Qwen/Qwen3-VL-8B-Instruct"

    # 32B classification model
    tier2_model_name: str = "qwen3-vl-32b-fp8"
    tier2_model_path: str = "Qwen/Qwen3-VL-32B-Instruct-FP8"

    # Triage scanner settings
    frame_width: int = 960
    frames_per_window: int = 7
    window_span_sec: float = 10.0
    step_sec: float = 4.0  # Tighter step = more overlap = fewer boundary misses

    # Candidate merging (tight — ball_zone + gap splitting)
    merge_gap_sec: float = 4.0
    merge_pad_sec: float = 6.0  # Run #5: extra post-event context for goal celebrations
    max_window_sec: float = 60.0

    # 32B classification
    classify_max_frames: int = 45  # Max frames per 32B call
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
        """Run the full dual-pass detection pipeline.

        Progress allocation:
            0.00 - 0.02: Model swap to 8B
            0.02 - 0.45: 8B triage scan
            0.45 - 0.50: Model swap to 32B
            0.50 - 0.95: 32B classification
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
            _progress(0.02 + pct * 0.43)

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

        _progress(0.45)

        # ── Phase 3: Model swap to 32B ─────────────────────────────────
        log.info("dual_pass.phase3_swap_to_32b", job_id=self._job_id)
        self._swap_model(
            model_name=self._cfg.tier2_model_name,
            model_path=self._cfg.tier2_model_path,
            tier="tier2",
        )

        # ── Canary 3: Model swap health ping ──────────────────────────
        if self._cfg.canary_enabled and self._cfg.swap_canary_enabled:
            self._check_swap_canary()

        _progress(0.50)

        # ── Phase 4: 32B classification of each candidate window ───────
        log.info("dual_pass.phase4_classify",
                 job_id=self._job_id, windows=len(candidates))

        all_events: list[Event] = []
        self._canary_sub_total = 0
        self._canary_sub_nonempty = 0
        self._canary_checked = False
        self._classify_latencies: list[float] = []
        self._diversity_checked = False

        for idx, window in enumerate(candidates):
            events = self._classify_window(window)
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

            _progress(0.50 + 0.45 * (idx + 1) / len(candidates))

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

    def _classify_window(self, window: CandidateWindow) -> list[Event]:
        """Classify a candidate window using sub-window chunking.

        Windows >sub_window_sec are split into overlapping sub-windows,
        each classified independently. Results are deduplicated.

        Frame density adapts to sub-window size:
          - ≤30s: 1fps (every frame)
          - >30s: 1 frame per 2s
        """
        sub_size = self._cfg.sub_window_sec
        overlap = self._cfg.sub_window_overlap_sec

        # Build sub-window boundaries
        if window.duration_sec <= sub_size + overlap:
            # Small enough — single call
            sub_windows = [(window.start_sec, window.end_sec)]
        else:
            step = sub_size - overlap
            sub_windows = []
            t = window.start_sec
            while t < window.end_sec:
                end = min(t + sub_size, window.end_sec)
                sub_windows.append((t, end))
                t += step
                if end >= window.end_sec:
                    break

        log.info("dual_pass.classify_sub_windows",
                 window_start=window.start_sec,
                 window_dur=window.duration_sec,
                 n_sub=len(sub_windows))

        all_events: list[Event] = []
        for sw_start, sw_end in sub_windows:
            sw_dur = sw_end - sw_start
            # Adaptive frame interval
            interval = 1.0 if sw_dur <= 30 else 2.0

            events = self._classify_sub_window(
                start_sec=sw_start,
                end_sec=sw_end,
                interval_sec=interval,
                triage_labels=list(set(window.labels)),
            )
            all_events.extend(events)

        return all_events

    def _classify_sub_window(
        self,
        start_sec: float,
        end_sec: float,
        interval_sec: float,
        triage_labels: list[str],
    ) -> list[Event]:
        """Send a single sub-window to the 32B model for classification."""
        import httpx
        import time

        duration = end_sec - start_sec
        frames = self._sampler.sample_range(
            center_sec=(start_sec + end_sec) / 2,
            window_sec=duration / 2,
            interval_sec=interval_sec,
            duration_sec=self._video_duration,
        )

        if not frames:
            return []

        # Limit frame count
        if len(frames) > self._cfg.classify_max_frames:
            step = len(frames) / self._cfg.classify_max_frames
            indices = [int(i * step) for i in range(self._cfg.classify_max_frames)]
            frames = [frames[i] for i in indices]

        valid_types_block = _build_valid_types_block(triage_labels)
        prompt = _CLASSIFY_PROMPT.format(
            n_frames=len(frames),
            duration=duration,
            triage_labels=", ".join(triage_labels),
            valid_types_block=valid_types_block,
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
            "model": self._cfg.tier2_model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1000,
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
                return []

            text = r.json()["choices"][0]["message"]["content"].strip()
            log.info("dual_pass.raw_32b_response",
                     sub_start=start_sec, text=text[:300])
            # Create a temporary CandidateWindow for parsing
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
                if conf < 0.4:
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
        self, events: list[Event], overlap_threshold: float = 0.5
    ) -> list[Event]:
        """Remove duplicate events that overlap significantly."""
        if len(events) <= 1:
            return events

        # Sort by start time
        sorted_events = sorted(events, key=lambda e: e.timestamp_start)
        kept: list[Event] = [sorted_events[0]]

        for event in sorted_events[1:]:
            prev = kept[-1]
            # Check overlap with previous event of same type
            if event.event_type == prev.event_type:
                overlap_start = max(event.timestamp_start, prev.timestamp_start)
                overlap_end = min(event.timestamp_end, prev.timestamp_end)
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    shorter = min(event.duration_sec, prev.duration_sec)
                    if shorter > 0 and overlap / shorter >= overlap_threshold:
                        # Keep the one with higher confidence
                        if event.confidence > prev.confidence:
                            kept[-1] = event
                        continue
            kept.append(event)

        if len(kept) < len(events):
            log.info("dual_pass.deduplicated",
                     before=len(events), after=len(kept))
        return kept

    def _post_filter_events(self, events: list[Event]) -> list[Event]:
        """Apply contextual post-filters.

        - shot_stop_standing → catch promotion: standing GK blocks are often
          catches; reclassify since GT doesn't have shot_stop_standing.
        """
        kept: list[Event] = []
        dropped = 0
        for event in events:
            # Run #9: kickoff is offered to the 32B as a magnet so it stops
            # mislabeling mid-pitch stationary-ball scenes as free_kick_shot,
            # but GT does not score kickoffs — drop them after classification.
            if event.event_type == EventType.KICKOFF:
                dropped += 1
                continue

            # Reclassify shot_stop_standing → catch
            # (GT never has shot_stop_standing; these are usually catches)
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

            kept.append(event)

        if dropped:
            log.info("dual_pass.post_filtered",
                     dropped=dropped, kept=len(kept))
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
