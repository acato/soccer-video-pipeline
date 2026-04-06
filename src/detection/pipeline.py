"""
Detection pipeline orchestrator — motion-first architecture.

Phase ordering:
  1. Motion scan — dense frame-differencing to find activity spikes
  2. Audio detection — optional booster for co-located motion candidates
  3. VLM classification — single-pass classify via Qwen3-VL / Claude
  3a.5. Save reclassification — goal_kicks with save evidence → saves
  3a.6. Shot reclassification — rejected verdicts with shot evidence → shots
  3b. Goal inference — kickoff rescan to upgrade shots, dedup nearby goals
  3c. Set-piece inference — corner/goal-kick rescan after shots
  3d. Corner scan — independent corner detection at uncovered candidates
  3e. Reverse restart inference — work backwards from restarts to find shots
  3f. Shot scan — binary VLM re-probe of remaining rejected candidates
  3g. Catch scan — structural catch inference + VLM probe for GK holding ball

Produces Event objects compatible with the existing segmentation/assembly
pipeline.

Usage::

    pipeline = DetectionPipeline(
        source_file="/path/to/match.mp4",
        video_duration=5400.0,
        fps=30.0,
        job_id="abc-123",
    )
    events = pipeline.run()
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable, Optional

import structlog

from src.detection.audio_detector import AudioCandidate, AudioCueType, AudioDetector
from src.detection.models import Event, EventType
from src.detection.visual_candidate import (
    CandidateSource,
    EventCandidate,
    VisualCandidateGenerator,
    VisualContext,
)
from src.detection.vlm_verifier import (
    VLMVerdict,
    VLMVerifier,
    VerificationResult,
)
from src.ingestion.models import MatchConfig

log = structlog.get_logger(__name__)


class DetectionPipeline:
    """Orchestrate motion → audio-boost → VLM → Event production."""

    def __init__(
        self,
        source_file: str | Path,
        video_duration: float,
        fps: float,
        job_id: str,
        *,
        match_config: Optional[MatchConfig] = None,
        game_start_sec: float = 0.0,
        # Audio config
        audio_enabled: bool = True,
        bandpass_low_hz: int = 2000,
        bandpass_high_hz: int = 4000,
        min_whistle_sec: float = 0.2,
        surge_stddev_threshold: float = 2.0,
        # Visual config
        yolo_model_path: Optional[str] = None,
        use_gpu: bool = False,
        yolo_inference_size: int = 640,
        scan_interval_sec: float = 15.0,
        # VLM config
        vllm_url: Optional[str] = None,
        vllm_model: str = "soccer-event-classifier",
        anthropic_api_key: Optional[str] = None,
        anthropic_model: str = "claude-sonnet-4-20250514",
        vlm_min_confidence: float = 0.5,
        vlm_enabled: bool = True,
        # Two-tier VLM config
        tiered_vlm: bool = False,
        tier1_model: str = "",
        tier1_model_path: str = "",
        tier1_lora_path: str = "",
        tier2_model: str = "",
        tier2_model_path: str = "",
        tier1_min_confidence: float = 0.6,
        tier1_broken_threshold: float = 3.0,
        tier2_spot_check_rate: float = 0.10,
        tier2_escalation_cap: float = 0.50,
        model_swap_script: str = "",
        model_swap_timeout_sec: int = 120,
        # General
        working_dir: Optional[str | Path] = None,
        min_event_confidence: float = 0.5,
    ):
        self._source = Path(source_file)
        self._duration = video_duration
        self._fps = fps
        self._job_id = job_id
        self._match_config = match_config
        self._game_start = game_start_sec
        self._min_conf = min_event_confidence

        if working_dir:
            self._work = Path(working_dir)
        else:
            self._work = Path("/tmp/soccer-pipeline")

        # Phase 1: Motion (primary)
        self._visual_gen = VisualCandidateGenerator(
            source_file,
            yolo_model_path=yolo_model_path,
            use_gpu=use_gpu,
            inference_size=yolo_inference_size,
            fps=fps,
            working_dir=self._work,
        )

        # Phase 2: Audio (supplementary booster)
        self._audio_enabled = audio_enabled
        self._audio_detector = AudioDetector(
            source_file,
            bandpass_low_hz=bandpass_low_hz,
            bandpass_high_hz=bandpass_high_hz,
            min_whistle_sec=min_whistle_sec,
            surge_stddev_threshold=surge_stddev_threshold,
            game_start_sec=game_start_sec,
            working_dir=self._work,
        )

        # Phase 3: VLM
        self._vlm_enabled = vlm_enabled
        self._vlm_verifier = VLMVerifier(
            vllm_url=vllm_url,
            vllm_model=vllm_model,
            anthropic_api_key=anthropic_api_key,
            anthropic_model=anthropic_model,
            min_confidence=vlm_min_confidence,
            source_file=source_file,
            working_dir=self._work,
        )

        # Phase 3 — Two-tier VLM (optional)
        self._tiered_vlm = tiered_vlm and vlm_enabled
        self._model_manager = None
        self._tier_router = None

        if self._tiered_vlm and vllm_url:
            from src.detection.model_manager import ModelManager, ModelConfig
            from src.detection.tier_router import TierRouter

            tier1_cfg = ModelConfig(
                name=tier1_model,
                tier="tier1",
                model_path=tier1_model_path,
                lora_path=tier1_lora_path,
            ) if tier1_model else None

            tier2_cfg = ModelConfig(
                name=tier2_model,
                tier="tier2",
                model_path=tier2_model_path,
            ) if tier2_model else None

            self._model_manager = ModelManager(
                vllm_url=vllm_url,
                tier1=tier1_cfg,
                tier2=tier2_cfg,
                swap_script=model_swap_script,
                swap_timeout_sec=model_swap_timeout_sec,
            )
            self._tier_router = TierRouter(
                min_confidence=tier1_min_confidence,
                broken_threshold=tier1_broken_threshold,
                spot_check_rate=tier2_spot_check_rate,
                escalation_cap=tier2_escalation_cap,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> list[Event]:
        """Run the full detection pipeline.

        Args:
            progress_callback: Called with 0-1 progress fraction.
            cancel_check: Called between phases; return True to abort.

        Returns:
            List of Event objects ready for segmentation.
        """
        events: list[Event] = []

        # ── Phase 1: Motion scan (0% → 40%) ───────────────────────────
        log.info("pipeline.phase1_motion", source=str(self._source))

        def motion_progress(p: float):
            if progress_callback:
                progress_callback(p * 0.40)

        candidates = self._visual_gen.motion_scan(
            video_duration=self._duration,
            progress_callback=motion_progress,
        )
        log.info("pipeline.motion_done", candidates=len(candidates))

        # Diagnostic: dump motion candidates
        self._dump_diagnostics("motion_candidates", [
            {"timestamp": c.timestamp,
             "confidence": round(c.confidence, 3),
             "motion": round(c.context.motion_magnitude, 5),
             "spike_dur": round(c.context.spike_duration_sec, 2),
             "mm_ss": f"{int(c.timestamp//60):02d}:{c.timestamp%60:05.2f}"}
            for c in candidates
        ])

        if progress_callback:
            progress_callback(0.40)

        if cancel_check and cancel_check():
            return events

        # ── Phase 2: Audio boost (40% → 55%) ──────────────────────────
        audio_candidates = []
        if self._audio_enabled:
            log.info("pipeline.phase2_audio", source=str(self._source))

            def audio_progress(p: float):
                if progress_callback:
                    progress_callback(0.40 + p * 0.15)

            audio_candidates = self._audio_detector.detect(
                progress_callback=audio_progress,
            )
            log.info("pipeline.audio_done", candidates=len(audio_candidates))

            self._dump_diagnostics("audio_candidates", [
                {"timestamp": ac.timestamp, "cue_type": ac.cue_type.value,
                 "duration": ac.duration_sec, "amplitude": ac.amplitude,
                 "freq_hz": ac.frequency_hz,
                 "mm_ss": f"{int(ac.timestamp//60):02d}:{ac.timestamp%60:05.2f}"}
                for ac in audio_candidates
            ])

            # Boost motion candidates with co-located audio cues
            candidates = self._visual_gen.boost_with_audio(
                candidates, audio_candidates,
            )
        else:
            log.info("pipeline.audio_disabled")

        if progress_callback:
            progress_callback(0.55)

        # Diagnostic: dump final candidates (after audio boost)
        self._dump_diagnostics("final_candidates", [
            {"timestamp": c.timestamp, "source": c.source.value,
             "confidence": round(c.confidence, 3),
             "motion": round(c.context.motion_magnitude, 5),
             "spike_dur": round(c.context.spike_duration_sec, 2),
             "audio_boost": c.context.audio_boost,
             "audio_cue": c.audio_cue.cue_type.value if c.audio_cue else None,
             "mm_ss": f"{int(c.timestamp//60):02d}:{c.timestamp%60:05.2f}"}
            for c in candidates
        ])

        if cancel_check and cancel_check():
            return events

        # ── Phase 2b: Match structure + filtering ─────────────────────
        match_struct = self._detect_match_structure(candidates, audio_candidates)
        candidates = self._filter_by_match_structure(candidates, match_struct)

        # ── Phase 2c: Audio gap fill — promote orphan audio cues ──────
        candidates = self._audio_gap_fill(candidates, audio_candidates, match_struct)

        # ── Phase 2d: Spot-check probes in temporal gaps ──────────────
        candidates = self._spot_check_probes(candidates, match_struct)

        self._dump_diagnostics("filtered_candidates", [
            {"timestamp": c.timestamp, "source": c.source.value,
             "confidence": round(c.confidence, 3),
             "mm_ss": f"{int(c.timestamp//60):02d}:{c.timestamp%60:05.2f}"}
            for c in candidates
        ])

        # ── Phase 3: VLM classification (55% → 90%) ───────────────────
        # Keep full candidate list for goal inference rescan
        all_motion_candidates = list(candidates)

        # Cap candidates sent to VLM — distribute evenly across time
        _VLM_MAX_CANDIDATES = 120
        if len(candidates) > _VLM_MAX_CANDIDATES:
            candidates = self._time_distributed_sample(
                candidates, _VLM_MAX_CANDIDATES,
            )

        if self._vlm_enabled and candidates:
            vlm_available = self._vlm_verifier.is_available()
            if vlm_available:
                log.info("pipeline.phase3_vlm",
                         candidates=len(candidates),
                         tiered=self._tiered_vlm)

                def vlm_progress(p: float):
                    if progress_callback:
                        progress_callback(0.55 + p * 0.30)

                if self._tiered_vlm and self._model_manager and self._tier_router:
                    verdicts = self._vlm_verifier.verify_tiered(
                        candidates,
                        match_config=self._match_config,
                        progress_callback=vlm_progress,
                        model_manager=self._model_manager,
                        tier_router=self._tier_router,
                    )
                else:
                    verdicts = self._vlm_verifier.verify(
                        candidates,
                        match_config=self._match_config,
                        progress_callback=vlm_progress,
                    )

                # Diagnostic: dump VLM verdicts
                self._dump_diagnostics("vlm_verdicts", [
                    {"timestamp": v.candidate.timestamp,
                     "result": v.result.value,
                     "event_type": v.event_type.value if v.event_type else None,
                     "confidence": round(v.confidence, 3),
                     "reasoning": v.reasoning,
                     "model": v.model_used,
                     "mm_ss": f"{int(v.candidate.timestamp//60):02d}:{v.candidate.timestamp%60:05.2f}"}
                    for v in verdicts
                ])

                # Phase 3a.5: Save reclassification — goal_kicks
                # with save evidence → SHOT_STOP_DIVING
                verdicts = self._save_reclassification(verdicts)

                # Phase 3a.6: Shot reclassification — rejected
                # verdicts whose reasoning describes a shot
                verdicts = self._shot_reclassification(verdicts)

                # Phase 3b: Goal inference — kickoff rescan (82% → 87%)
                if progress_callback:
                    progress_callback(0.82)
                verdicts = self._goal_inference(
                    verdicts, all_motion_candidates,
                )

                # Phase 3c: Set-piece inference — corner/goal-kick
                # rescan after shots (87% → 90%)
                if progress_callback:
                    progress_callback(0.87)
                verdicts = self._set_piece_inference(
                    verdicts, all_motion_candidates,
                )

                # Phase 3c.5: Reclassify shots followed by goal kicks
                # as SHOT_OFF_TARGET (ball went out without GK save)
                verdicts = self._shot_restart_reclassify(verdicts)

                # Phase 3d: Independent corner scan (90% → 92%)
                if progress_callback:
                    progress_callback(0.90)
                verdicts = self._corner_scan(
                    verdicts, all_motion_candidates,
                )

                # Phase 3e: Reverse restart inference — work
                # backwards from restarts to find missed shots
                if progress_callback:
                    progress_callback(0.92)
                verdicts = self._reverse_restart_inference(verdicts)

                # Phase 3f: Binary shot scan — re-probe rejected
                # candidates with a focused "was a shot taken?" prompt
                if progress_callback:
                    progress_callback(0.93)
                verdicts = self._shot_scan(verdicts)

                # Phase 3g: Catch scan — structural inference +
                # VLM probe for GK holding ball after shots
                if progress_callback:
                    progress_callback(0.94)
                verdicts = self._catch_scan(verdicts)

                # Phase 3h: Claude goal verification — re-check
                # save/free_kick events with kickoff evidence
                if progress_callback:
                    progress_callback(0.95)
                verdicts = self._claude_goal_verification(
                    verdicts, all_motion_candidates,
                )

                events = self._verdicts_to_events(verdicts)
            else:
                log.warning("pipeline.vlm_unavailable",
                            msg="passing candidates through unverified")
                events = self._candidates_to_events(candidates)
        elif candidates:
            log.info("pipeline.vlm_disabled",
                     msg="converting candidates directly to events")
            events = self._candidates_to_events(candidates)

        if progress_callback:
            progress_callback(0.95)

        # ── Post-processing (95% → 100%) ──────────────────────────────
        events = self._deduplicate(events)
        events.sort(key=lambda e: e.timestamp_start)

        log.info("pipeline.complete",
                 total_events=len(events),
                 event_types=[e.event_type.value for e in events])

        if progress_callback:
            progress_callback(1.0)

        return events

    # ------------------------------------------------------------------
    # Internal — save reclassification (Phase 3a.5)
    # ------------------------------------------------------------------

    # Keywords in VLM reasoning that indicate a save occurred
    _SAVE_KEYWORDS = [
        "save", "saved", "stopped", "blocked by the keeper",
        "blocked by the goalkeeper", "parr", "catch", "punched away",
        "tipped over", "tipped away", "pushed away", "kept out",
        "denied", "goalkeeper.*block", "keeper.*block",
    ]

    def _save_reclassification(
        self, verdicts: list[VLMVerdict],
    ) -> list[VLMVerdict]:
        """Reclassify goal_kick verdicts as saves when reasoning indicates one.

        The VLM often sees shot → save → goal kick restart and classifies the
        whole sequence as 'goal_kick' because the restart is the most visible
        action.  When the reasoning text mentions a save, the *real* event is
        the save, not the goal kick.
        """
        import re

        modified = list(verdicts)
        reclassified = 0

        for i, v in enumerate(modified):
            if v.result != VerificationResult.CONFIRMED:
                continue
            if v.event_type != EventType.GOAL_KICK:
                continue

            reasoning_lower = v.reasoning.lower()
            has_save = any(
                re.search(kw, reasoning_lower) for kw in self._SAVE_KEYWORDS
            )
            if not has_save:
                continue

            modified[i] = VLMVerdict(
                candidate=v.candidate,
                result=VerificationResult.CONFIRMED,
                event_type=EventType.SHOT_STOP_DIVING,
                confidence=v.confidence,
                reasoning=v.reasoning,
                model_used=v.model_used,
            )
            reclassified += 1
            t = v.candidate.timestamp
            log.info("save_reclass.upgrade",
                     time=f"{int(t//60):02d}:{t%60:05.2f}",
                     reasoning_snippet=v.reasoning[:80])

        log.info("save_reclass.complete",
                 total_goal_kicks=sum(
                     1 for v in verdicts
                     if v.event_type == EventType.GOAL_KICK),
                 reclassified=reclassified)

        return modified

    # ------------------------------------------------------------------
    # Internal — shot reclassification (Phase 3a.6)
    # ------------------------------------------------------------------

    # Positive evidence: VLM *describes* a shot actually happening.
    # These must NOT match negated phrases like "no shot on goal".
    _SHOT_POSITIVE_KW = [
        r"(?:a |the )?player.{0,20}(shot|shoots|struck|fired|kicked).{0,15}(toward|towards|at|on).{0,10}goal",
        r"(?:a |the )?shot (?:is |was )?taken",
        r"(?:a |the )?shot (?:is |was )?fired",
        r"took a shot",
        r"takes a shot",
        r"hit the (cross)?bar",
        r"(?:ball |shot )(?:went |goes |travel\w+ )(?:wide|over)",
        r"over the bar",
        r"wide of (the )?goal",
        r"off.{0,5}target",
        r"blocked.{0,10}(?:by a |by the )?(?:defender|field player)",
        r"deflect\w* (?:off|by|away)",
        r"struck.{0,10}(?:toward|at) goal",
    ]

    def _shot_reclassification(
        self, verdicts: list[VLMVerdict],
    ) -> list[VLMVerdict]:
        """Recover shots from rejected verdicts whose reasoning describes one.

        The VLM often *sees* a shot (mentions it in reasoning) but classifies
        the sequence as "none" because no clear restart follows.  If the
        reasoning positively describes a shot actually happening (not just
        "no shot on goal"), reclassify as SHOT_ON_TARGET or SHOT_OFF_TARGET.
        """
        import re

        modified = list(verdicts)
        reclassified = 0

        for i, v in enumerate(modified):
            if v.result != VerificationResult.REJECTED:
                continue

            reasoning = v.reasoning.lower()

            has_positive = any(
                re.search(kw, reasoning) for kw in self._SHOT_POSITIVE_KW
            )
            if not has_positive:
                continue

            # Determine on/off target from wording
            off_target_kw = [
                r"wide", r"over the bar", r"off.{0,5}target",
                r"missed", r"blocked", r"deflect",
            ]
            is_off = any(re.search(kw, reasoning) for kw in off_target_kw)
            event_type = (EventType.SHOT_OFF_TARGET if is_off
                          else EventType.SHOT_ON_TARGET)

            modified[i] = VLMVerdict(
                candidate=v.candidate,
                result=VerificationResult.CONFIRMED,
                event_type=event_type,
                confidence=v.confidence * 0.7,   # Penalise reclassified
                reasoning=f"SHOT_RECLASS: {v.reasoning}",
                model_used=v.model_used,
            )
            reclassified += 1
            t = v.candidate.timestamp
            log.info("shot_reclass.upgrade",
                     time=f"{int(t//60):02d}:{t%60:05.2f}",
                     event_type=event_type.value,
                     reasoning_snippet=v.reasoning[:80])

        log.info("shot_reclass.complete",
                 total_rejected=sum(
                     1 for v in verdicts
                     if v.result == VerificationResult.REJECTED),
                 reclassified=reclassified)

        return modified

    # ------------------------------------------------------------------
    # Internal — goal inference (Phase 3b)
    # ------------------------------------------------------------------

    _KICKOFF_RESCAN_MIN_GAP = 20.0    # Kickoff must be ≥20s after shot
    _KICKOFF_RESCAN_MAX_GAP = 90.0    # ... and ≤90s after shot
    _KICKOFF_DIRECT_PROBES = [30, 45, 60, 75, 90]  # Denser probe offsets
    _GOAL_DEDUP_WINDOW = 240.0        # Merge goals within 4 min
    _GOAL_CONFIRM_ENABLED = False     # VLM celebration probe — disabled: VLM cannot see celebrations from sideline camera at 50m

    def _goal_inference(
        self,
        verdicts: list[VLMVerdict],
        all_candidates: list[EventCandidate],
    ) -> list[VLMVerdict]:
        """Post-VLM goal inference using temporal kickoff patterns.

        1. For each confirmed shot/save AND each VLM-classified goal, rescan
           for a kickoff 20-90s later using dense probe offsets.
        2. Exclude pre-game shots: any shot BEFORE the first confirmed kickoff
           cannot be upgraded (opening kickoff ≠ post-goal restart).
        3. Upgrade shots/saves with confirmed kickoff → goal, with optional
           VLM celebration probe as secondary confirmation.
        4. Downgrade VLM goals WITHOUT confirmed kickoff → shot.
        5. Deduplicate goals within 4 min of each other.
        """
        confirmed = [v for v in verdicts
                     if v.result == VerificationResult.CONFIRMED]

        shot_types = {EventType.SHOT_ON_TARGET, EventType.SHOT_STOP_DIVING}
        shots = [v for v in confirmed if v.event_type in shot_types]
        vlm_goals = [v for v in confirmed if v.event_type == EventType.GOAL]
        kickoffs = [v for v in confirmed if v.event_type == EventType.KICKOFF]

        vlm_times = {v.candidate.timestamp for v in verdicts}

        # ── Step 0: Identify first kickoff (match start) ─────────────
        # Any shot BEFORE the first confirmed kickoff is pre-game and must
        # not be matched to the opening kickoff.
        first_kickoff_t = None
        if kickoffs:
            first_kickoff_t = min(ko.candidate.timestamp for ko in kickoffs)
            log.info("goal_inference.first_kickoff",
                     time=f"{int(first_kickoff_t//60):02d}:"
                          f"{first_kickoff_t%60:05.2f}")

        # ── Step 1: Find candidates to rescan for kickoffs ────────────
        events_needing_kickoff = shots + vlm_goals
        rescan_map: dict[float, VLMVerdict] = {}
        rescan_candidates: list[EventCandidate] = []
        events_without_rescan: list[VLMVerdict] = []
        skipped_pregame = 0

        for ev in events_needing_kickoff:
            t = ev.candidate.timestamp
            # Skip pre-game events — they cannot produce goals
            if first_kickoff_t is not None and t < first_kickoff_t:
                skipped_pregame += 1
                continue

            # Already have a kickoff after this event (excluding the
            # opening kickoff which is not a post-goal restart)?
            has_ko = any(
                self._KICKOFF_RESCAN_MIN_GAP
                < ko.candidate.timestamp - t
                < self._KICKOFF_RESCAN_MAX_GAP
                and ko.candidate.timestamp != first_kickoff_t
                for ko in kickoffs
            )
            if has_ko:
                continue

            # Find first unclassified motion candidate in the window
            found = False
            for cand in sorted(all_candidates, key=lambda c: c.timestamp):
                gap = cand.timestamp - t
                if gap < self._KICKOFF_RESCAN_MIN_GAP:
                    continue
                if gap > self._KICKOFF_RESCAN_MAX_GAP:
                    break
                if cand.timestamp not in vlm_times:
                    rescan_map[cand.timestamp] = ev
                    rescan_candidates.append(cand)
                    found = True
                    break

            if not found:
                events_without_rescan.append(ev)

        if skipped_pregame:
            log.info("goal_inference.pregame_excluded",
                     skipped=skipped_pregame)

        # For events with no motion candidate in the window, create
        # synthetic probe candidates at fixed offsets (direct extraction)
        for ev in events_without_rescan:
            t = ev.candidate.timestamp
            for offset in self._KICKOFF_DIRECT_PROBES:
                probe_t = t + offset
                if probe_t > self._duration:
                    continue
                probe = EventCandidate(
                    timestamp=probe_t,
                    source=CandidateSource.SPOT_CHECK,
                    confidence=0.5,
                    context=VisualContext(),
                    clip_start=max(0, probe_t - 5),
                    clip_end=min(self._duration, probe_t + 15),
                )
                rescan_map[probe_t] = ev
                rescan_candidates.append(probe)

        if events_without_rescan:
            log.info("goal_inference.direct_probes",
                     events=len(events_without_rescan),
                     probes=sum(1 for _ in events_without_rescan
                                for _ in self._KICKOFF_DIRECT_PROBES))

        # ── Step 2: Rescan candidates via focused kickoff prompt ──────
        new_kickoffs: list[VLMVerdict] = []
        if rescan_candidates:
            log.info("goal_inference.kickoff_rescan",
                     events=len(events_needing_kickoff),
                     rescanning=len(rescan_candidates))

            rescan_verdicts = self._vlm_verifier.verify_kickoff(
                rescan_candidates,
                match_config=self._match_config,
            )

            self._dump_diagnostics("kickoff_rescan", [
                {"timestamp": rv.candidate.timestamp,
                 "result": rv.result.value,
                 "event_type": rv.event_type.value if rv.event_type else None,
                 "confidence": round(rv.confidence, 3),
                 "reasoning": rv.reasoning,
                 "for_event_at": (rescan_map[rv.candidate.timestamp]
                                  .candidate.timestamp
                                  if rv.candidate.timestamp in rescan_map
                                  else None),
                 "mm_ss": f"{int(rv.candidate.timestamp//60):02d}:"
                          f"{rv.candidate.timestamp%60:05.2f}"}
                for rv in rescan_verdicts
            ])

            new_kickoffs = [
                rv for rv in rescan_verdicts
                if rv.result == VerificationResult.CONFIRMED
                and rv.event_type == EventType.KICKOFF
            ]
            log.info("goal_inference.kickoffs_found",
                     found=len(new_kickoffs))
        else:
            log.info("goal_inference.no_rescan_needed")

        all_kickoffs = kickoffs + new_kickoffs

        # Update first_kickoff_t to include rescan kickoffs — without this,
        # pre-game exclusion fails when no kickoffs are in the initial phase
        if all_kickoffs:
            earliest = min(ko.candidate.timestamp for ko in all_kickoffs)
            if first_kickoff_t is None or earliest < first_kickoff_t:
                first_kickoff_t = earliest
                log.info("goal_inference.first_kickoff_updated",
                         time=f"{int(first_kickoff_t//60):02d}:"
                              f"{first_kickoff_t%60:05.2f}",
                         source="rescan")

        # Build set of timestamps that have a confirmed kickoff after them
        # (excluding the opening kickoff — it's the match start, not a
        # post-goal restart)
        kickoff_confirmed: set[float] = set()
        for ev in events_needing_kickoff:
            t = ev.candidate.timestamp
            # Skip pre-game events
            if first_kickoff_t is not None and t < first_kickoff_t:
                continue
            if any(
                self._KICKOFF_RESCAN_MIN_GAP
                < ko.candidate.timestamp - t
                < self._KICKOFF_RESCAN_MAX_GAP
                and ko.candidate.timestamp != first_kickoff_t
                for ko in all_kickoffs
            ):
                kickoff_confirmed.add(t)

        # ── Step 3: Upgrade shots/saves with kickoff → goal ───────────
        modified = list(verdicts)
        # Build id→index map so we never need fragile .index() lookups
        _id_to_idx = {id(v): i for i, v in enumerate(modified)}
        upgrades = 0

        for shot in shots:
            t = shot.candidate.timestamp
            if t not in kickoff_confirmed:
                continue
            # Find the matching kickoff for logging
            best_ko = min(
                (ko for ko in all_kickoffs
                 if self._KICKOFF_RESCAN_MIN_GAP
                 < ko.candidate.timestamp - t
                 < self._KICKOFF_RESCAN_MAX_GAP
                 and ko.candidate.timestamp != first_kickoff_t),
                key=lambda ko: ko.candidate.timestamp - t,
            )
            gap = best_ko.candidate.timestamp - t
            idx = _id_to_idx.get(id(shot))
            if idx is None:
                continue
            modified[idx] = VLMVerdict(
                candidate=shot.candidate,
                result=VerificationResult.CONFIRMED,
                event_type=EventType.GOAL,
                confidence=min(shot.confidence + 0.1, 1.0),
                reasoning=(
                    f"INFERRED GOAL: {shot.event_type.value} "
                    f"followed by kickoff {gap:.0f}s later. "
                    f"Original: {shot.reasoning}"
                ),
                model_used=shot.model_used,
            )
            mm = int(t // 60)
            ss = t % 60
            log.info("goal_inference.upgrade",
                     time=f"{mm:02d}:{ss:05.2f}",
                     kickoff_gap=round(gap),
                     original=shot.event_type.value)
            upgrades += 1

        # ── Step 3b: VLM celebration probe ────────────────────────────
        # For each candidate goal, ask VLM "is there a celebration or
        # did the ball cross the line?" as secondary confirmation.
        # Goals that fail the probe are downgraded back to shot.
        if self._GOAL_CONFIRM_ENABLED:
            celebration_downgrades = 0
            candidate_goals = [
                (i, v) for i, v in enumerate(modified)
                if v.result == VerificationResult.CONFIRMED
                and v.event_type == EventType.GOAL
            ]
            if candidate_goals:
                goal_candidates = [
                    v.candidate for _, v in candidate_goals
                ]
                celebration_verdicts = (
                    self._vlm_verifier.verify_goal_celebration(
                        goal_candidates,
                        match_config=self._match_config,
                    )
                )
                self._dump_diagnostics("goal_celebration_probe", [
                    {"timestamp": cv.candidate.timestamp,
                     "result": cv.result.value,
                     "event_type": (cv.event_type.value
                                    if cv.event_type else None),
                     "confidence": round(cv.confidence, 3),
                     "reasoning": cv.reasoning,
                     "mm_ss": f"{int(cv.candidate.timestamp//60):02d}:"
                              f"{cv.candidate.timestamp%60:05.2f}"}
                    for cv in celebration_verdicts
                ])
                for (idx, orig_v), cel_v in zip(
                    candidate_goals, celebration_verdicts
                ):
                    if cel_v.result != VerificationResult.CONFIRMED:
                        # Celebration not confirmed — downgrade
                        modified[idx] = VLMVerdict(
                            candidate=orig_v.candidate,
                            result=VerificationResult.CONFIRMED,
                            event_type=EventType.SHOT_ON_TARGET,
                            confidence=orig_v.confidence * 0.7,
                            reasoning=(
                                f"GOAL_PROBE_REJECTED: no celebration"
                                f"/scoring confirmed by VLM. "
                                f"Original: {orig_v.reasoning[:150]}"
                            ),
                            model_used=orig_v.model_used,
                        )
                        celebration_downgrades += 1
                        ct = orig_v.candidate.timestamp
                        log.info(
                            "goal_inference.celebration_rejected",
                            time=f"{int(ct//60):02d}:{ct%60:05.2f}",
                            probe_reasoning=cel_v.reasoning[:100],
                        )
                    else:
                        ct = orig_v.candidate.timestamp
                        log.info(
                            "goal_inference.celebration_confirmed",
                            time=f"{int(ct//60):02d}:{ct%60:05.2f}",
                            probe_reasoning=cel_v.reasoning[:100],
                        )
                log.info("goal_inference.celebration_probe",
                         probed=len(candidate_goals),
                         rejected=celebration_downgrades)

        # ── Step 4: Downgrade VLM goals WITHOUT kickoff → shot ────────
        goal_downgrades = 0
        for goal in vlm_goals:
            t = goal.candidate.timestamp
            if t in kickoff_confirmed:
                continue
            idx = _id_to_idx.get(id(goal))
            if idx is None:
                continue
            modified[idx] = VLMVerdict(
                candidate=goal.candidate,
                result=VerificationResult.CONFIRMED,
                event_type=EventType.SHOT_ON_TARGET,
                confidence=goal.confidence * 0.85,
                reasoning=f"DOWNGRADED: no kickoff confirmation. "
                          f"Original: {goal.reasoning}",
                model_used=goal.model_used,
            )
            goal_downgrades += 1
            log.info("goal_inference.downgrade_no_kickoff",
                     time=f"{int(t//60):02d}:{t%60:05.2f}")

        # ── Step 5: Deduplicate goals within 4 min window ─────────────
        all_goals = sorted(
            [v for v in modified
             if v.result == VerificationResult.CONFIRMED
             and v.event_type == EventType.GOAL],
            key=lambda v: v.candidate.timestamp,
        )
        goals_to_remove: set[float] = set()
        for i, g in enumerate(all_goals):
            if g.candidate.timestamp in goals_to_remove:
                continue
            for j in range(i + 1, len(all_goals)):
                gap = (all_goals[j].candidate.timestamp
                       - g.candidate.timestamp)
                if gap < self._GOAL_DEDUP_WINDOW:
                    loser = (all_goals[j] if g.confidence >= all_goals[j].confidence
                             else g)
                    goals_to_remove.add(loser.candidate.timestamp)
                    lt = loser.candidate.timestamp
                    log.info("goal_inference.goal_dedup",
                             removed=f"{int(lt//60):02d}:{lt%60:05.2f}",
                             kept=f"{int(g.candidate.timestamp//60):02d}:"
                                  f"{g.candidate.timestamp%60:05.2f}",
                             gap=round(gap))

        for ts in goals_to_remove:
            for i, v in enumerate(modified):
                if (v.candidate.timestamp == ts
                        and v.event_type == EventType.GOAL):
                    modified[i] = VLMVerdict(
                        candidate=v.candidate,
                        result=VerificationResult.CONFIRMED,
                        event_type=EventType.SHOT_ON_TARGET,
                        confidence=v.confidence * 0.8,
                        reasoning=f"DEDUP: nearby goal takes precedence. "
                                  f"Original: {v.reasoning}",
                        model_used=v.model_used,
                    )
                    break

        log.info("goal_inference.complete",
                 shot_upgrades=upgrades,
                 goal_downgrades=goal_downgrades,
                 deduped=len(goals_to_remove),
                 final_goals=sum(
                     1 for v in modified
                     if v.result == VerificationResult.CONFIRMED
                     and v.event_type == EventType.GOAL))

        return modified

    # ------------------------------------------------------------------
    # Internal — set-piece inference (Phase 3c)
    # ------------------------------------------------------------------

    _SET_PIECE_MIN_GAP = 10.0     # Set piece ≥10s after shot
    _SET_PIECE_MAX_GAP = 90.0     # ... and ≤90s after shot
    _SET_PIECE_PROBES = [20, 35, 50]  # Direct probe offsets

    def _set_piece_inference(
        self,
        verdicts: list[VLMVerdict],
        all_candidates: list[EventCandidate],
    ) -> list[VLMVerdict]:
        """Post-VLM set-piece detection: rescan after shots for corners/goal kicks.

        Shots and saves often precede corners or goal kicks, but the
        restart candidate may not reach VLM (dropped by time-distributed
        sampling). This pass rescans for set pieces 10-90s after each shot.
        """
        confirmed = [v for v in verdicts
                     if v.result == VerificationResult.CONFIRMED]

        shot_types = {
            EventType.SHOT_ON_TARGET, EventType.SHOT_STOP_DIVING,
            EventType.SHOT_OFF_TARGET,
        }
        shots = [v for v in confirmed if v.event_type in shot_types]

        if not shots:
            log.info("set_piece_inference.no_shots")
            return verdicts

        vlm_times = {v.candidate.timestamp for v in verdicts}

        # Find rescan candidates after each shot
        rescan_map: dict[float, VLMVerdict] = {}
        rescan_candidates: list[EventCandidate] = []
        shots_without_rescan: list[VLMVerdict] = []

        for shot in shots:
            t = shot.candidate.timestamp
            found = False
            for cand in sorted(all_candidates, key=lambda c: c.timestamp):
                gap = cand.timestamp - t
                if gap < self._SET_PIECE_MIN_GAP:
                    continue
                if gap > self._SET_PIECE_MAX_GAP:
                    break
                if cand.timestamp not in vlm_times:
                    rescan_map[cand.timestamp] = shot
                    rescan_candidates.append(cand)
                    found = True
                    break

            if not found:
                shots_without_rescan.append(shot)

        # Direct probes for shots without motion candidates

        for shot in shots_without_rescan:
            t = shot.candidate.timestamp
            for offset in self._SET_PIECE_PROBES:
                probe_t = t + offset
                if probe_t > self._duration:
                    continue
                probe = EventCandidate(
                    timestamp=probe_t,
                    source=CandidateSource.SPOT_CHECK,
                    confidence=0.5,
                    context=VisualContext(),
                    clip_start=max(0, probe_t - 5),
                    clip_end=min(self._duration, probe_t + 15),
                )
                rescan_map[probe_t] = shot
                rescan_candidates.append(probe)

        if not rescan_candidates:
            log.info("set_piece_inference.no_candidates")
            return verdicts

        log.info("set_piece_inference.rescanning",
                 shots=len(shots),
                 candidates=len(rescan_candidates),
                 direct_probes=sum(
                     len(self._SET_PIECE_PROBES)
                     for _ in shots_without_rescan))

        # Send to VLM with focused set-piece prompt
        sp_verdicts = self._vlm_verifier.verify_set_piece(
            rescan_candidates,
            match_config=self._match_config,
        )

        # Dump diagnostics
        self._dump_diagnostics("set_piece_rescan", [
            {"timestamp": rv.candidate.timestamp,
             "result": rv.result.value,
             "event_type": rv.event_type.value if rv.event_type else None,
             "confidence": round(rv.confidence, 3),
             "reasoning": rv.reasoning,
             "for_shot_at": (rescan_map[rv.candidate.timestamp]
                             .candidate.timestamp
                             if rv.candidate.timestamp in rescan_map
                             else None),
             "mm_ss": f"{int(rv.candidate.timestamp//60):02d}:"
                      f"{rv.candidate.timestamp%60:05.2f}"}
            for rv in sp_verdicts
        ])

        # Collect confirmed set pieces
        new_events = [
            rv for rv in sp_verdicts
            if rv.result == VerificationResult.CONFIRMED
            and rv.event_type is not None
        ]

        if new_events:
            from collections import Counter
            types = Counter(e.event_type.value for e in new_events)
            log.info("set_piece_inference.found",
                     count=len(new_events), types=dict(types))
        else:
            log.info("set_piece_inference.none_found")

        # Add new set-piece verdicts to the verdict list
        modified = list(verdicts) + new_events
        return modified

    # ------------------------------------------------------------------
    # Internal — shot→restart reclassification (Phase 3c.5)
    # ------------------------------------------------------------------

    _SHOT_RESTART_WINDOW = 90.0  # seconds after shot to look for a restart
    _MISS_GAP_MAX = 25.0  # goal_kick within 25s of shot = likely miss

    def _shot_restart_reclassify(
        self, verdicts: list[VLMVerdict],
    ) -> list[VLMVerdict]:
        """Reclassify shots based on the restart that follows.

        - shot + corner_kick           → SHOT_STOP_DIVING (GK parried it out)
        - shot + goal_kick (gap <25s)  → SHOT_OFF_TARGET  (ball went out quickly)
        - shot + goal_kick (gap ≥25s)  → leave as SHOT_ON_TARGET (catch scan probes)

        The restart type is the strongest signal for what happened after a
        shot, since the VLM cannot reliably see GK save actions at distance.
        A long gap before a goal_kick may indicate a catch-then-distribution
        rather than a direct miss — the catch scan (Phase 3g) handles these.
        """
        confirmed = [v for v in verdicts
                     if v.result == VerificationResult.CONFIRMED]

        goal_kick_times = sorted(
            v.candidate.timestamp for v in confirmed
            if v.event_type == EventType.GOAL_KICK
        )
        corner_kick_times = sorted(
            v.candidate.timestamp for v in confirmed
            if v.event_type == EventType.CORNER_KICK
        )
        save_types = {
            EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING,
            EventType.CATCH, EventType.PUNCH,
        }
        save_times = sorted(
            v.candidate.timestamp for v in confirmed
            if v.event_type in save_types
        )

        modified = list(verdicts)
        to_off_target = 0
        to_parry = 0
        left_for_catch = 0

        for i, v in enumerate(modified):
            if (v.result != VerificationResult.CONFIRMED
                    or v.event_type != EventType.SHOT_ON_TARGET):
                continue

            t = v.candidate.timestamp

            # Skip if there's already a nearby save detected
            has_save = any(
                abs(st - t) < 15.0 for st in save_times
            )
            if has_save:
                continue

            # shot + corner = parry (GK deflected it over the line)
            has_corner = any(
                0 < ck - t < self._SHOT_RESTART_WINDOW
                for ck in corner_kick_times
            )
            if has_corner:
                modified[i] = VLMVerdict(
                    candidate=v.candidate,
                    result=VerificationResult.CONFIRMED,
                    event_type=EventType.SHOT_STOP_DIVING,
                    confidence=v.confidence,
                    reasoning=f"PARRY_INFERRED: corner_kick follows shot — "
                              f"GK deflected ball out. "
                              f"Original: {v.reasoning[:100]}",
                    model_used=v.model_used,
                )
                to_parry += 1
                continue

            # shot + goal_kick — check gap to distinguish miss from catch
            closest_gk_gap = float("inf")
            for gk in goal_kick_times:
                gap = gk - t
                if 0 < gap < self._SHOT_RESTART_WINDOW:
                    closest_gk_gap = min(closest_gk_gap, gap)

            has_goal_kick = closest_gk_gap < self._SHOT_RESTART_WINDOW

            if has_goal_kick and closest_gk_gap < self._MISS_GAP_MAX:
                # Short gap — ball went out quickly = miss
                modified[i] = VLMVerdict(
                    candidate=v.candidate,
                    result=VerificationResult.CONFIRMED,
                    event_type=EventType.SHOT_OFF_TARGET,
                    confidence=v.confidence,
                    reasoning=f"RESTART_RECLASS: goal_kick follows in "
                              f"{closest_gk_gap:.0f}s (short gap = miss). "
                              f"Original: {v.reasoning[:100]}",
                    model_used=v.model_used,
                )
                to_off_target += 1
            elif has_goal_kick:
                # Long gap — may be catch-then-distribution, leave for catch scan
                left_for_catch += 1
                log.info("shot_restart_reclass.long_gap",
                         time=f"{int(t//60):02d}:{t%60:05.2f}",
                         gap=f"{closest_gk_gap:.0f}s",
                         msg="leaving for catch scan")

        log.info("shot_restart_reclass.complete",
                 shots_checked=sum(
                     1 for v in verdicts
                     if v.result == VerificationResult.CONFIRMED
                     and v.event_type == EventType.SHOT_ON_TARGET),
                 to_parry=to_parry,
                 to_off_target=to_off_target,
                 left_for_catch=left_for_catch)
        return modified

    # ------------------------------------------------------------------
    # Internal — independent corner scan (Phase 3d)
    # ------------------------------------------------------------------

    _CORNER_DEDUP_WINDOW = 45.0    # Skip if corner already found within 45s
    _CORNER_GAP_THRESHOLD = 150.0  # Probe gaps between verdicts >2.5 min
    _CORNER_GAP_INTERVAL = 60.0    # Probe every 60s inside large gaps
    _CORNER_MAX_SCANS = 20         # Hard cap on corner scan VLM calls

    def _corner_scan(
        self,
        verdicts: list[VLMVerdict],
        all_candidates: list[EventCandidate],
    ) -> list[VLMVerdict]:
        """Independent corner detection via gap probes and targeted rescans.

        Corners are visually distinctive but sparse.  The set-piece rescan
        (Phase 3c) only triggers after shots/saves and uses fixed offsets
        that may miss the actual corner moment.

        This phase uses two strategies:
        1. **Gap probes** — in time gaps >2.5 min between VLM verdicts,
           create direct frame probes every 60s.  Catches corners in
           stretches of play where no motion spike was sampled.
        2. **Throw-in re-check** — throw-ins near the goal area can be
           misclassified corners.  Re-probe these with a corner-focused
           prompt.
        """
        confirmed = [v for v in verdicts
                     if v.result == VerificationResult.CONFIRMED]

        # Already-found corner timestamps — avoid re-scanning nearby
        existing_corners = {
            v.candidate.timestamp for v in confirmed
            if v.event_type == EventType.CORNER_KICK
        }

        # All timestamps already sent to VLM
        vlm_times = {v.candidate.timestamp for v in verdicts}

        scan_candidates: list[EventCandidate] = []
        scan_seen: set[float] = set()

        # ── Strategy 1: direct probes in large gaps ───────────────────

        sorted_times = sorted(vlm_times)
        gap_probes = 0

        for i in range(len(sorted_times) - 1):
            gap_start = sorted_times[i]
            gap_end = sorted_times[i + 1]
            gap = gap_end - gap_start

            if gap < self._CORNER_GAP_THRESHOLD:
                continue

            # Skip halftime gap (>10 min between events)
            if gap > 600:
                continue

            # Probe every _CORNER_GAP_INTERVAL seconds inside the gap
            probe_t = gap_start + self._CORNER_GAP_INTERVAL
            while probe_t < gap_end - 10:
                if probe_t in scan_seen:
                    probe_t += self._CORNER_GAP_INTERVAL
                    continue
                if any(abs(probe_t - ct) < self._CORNER_DEDUP_WINDOW
                       for ct in existing_corners):
                    probe_t += self._CORNER_GAP_INTERVAL
                    continue
                probe = EventCandidate(
                    timestamp=probe_t,
                    source=CandidateSource.SPOT_CHECK,
                    confidence=0.5,
                    context=VisualContext(),
                    clip_start=max(0, probe_t - 5),
                    clip_end=min(self._duration, probe_t + 15),
                )
                scan_candidates.append(probe)
                scan_seen.add(probe_t)
                gap_probes += 1
                probe_t += self._CORNER_GAP_INTERVAL

        # ── Strategy 2: re-probe throw-ins with corner prompt ─────────
        # Throw-ins near the goal line can be misclassified corners.
        throw_in_reprobes = 0
        throw_ins = [v for v in confirmed
                     if v.event_type == EventType.THROW_IN]
        for ti in throw_ins:
            t = ti.candidate.timestamp
            if t in scan_seen:
                continue
            if any(abs(t - ct) < self._CORNER_DEDUP_WINDOW
                   for ct in existing_corners):
                continue
            scan_candidates.append(ti.candidate)
            scan_seen.add(t)
            throw_in_reprobes += 1

        if not scan_candidates:
            log.info("corner_scan.no_candidates")
            return verdicts

        # Hard cap to avoid excessive VLM calls
        if len(scan_candidates) > self._CORNER_MAX_SCANS:
            scan_candidates.sort(key=lambda c: c.timestamp)
            scan_candidates = scan_candidates[:self._CORNER_MAX_SCANS]

        # Sort by time for orderly processing
        scan_candidates.sort(key=lambda c: c.timestamp)

        log.info("corner_scan.scanning",
                 gap_probes=gap_probes,
                 throw_in_reprobes=throw_in_reprobes,
                 total=len(scan_candidates))

        # Send focused corner prompt
        corner_verdicts = self._vlm_verifier.verify_corner(
            scan_candidates,
            match_config=self._match_config,
        )

        # Dump diagnostics
        self._dump_diagnostics("corner_scan", [
            {"timestamp": rv.candidate.timestamp,
             "result": rv.result.value,
             "event_type": rv.event_type.value if rv.event_type else None,
             "confidence": round(rv.confidence, 3),
             "reasoning": rv.reasoning,
             "mm_ss": f"{int(rv.candidate.timestamp//60):02d}:"
                      f"{rv.candidate.timestamp%60:05.2f}"}
            for rv in corner_verdicts
        ])

        # Collect confirmed corners
        new_corners = [
            rv for rv in corner_verdicts
            if rv.result == VerificationResult.CONFIRMED
            and rv.event_type == EventType.CORNER_KICK
        ]

        if new_corners:
            log.info("corner_scan.found",
                     count=len(new_corners),
                     times=[f"{int(c.candidate.timestamp//60):02d}:"
                            f"{c.candidate.timestamp%60:05.2f}"
                            for c in new_corners])
        else:
            log.info("corner_scan.none_found")

        modified = list(verdicts) + new_corners
        return modified

    # ------------------------------------------------------------------
    # Internal — match structure detection (Phase 2b)
    # ------------------------------------------------------------------

    _HALFTIME_GAP_MIN = 300.0   # Min gap (5 min) to qualify as halftime
    _MATCH_HALF_MAX = 3300.0    # Max half duration (55 min) for match end estimate
    _SPOT_CHECK_GAP = 90.0      # Insert probe in gaps >90s (was 180s)
    _SPOT_CHECK_MAX = 30        # Cap spot-check probes per match (was 15)
    _SPOT_CHECK_INTERVAL = 45.0  # Place probes every 45s within large gaps
    _AUDIO_GAP_WINDOW = 8.0     # Audio orphan if no motion within ±8s

    def _detect_match_structure(
        self,
        candidates: list[EventCandidate],
        audio_candidates: list[AudioCandidate],
    ) -> dict:
        """Detect halftime break and estimate match boundaries.

        Returns dict with keys:
          game_start: estimated start of play (seconds)
          halftime_start: start of halftime gap (or None)
          halftime_end: end of halftime gap (or None)
          match_end: estimated end of play (seconds)
        """
        if not candidates:
            return {
                "game_start": 0.0,
                "halftime_start": None,
                "halftime_end": None,
                "match_end": self._duration,
            }

        sorted_ts = sorted(c.timestamp for c in candidates)

        # Find largest gap ≥5 min between consecutive candidates
        best_gap = 0.0
        best_gap_start = 0.0
        best_gap_end = 0.0
        for i in range(len(sorted_ts) - 1):
            gap = sorted_ts[i + 1] - sorted_ts[i]
            if gap > best_gap:
                best_gap = gap
                best_gap_start = sorted_ts[i]
                best_gap_end = sorted_ts[i + 1]

        halftime_start = None
        halftime_end = None
        if best_gap >= self._HALFTIME_GAP_MIN:
            halftime_start = best_gap_start
            halftime_end = best_gap_end

        # Estimate game start: first cluster of activity
        game_start = max(0.0, sorted_ts[0] - 30)

        # Estimate match end: halftime_end + 55 min, or game_start + 100 min
        if halftime_end is not None:
            match_end = halftime_end + self._MATCH_HALF_MAX
        else:
            match_end = game_start + 6000.0  # 100 min fallback

        match_end = min(match_end, self._duration)

        log.info("pipeline.match_structure",
                 game_start=f"{game_start/60:.1f}min",
                 halftime=f"{halftime_start/60:.1f}-{halftime_end/60:.1f}min"
                          if halftime_start else "not_detected",
                 halftime_gap=f"{best_gap/60:.1f}min",
                 match_end=f"{match_end/60:.1f}min")

        return {
            "game_start": game_start,
            "halftime_start": halftime_start,
            "halftime_end": halftime_end,
            "match_end": match_end,
        }

    def _filter_by_match_structure(
        self,
        candidates: list[EventCandidate],
        match_struct: dict,
    ) -> list[EventCandidate]:
        """Remove candidates outside match boundaries and during halftime."""
        game_start = match_struct["game_start"]
        match_end = match_struct["match_end"]
        ht_start = match_struct.get("halftime_start")
        ht_end = match_struct.get("halftime_end")

        filtered = []
        for c in candidates:
            if c.timestamp < game_start or c.timestamp > match_end:
                continue
            if ht_start and ht_end and ht_start < c.timestamp < ht_end:
                continue
            filtered.append(c)

        dropped = len(candidates) - len(filtered)
        if dropped:
            log.info("pipeline.match_filter",
                     before=len(candidates), after=len(filtered),
                     dropped=dropped)
        return filtered

    # ------------------------------------------------------------------
    # Internal — audio gap fill (Phase 2c)
    # ------------------------------------------------------------------

    def _audio_gap_fill(
        self,
        candidates: list[EventCandidate],
        audio_candidates: list[AudioCandidate],
        match_struct: dict,
    ) -> list[EventCandidate]:
        """Promote orphan audio cues (not near any motion candidate) to candidates.

        Fills gaps where motion scan missed events but audio detected
        whistles or crowd surges.
        """
        if not audio_candidates:
            return candidates

        match_end = match_struct["match_end"]
        game_start = match_struct["game_start"]
        motion_ts = {c.timestamp for c in candidates}

        new_candidates: list[EventCandidate] = []
        for ac in audio_candidates:
            if ac.timestamp < game_start or ac.timestamp > match_end:
                continue
            # Only promote whistles and whistle+surge (not pure surges — too noisy)
            if ac.cue_type == AudioCueType.ENERGY_SURGE:
                continue
            # Check if any motion candidate is within the window
            is_orphan = all(
                abs(ac.timestamp - mt) > self._AUDIO_GAP_WINDOW
                for mt in motion_ts
            )
            if not is_orphan:
                continue

            clip_start = max(0, ac.timestamp - 5.0)
            clip_end = min(self._duration, ac.timestamp + 15.0)
            new_candidates.append(EventCandidate(
                timestamp=ac.timestamp,
                source=CandidateSource.AUDIO_WHISTLE
                       if ac.cue_type == AudioCueType.WHISTLE
                       else CandidateSource.AUDIO_BOTH,
                confidence=0.35,  # Low-energy events need VLM to confirm
                context=VisualContext(audio_boost=True),
                audio_cue=ac,
                clip_start=clip_start,
                clip_end=clip_end,
            ))

        if new_candidates:
            log.info("pipeline.audio_gap_fill",
                     orphan_audio=len(new_candidates),
                     times=[f"{int(c.timestamp//60):02d}:{c.timestamp%60:05.2f}"
                            for c in new_candidates])

        return candidates + new_candidates

    # ------------------------------------------------------------------
    # Internal — spot-check probes (Phase 2d)
    # ------------------------------------------------------------------

    def _spot_check_probes(
        self,
        candidates: list[EventCandidate],
        match_struct: dict,
    ) -> list[EventCandidate]:
        """Insert VLM probes in temporal gaps >90s between candidates.

        Places probes every _SPOT_CHECK_INTERVAL seconds within each gap,
        catching low-energy events (throw-ins, free kicks, goal kicks) that
        motion+audio detection missed.
        """
        if not candidates:
            return candidates

        match_end = match_struct["match_end"]
        ht_start = match_struct.get("halftime_start")
        ht_end = match_struct.get("halftime_end")

        sorted_ts = sorted(c.timestamp for c in candidates)
        probes: list[EventCandidate] = []

        for i in range(len(sorted_ts) - 1):
            gap_start = sorted_ts[i]
            gap_end = sorted_ts[i + 1]
            gap = gap_end - gap_start

            if gap < self._SPOT_CHECK_GAP:
                continue
            # Skip halftime gap
            if ht_start and ht_end and gap_start < ht_end and gap_end > ht_start:
                continue

            # Place probes every _SPOT_CHECK_INTERVAL within the gap
            # (offset from gap_start to avoid probing right at existing candidates)
            t = gap_start + self._SPOT_CHECK_INTERVAL
            while t < gap_end - 10.0:  # Don't probe too close to gap_end
                if t > match_end:
                    break

                probes.append(EventCandidate(
                    timestamp=t,
                    source=CandidateSource.SPOT_CHECK,
                    confidence=0.35,
                    context=VisualContext(),
                    clip_start=max(0, t - 5.0),
                    clip_end=min(self._duration, t + 15.0),
                ))

                if len(probes) >= self._SPOT_CHECK_MAX:
                    break

                t += self._SPOT_CHECK_INTERVAL

            if len(probes) >= self._SPOT_CHECK_MAX:
                break

        if probes:
            log.info("pipeline.spot_check_probes",
                     count=len(probes),
                     times=[f"{int(p.timestamp//60):02d}:{p.timestamp%60:05.2f}"
                            for p in probes])

        return candidates + probes

    # ------------------------------------------------------------------
    # Internal — reverse restart inference (Phase 3e)
    # ------------------------------------------------------------------

    _REVERSE_RESTART_LOOKBACK_SEC = 30.0   # Look up to 30s before restart
    _REVERSE_RESTART_MIN_GAP = 3.0         # At least 3s before restart

    def _reverse_restart_inference(
        self, verdicts: list[VLMVerdict],
    ) -> list[VLMVerdict]:
        """Work backwards from restarts to find missed shots.

        For each confirmed goal_kick or corner_kick, check whether there
        is a rejected VLM candidate 3-30s *before* it.  If so:
          - goal_kick with no save nearby → rejected candidate = shot off target
          - corner_kick with no save nearby → rejected candidate = blocked shot
        """
        confirmed = [v for v in verdicts
                     if v.result == VerificationResult.CONFIRMED]
        rejected = [v for v in verdicts
                    if v.result == VerificationResult.REJECTED]

        if not rejected:
            log.info("reverse_restart.no_rejected")
            return verdicts

        restarts = [
            v for v in confirmed
            if v.event_type in {EventType.GOAL_KICK, EventType.CORNER_KICK}
        ]
        if not restarts:
            log.info("reverse_restart.no_restarts")
            return verdicts

        # Timestamps of existing saves (to avoid re-inferring)
        save_types = {
            EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING,
            EventType.CATCH, EventType.PUNCH,
        }
        save_times = sorted(
            v.candidate.timestamp for v in confirmed
            if v.event_type in save_types
        )
        # Timestamps of already-recovered shots (from reclassification)
        shot_types = {
            EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
            EventType.GOAL, EventType.NEAR_MISS,
        }
        shot_times = sorted(
            v.candidate.timestamp for v in confirmed
            if v.event_type in shot_types
        )

        modified = list(verdicts)
        _id_to_idx = {id(v): i for i, v in enumerate(modified)}
        upgraded = 0

        for restart in restarts:
            rt = restart.candidate.timestamp

            # Is there already a save or shot within 30s before this restart?
            has_nearby_event = (
                any(rt - self._REVERSE_RESTART_LOOKBACK_SEC < st < rt
                    for st in save_times)
                or any(rt - self._REVERSE_RESTART_LOOKBACK_SEC < st < rt
                       for st in shot_times)
            )
            if has_nearby_event:
                continue

            # Find rejected candidates 3-30s before the restart
            best: VLMVerdict | None = None
            best_gap = float("inf")
            for rej in rejected:
                gap = rt - rej.candidate.timestamp
                if (self._REVERSE_RESTART_MIN_GAP < gap
                        < self._REVERSE_RESTART_LOOKBACK_SEC
                        and gap < best_gap):
                    best = rej
                    best_gap = gap

            if best is None:
                continue

            # Infer event type from the restart that follows
            if restart.event_type == EventType.CORNER_KICK:
                event_type = EventType.SHOT_OFF_TARGET   # Deflected → corner
                label = "blocked/deflected → corner"
            else:
                event_type = EventType.SHOT_OFF_TARGET   # Missed → goal kick
                label = "missed → goal kick"

            idx = _id_to_idx.get(id(best))
            if idx is None:
                continue
            modified[idx] = VLMVerdict(
                candidate=best.candidate,
                result=VerificationResult.CONFIRMED,
                event_type=event_type,
                confidence=0.6,
                reasoning=f"REVERSE_RESTART: {label} at "
                          f"{int(rt//60):02d}:{rt%60:05.2f}. "
                          f"Original: {best.reasoning[:100]}",
                model_used=best.model_used,
            )
            upgraded += 1
            t = best.candidate.timestamp
            log.info("reverse_restart.upgrade",
                     time=f"{int(t//60):02d}:{t%60:05.2f}",
                     restart_type=restart.event_type.value,
                     restart_time=f"{int(rt//60):02d}:{rt%60:05.2f}",
                     gap=round(best_gap, 1))

        log.info("reverse_restart.complete",
                 restarts=len(restarts), upgraded=upgraded)
        return modified

    # ------------------------------------------------------------------
    # Internal — binary shot scan (Phase 3f)
    # ------------------------------------------------------------------

    _SHOT_SCAN_MAX = 30     # Cap VLM calls for shot re-probes

    def _shot_scan(self, verdicts: list[VLMVerdict]) -> list[VLMVerdict]:
        """Re-probe remaining rejected candidates with a binary shot prompt.

        After all other phases, some rejected candidates may still contain
        shots the multi-class prompt missed.  A focused "was a shot taken?"
        question performs better for binary classification.
        """
        # Collect still-rejected verdicts (not recovered by earlier phases)
        rejected = [
            v for v in verdicts
            if v.result == VerificationResult.REJECTED
        ]
        if not rejected:
            log.info("shot_scan.no_rejected")
            return verdicts

        # Don't rescan candidates near already-confirmed events
        confirmed_times = sorted(
            v.candidate.timestamp for v in verdicts
            if v.result == VerificationResult.CONFIRMED
        )

        scan_candidates: list[EventCandidate] = []
        scan_map: dict[float, int] = {}  # candidate timestamp → verdict index
        _id_to_idx = {id(v): i for i, v in enumerate(verdicts)}

        for v in rejected:
            t = v.candidate.timestamp
            # Skip if near a confirmed event (within 10s)
            if any(abs(t - ct) < 10.0 for ct in confirmed_times):
                continue
            idx = _id_to_idx.get(id(v))
            if idx is None:
                continue
            scan_map[t] = idx
            scan_candidates.append(v.candidate)

        if not scan_candidates:
            log.info("shot_scan.all_near_confirmed")
            return verdicts

        # Cap and distribute evenly
        if len(scan_candidates) > self._SHOT_SCAN_MAX:
            scan_candidates = self._time_distributed_sample(
                scan_candidates, self._SHOT_SCAN_MAX,
            )
            scan_map = {
                c.timestamp: scan_map[c.timestamp]
                for c in scan_candidates
                if c.timestamp in scan_map
            }

        log.info("shot_scan.rescanning", candidates=len(scan_candidates))

        shot_verdicts = self._vlm_verifier.verify_shot(
            scan_candidates,
            match_config=self._match_config,
        )

        # Dump diagnostics
        self._dump_diagnostics("shot_scan", [
            {"timestamp": sv.candidate.timestamp,
             "result": sv.result.value,
             "event_type": sv.event_type.value if sv.event_type else None,
             "confidence": round(sv.confidence, 3),
             "reasoning": sv.reasoning,
             "mm_ss": f"{int(sv.candidate.timestamp//60):02d}:"
                      f"{sv.candidate.timestamp%60:05.2f}"}
            for sv in shot_verdicts
        ])

        # Merge confirmed shots back into verdict list
        modified = list(verdicts)
        upgraded = 0

        for sv in shot_verdicts:
            if sv.result != VerificationResult.CONFIRMED:
                continue
            t = sv.candidate.timestamp
            if t not in scan_map:
                continue
            idx = scan_map[t]
            modified[idx] = sv
            upgraded += 1
            log.info("shot_scan.upgrade",
                     time=f"{int(t//60):02d}:{t%60:05.2f}",
                     event_type=sv.event_type.value if sv.event_type else None,
                     reasoning=sv.reasoning[:80])

        log.info("shot_scan.complete",
                 scanned=len(scan_candidates), upgraded=upgraded)
        return modified

    # ------------------------------------------------------------------
    # Internal — catch scan (Phase 3g)
    # ------------------------------------------------------------------

    _CATCH_SCAN_MAX = 25          # Cap VLM calls for catch probes
    _CATCH_NO_RESTART_WINDOW = 60.0  # No restart within 60s → structural catch

    def _catch_scan(self, verdicts: list[VLMVerdict]) -> list[VLMVerdict]:
        """Detect catches via structural inference and VLM probing.

        Catches are invisible to motion detection (GK just holds the ball)
        and hard for VLMs to see at distance.  But they have a distinctive
        structural signature: after a catch, the GK distributes in open play
        — no dead-ball restart (corner/goal_kick) follows.

        Two strategies:
        1. **Structural**: shot + no restart within 60s → CATCH (GK caught
           and distributed without a stoppage).
        2. **VLM probe**: shot + goal_kick with long gap (≥25s) → probe
           frames at shot+3s…+8s for "is the GK holding the ball?"
        """
        confirmed = [v for v in verdicts
                     if v.result == VerificationResult.CONFIRMED]

        goal_kick_times = sorted(
            v.candidate.timestamp for v in confirmed
            if v.event_type == EventType.GOAL_KICK
        )
        corner_kick_times = sorted(
            v.candidate.timestamp for v in confirmed
            if v.event_type == EventType.CORNER_KICK
        )
        kickoff_times = sorted(
            v.candidate.timestamp for v in confirmed
            if v.event_type == EventType.KICKOFF
        )
        goal_times = sorted(
            v.candidate.timestamp for v in confirmed
            if v.event_type == EventType.GOAL
        )
        save_types = {
            EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING,
            EventType.CATCH, EventType.PUNCH,
        }
        save_times = sorted(
            v.candidate.timestamp for v in confirmed
            if v.event_type in save_types
        )

        modified = list(verdicts)
        structural_catches = 0
        vlm_probe_candidates: list[EventCandidate] = []
        vlm_probe_indices: list[int] = []

        for i, v in enumerate(modified):
            if (v.result != VerificationResult.CONFIRMED
                    or v.event_type != EventType.SHOT_ON_TARGET):
                continue

            t = v.candidate.timestamp

            # Skip if already has a nearby save
            if any(abs(st - t) < 15.0 for st in save_times):
                continue

            # Skip if near a goal (goal inference handled it)
            if any(abs(gt - t) < 30.0 for gt in goal_times):
                continue

            # Check for restarts after the shot
            has_corner = any(
                0 < ck - t < self._SHOT_RESTART_WINDOW
                for ck in corner_kick_times
            )
            has_kickoff = any(
                0 < ko - t < 180.0 for ko in kickoff_times
            )

            if has_corner or has_kickoff:
                # Parry/goal inference already handled these
                continue

            # Find closest goal kick
            closest_gk_gap = float("inf")
            for gk in goal_kick_times:
                gap = gk - t
                if 0 < gap < self._SHOT_RESTART_WINDOW:
                    closest_gk_gap = min(closest_gk_gap, gap)

            has_goal_kick = closest_gk_gap < self._SHOT_RESTART_WINDOW

            if not has_goal_kick:
                # No restart at all → structural catch
                modified[i] = VLMVerdict(
                    candidate=v.candidate,
                    result=VerificationResult.CONFIRMED,
                    event_type=EventType.CATCH,
                    confidence=v.confidence * 0.85,
                    reasoning=f"CATCH_INFERRED: no restart follows shot "
                              f"within {self._CATCH_NO_RESTART_WINDOW:.0f}s "
                              f"— GK caught and distributed in open play. "
                              f"Original: {v.reasoning[:100]}",
                    model_used=v.model_used,
                )
                structural_catches += 1
                t_fmt = f"{int(t//60):02d}:{t%60:05.2f}"
                log.info("catch_scan.structural",
                         time=t_fmt, confidence=v.confidence * 0.85)
            else:
                # Long gap to goal kick — VLM probe for GK holding ball
                vlm_probe_candidates.append(v.candidate)
                vlm_probe_indices.append(i)

        # VLM catch probes for ambiguous cases (shot + long-gap goal_kick)
        vlm_catches = 0
        if vlm_probe_candidates and self._vlm_enabled:
            vlm_available = self._vlm_verifier.is_available()
            if vlm_available:
                if len(vlm_probe_candidates) > self._CATCH_SCAN_MAX:
                    vlm_probe_candidates = (
                        vlm_probe_candidates[:self._CATCH_SCAN_MAX]
                    )
                    vlm_probe_indices = (
                        vlm_probe_indices[:self._CATCH_SCAN_MAX]
                    )

                log.info("catch_scan.vlm_probing",
                         candidates=len(vlm_probe_candidates))

                catch_verdicts = self._vlm_verifier.verify_catch(
                    vlm_probe_candidates,
                    match_config=self._match_config,
                )

                self._dump_diagnostics("catch_scan", [
                    {"timestamp": cv.candidate.timestamp,
                     "result": cv.result.value,
                     "event_type": (cv.event_type.value
                                    if cv.event_type else None),
                     "confidence": round(cv.confidence, 3),
                     "reasoning": cv.reasoning,
                     "mm_ss": f"{int(cv.candidate.timestamp//60):02d}:"
                              f"{cv.candidate.timestamp%60:05.2f}"}
                    for cv in catch_verdicts
                ])

                for cv, idx in zip(catch_verdicts, vlm_probe_indices):
                    if cv.result == VerificationResult.CONFIRMED:
                        modified[idx] = cv
                        vlm_catches += 1
                        t = cv.candidate.timestamp
                        log.info("catch_scan.vlm_confirmed",
                                 time=f"{int(t//60):02d}:{t%60:05.2f}",
                                 reasoning=cv.reasoning[:100])

        # Downgrade unconfirmed probes to SHOT_OFF_TARGET.
        # These are shots with a long-gap goal_kick where the catch
        # hypothesis was tested (VLM probe) and rejected — so the
        # ball went out without a save, just with a longer delay.
        downgraded = 0
        probed_set = {c.timestamp for c in vlm_probe_candidates}
        for i, v in enumerate(modified):
            if (v.result != VerificationResult.CONFIRMED
                    or v.event_type != EventType.SHOT_ON_TARGET):
                continue
            if v.candidate.timestamp not in probed_set:
                continue
            # Still SHOT_ON_TARGET → catch probe rejected → off target
            modified[i] = VLMVerdict(
                candidate=v.candidate,
                result=VerificationResult.CONFIRMED,
                event_type=EventType.SHOT_OFF_TARGET,
                confidence=v.confidence,
                reasoning=f"CATCH_REJECTED: VLM probe found no GK "
                          f"holding ball — goal_kick follows, likely "
                          f"miss. Original: {v.reasoning[:100]}",
                model_used=v.model_used,
            )
            downgraded += 1
            t = v.candidate.timestamp
            log.info("catch_scan.downgrade_to_off_target",
                     time=f"{int(t//60):02d}:{t%60:05.2f}")

        log.info("catch_scan.complete",
                 structural=structural_catches,
                 vlm_probed=len(vlm_probe_candidates),
                 vlm_confirmed=vlm_catches,
                 downgraded_to_off=downgraded,
                 total_catches=structural_catches + vlm_catches)
        return modified

    # ------------------------------------------------------------------
    # Phase 3h: Claude goal verification
    # ------------------------------------------------------------------

    _CLAUDE_GOAL_ENABLED = True
    _CLAUDE_GOAL_KICKOFF_MIN = 20.0   # Min gap to kickoff (seconds)
    _CLAUDE_GOAL_KICKOFF_MAX = 120.0  # Max gap to kickoff (seconds)
    _CLAUDE_GOAL_MIN_CONF = 0.7       # Min Claude confidence to upgrade

    def _claude_goal_verification(
        self,
        verdicts: list[VLMVerdict],
        all_candidates: list[EventCandidate],
    ) -> list[VLMVerdict]:
        """Re-verify non-goal events with Claude when kickoff evidence exists.

        For each save/free_kick event that has a kickoff 20-120s later,
        send frames to Claude to check if it's actually a goal.
        """
        if not self._CLAUDE_GOAL_ENABLED:
            return verdicts
        if not self._vlm_verifier._anthropic_key:
            log.info("claude_goal_verify.skipped", reason="no API key")
            return verdicts

        confirmed = [v for v in verdicts
                     if v.result == VerificationResult.CONFIRMED]

        # Types that might be misclassified goals
        recheck_types = {
            EventType.SHOT_STOP_DIVING,
            EventType.FREE_KICK_SHOT,
            EventType.SHOT_ON_TARGET,
        }
        recheck = [v for v in confirmed if v.event_type in recheck_types]

        if not recheck:
            return verdicts

        # Find kickoffs (including structurally inferred ones)
        kickoff_times = []
        for v in confirmed:
            if v.event_type == EventType.KICKOFF:
                kickoff_times.append(v.candidate.timestamp)
        # Also check all motion candidates for potential kickoff evidence
        vlm_times = {v.candidate.timestamp for v in verdicts}
        for cand in sorted(all_candidates, key=lambda c: c.timestamp):
            if cand.timestamp not in vlm_times:
                continue

        kickoff_times.sort()

        # Find candidates with kickoff evidence
        candidates_for_claude: list[tuple[EventCandidate, str, float]] = []
        candidate_verdicts: list[VLMVerdict] = []

        for v in recheck:
            t = v.candidate.timestamp
            # Check for a kickoff 20-120s after this event
            for ko_t in kickoff_times:
                gap = ko_t - t
                if self._CLAUDE_GOAL_KICKOFF_MIN < gap < self._CLAUDE_GOAL_KICKOFF_MAX:
                    candidates_for_claude.append((
                        v.candidate,
                        v.event_type.value,
                        gap,
                    ))
                    candidate_verdicts.append(v)
                    break

        if not candidates_for_claude:
            log.info("claude_goal_verify.no_candidates_with_kickoff")
            return verdicts

        log.info("claude_goal_verify.start",
                 candidates=len(candidates_for_claude))

        # Call Claude
        results = self._vlm_verifier.goal_verify_claude(
            candidates_for_claude,
            source_file=self._source,
        )

        # Upgrade confirmed goals
        verdict_map = {id(v): v for v in verdicts}
        modified = list(verdicts)
        upgrades = 0

        for i, (candidate, is_goal, conf, reason) in enumerate(results):
            if is_goal and conf >= self._CLAUDE_GOAL_MIN_CONF:
                old_verdict = candidate_verdicts[i]
                t = old_verdict.candidate.timestamp
                mm, ss = int(t // 60), t % 60

                new_verdict = VLMVerdict(
                    candidate=old_verdict.candidate,
                    result=VerificationResult.CONFIRMED,
                    event_type=EventType.GOAL,
                    confidence=conf,
                    reasoning=f"CLAUDE GOAL: {reason}. "
                              f"Original 8B: {old_verdict.event_type.value}",
                    model_used="claude-goal-verify",
                )
                # Replace in the list
                idx = modified.index(old_verdict)
                modified[idx] = new_verdict
                upgrades += 1

                log.info("claude_goal_verify.upgrade",
                         time=f"{mm:02d}:{ss:05.2f}",
                         from_type=old_verdict.event_type.value,
                         confidence=conf, reasoning=reason[:80])

        log.info("claude_goal_verify.complete",
                 checked=len(candidates_for_claude),
                 upgraded=upgrades)

        return modified

    # ------------------------------------------------------------------
    # Internal — diagnostics
    # ------------------------------------------------------------------

    def _dump_diagnostics(self, stage: str, data: list[dict]):
        """Write diagnostic data to a JSONL file for post-run analysis."""
        diag_dir = self._work / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        path = diag_dir / f"{stage}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        log.info(f"pipeline.diag.{stage}", path=str(path), count=len(data))

    # ------------------------------------------------------------------
    # Internal — time-distributed sampling
    # ------------------------------------------------------------------

    def _time_distributed_sample(
        self,
        candidates: list[EventCandidate],
        max_count: int,
    ) -> list[EventCandidate]:
        """Sample candidates evenly across time instead of by confidence.

        Divides the video into equal time bins and takes the highest-
        confidence candidate from each bin.  Audio-boosted candidates
        get priority within their bin.
        """
        if len(candidates) <= max_count:
            return candidates

        sorted_c = sorted(candidates, key=lambda c: c.timestamp)
        t_min = sorted_c[0].timestamp
        t_max = sorted_c[-1].timestamp
        t_range = max(t_max - t_min, 1.0)

        # Number of bins = max_count
        bin_size = t_range / max_count
        bins: dict[int, list[EventCandidate]] = {}
        for c in sorted_c:
            b = int((c.timestamp - t_min) / bin_size)
            b = min(b, max_count - 1)  # Clamp last bin
            bins.setdefault(b, []).append(c)

        # From each bin, pick best candidate (audio-boosted first, then confidence)
        selected: list[EventCandidate] = []
        for b in range(max_count):
            if b not in bins:
                continue
            bin_candidates = bins[b]
            # Sort: audio-boosted first, then by confidence desc
            bin_candidates.sort(
                key=lambda c: (c.context.audio_boost, c.confidence),
                reverse=True,
            )
            selected.append(bin_candidates[0])

        dropped = len(candidates) - len(selected)
        log.info("pipeline.vlm_sampled",
                 total=len(candidates), kept=len(selected),
                 dropped=dropped, bins=max_count,
                 bin_sec=round(bin_size, 1))
        return selected

    # ------------------------------------------------------------------
    # Internal — verdict/candidate → Event conversion
    # ------------------------------------------------------------------

    # Save types that imply a shot on target (dual-emit for highlights reel)
    _SAVE_TYPES = {
        EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING,
        EventType.CATCH, EventType.PUNCH,
    }

    def _verdicts_to_events(self, verdicts: list[VLMVerdict]) -> list[Event]:
        """Convert VLM verdicts to Event objects.

        Save verdicts also emit a companion SHOT_ON_TARGET event so
        that the highlights reel includes shots that resulted in saves.
        """
        events: list[Event] = []

        for v in verdicts:
            if v.result == VerificationResult.REJECTED:
                continue

            event_type = v.event_type
            if event_type is None:
                # Uncertain with no type — skip
                if v.result == VerificationResult.UNCERTAIN:
                    continue
                event_type = EventType.SHOT_ON_TARGET  # Default fallback

            confidence = v.confidence
            if confidence < self._min_conf:
                continue

            c = v.candidate
            base_meta = {
                "source": c.source.value,
                "vlm_result": v.result.value,
                "vlm_reasoning": v.reasoning,
                "vlm_model": v.model_used,
                "audio_cue": c.audio_cue.cue_type.value if c.audio_cue else None,
                "motion_magnitude": c.context.motion_magnitude,
                "spike_duration": c.context.spike_duration_sec,
                "audio_boost": c.context.audio_boost,
            }
            events.append(self._make_event(
                event_type=event_type,
                timestamp=c.timestamp,
                clip_start=c.clip_start,
                clip_end=c.clip_end,
                confidence=confidence,
                metadata=base_meta,
            ))

            # Dual-emit: every save implies a shot on target
            if event_type in self._SAVE_TYPES:
                events.append(self._make_event(
                    event_type=EventType.SHOT_ON_TARGET,
                    timestamp=c.timestamp,
                    clip_start=c.clip_start,
                    clip_end=c.clip_end,
                    confidence=confidence * 0.95,  # Slightly lower
                    metadata={**base_meta, "dual_emit_from": event_type.value},
                ))

        return events

    def _candidates_to_events(
        self, candidates: list[EventCandidate],
    ) -> list[Event]:
        """Convert unverified candidates to events (VLM unavailable path)."""
        events: list[Event] = []

        for c in candidates:
            if c.confidence < self._min_conf:
                continue

            # Without VLM, we can only guess the event type from context
            event_type = EventType.SHOT_ON_TARGET  # Safe default
            if c.source == CandidateSource.AUDIO_WHISTLE:
                event_type = EventType.FREE_KICK_SHOT

            events.append(self._make_event(
                event_type=event_type,
                timestamp=c.timestamp,
                clip_start=c.clip_start,
                clip_end=c.clip_end,
                confidence=c.confidence * 0.7,  # Penalise unverified
                metadata={
                    "source": c.source.value,
                    "vlm_result": "unverified",
                    "audio_cue": c.audio_cue.cue_type.value if c.audio_cue else None,
                    "motion_magnitude": c.context.motion_magnitude,
                    "spike_duration": c.context.spike_duration_sec,
                    "audio_boost": c.context.audio_boost,
                },
            ))

        return events

    def _make_event(
        self,
        event_type: EventType,
        timestamp: float,
        clip_start: float,
        clip_end: float,
        confidence: float,
        metadata: dict,
    ) -> Event:
        """Create an Event object from detection results."""
        from src.detection.models import (
            EVENT_REEL_MAP,
            EVENT_TYPE_CONFIG,
            is_gk_event_type,
        )

        reel_targets = list(EVENT_REEL_MAP.get(event_type, []))
        is_gk = is_gk_event_type(event_type)

        # Use EventTypeConfig padding if available
        cfg = EVENT_TYPE_CONFIG.get(event_type)
        if cfg:
            clip_start = max(0, timestamp - cfg.pre_pad_sec)
            clip_end = min(self._duration, timestamp + cfg.post_pad_sec)

        frame_start = int(clip_start * self._fps)
        frame_end = int(clip_end * self._fps)

        return Event(
            event_id=str(uuid.uuid4()),
            job_id=self._job_id,
            source_file=str(self._source),
            event_type=event_type,
            timestamp_start=clip_start,
            timestamp_end=clip_end,
            confidence=confidence,
            reel_targets=reel_targets,
            is_goalkeeper_event=is_gk,
            frame_start=frame_start,
            frame_end=frame_end,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal — deduplication
    # ------------------------------------------------------------------

    # Priority for cross-type dedup: higher priority wins
    _EVENT_PRIORITY: dict[EventType, int] = {
        EventType.GOAL: 100,
        EventType.PENALTY: 95,
        EventType.SHOT_STOP_DIVING: 90,
        EventType.SHOT_STOP_STANDING: 90,
        EventType.CATCH: 85,
        EventType.PUNCH: 85,
        EventType.ONE_ON_ONE: 85,
        EventType.SHOT_ON_TARGET: 70,
        EventType.SHOT_OFF_TARGET: 65,
        EventType.NEAR_MISS: 60,
        EventType.CORNER_KICK: 55,
        EventType.GOAL_KICK: 50,
        EventType.FREE_KICK_SHOT: 50,
        EventType.THROW_IN: 30,
        EventType.KICKOFF: 20,
    }

    # Related event groups — cross-type dedup applies within groups.
    # Saves and shots are in SEPARATE groups so both survive dedup
    # (saves → GK reel, shots → highlights reel).
    _RELATED_GROUPS: list[set[EventType]] = [
        # GK-reel save events (dedup among themselves)
        {EventType.SHOT_STOP_DIVING, EventType.SHOT_STOP_STANDING,
         EventType.CATCH, EventType.PUNCH, EventType.ONE_ON_ONE},
        # Highlights-reel shot events (dedup among themselves)
        {EventType.GOAL, EventType.SHOT_ON_TARGET, EventType.SHOT_OFF_TARGET,
         EventType.NEAR_MISS, EventType.PENALTY, EventType.FREE_KICK_SHOT},
        # Set pieces
        {EventType.CORNER_KICK, EventType.THROW_IN, EventType.GOAL_KICK},
    ]

    def _deduplicate(
        self, events: list[Event], window_sec: float = 15.0,
    ) -> list[Event]:
        """Remove near-duplicate events — same-type and cross-type.

        Same-type: merge within *window_sec*, keep higher confidence.
        Cross-type: within *window_sec* and same related group, keep
        higher-priority event type.
        """
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e.timestamp_start)

        # Pass 1: same-type dedup
        same_deduped: list[Event] = [sorted_events[0]]
        for e in sorted_events[1:]:
            prev = same_deduped[-1]
            if (
                e.event_type == prev.event_type
                and abs(e.timestamp_start - prev.timestamp_start) < window_sec
            ):
                if e.confidence > prev.confidence:
                    same_deduped[-1] = e
            else:
                same_deduped.append(e)

        # Pass 2: cross-type dedup within related groups
        result: list[Event] = [same_deduped[0]]
        for e in same_deduped[1:]:
            prev = result[-1]
            if abs(e.timestamp_start - prev.timestamp_start) < window_sec:
                # Check if they're in the same related group
                same_group = any(
                    e.event_type in grp and prev.event_type in grp
                    for grp in self._RELATED_GROUPS
                )
                if same_group:
                    # Keep the higher-priority event type
                    e_pri = self._EVENT_PRIORITY.get(e.event_type, 0)
                    p_pri = self._EVENT_PRIORITY.get(prev.event_type, 0)
                    if e_pri > p_pri:
                        result[-1] = e
                    # else keep prev (higher or equal priority)
                    continue
            result.append(e)

        if len(sorted_events) != len(result):
            log.info("pipeline.dedup",
                     before=len(sorted_events), after=len(result),
                     same_type_removed=len(sorted_events) - len(same_deduped),
                     cross_type_removed=len(same_deduped) - len(result))

        return result
