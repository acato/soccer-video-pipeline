"""
Detection pipeline orchestrator — motion-first architecture.

Phase ordering:
  1. Motion scan — dense frame-differencing to find activity spikes
  2. Audio detection — optional booster for co-located motion candidates
  3. VLM classification — two-pass verify + classify via Qwen3-VL / Claude
  3b. Goal inference — kickoff rescan to upgrade shots, dedup nearby goals

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

from src.detection.audio_detector import AudioDetector
from src.detection.models import Event, EventType
from src.detection.visual_candidate import (
    CandidateSource,
    EventCandidate,
    VisualCandidateGenerator,
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
        vllm_model: str = "Qwen/Qwen3-VL-32B-Instruct-FP8",
        anthropic_api_key: Optional[str] = None,
        anthropic_model: str = "claude-sonnet-4-20250514",
        vlm_min_confidence: float = 0.5,
        vlm_enabled: bool = True,
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

        # ── Phase 3: VLM classification (55% → 90%) ───────────────────
        # Keep full candidate list for goal inference rescan
        all_motion_candidates = list(candidates)

        # Cap candidates sent to VLM — distribute evenly across time
        _VLM_MAX_CANDIDATES = 60
        if len(candidates) > _VLM_MAX_CANDIDATES:
            candidates = self._time_distributed_sample(
                candidates, _VLM_MAX_CANDIDATES,
            )

        if self._vlm_enabled and candidates:
            vlm_available = self._vlm_verifier.is_available()
            if vlm_available:
                log.info("pipeline.phase3_vlm", candidates=len(candidates))

                def vlm_progress(p: float):
                    if progress_callback:
                        progress_callback(0.55 + p * 0.30)

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

                # Phase 3b: Goal inference — kickoff rescan (85% → 90%)
                if progress_callback:
                    progress_callback(0.85)
                verdicts = self._goal_inference(
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
            progress_callback(0.90)

        # ── Post-processing (90% → 100%) ──────────────────────────────
        events = self._deduplicate(events)
        events.sort(key=lambda e: e.timestamp_start)

        log.info("pipeline.complete",
                 total_events=len(events),
                 event_types=[e.event_type.value for e in events])

        if progress_callback:
            progress_callback(1.0)

        return events

    # ------------------------------------------------------------------
    # Internal — goal inference (Phase 3b)
    # ------------------------------------------------------------------

    _KICKOFF_RESCAN_MIN_GAP = 30.0    # Kickoff must be ≥30s after shot
    _KICKOFF_RESCAN_MAX_GAP = 180.0   # ... and ≤180s after shot
    _KICKOFF_DIRECT_PROBES = [60, 90, 120]  # Direct frame-extraction fallback
    _GOAL_DEDUP_WINDOW = 240.0        # Merge goals within 4 min

    def _goal_inference(
        self,
        verdicts: list[VLMVerdict],
        all_candidates: list[EventCandidate],
    ) -> list[VLMVerdict]:
        """Post-VLM goal inference using temporal shot→kickoff patterns.

        1. For each confirmed shot/save, rescan for a kickoff 30-180s later.
           If found → upgrade the shot to a goal.
        2. Deduplicate goals within 3 min of each other (sideline camera
           often produces consecutive "ball in net" spikes for one goal).
        3. Downgrade goals with weak evidence (ball disappeared, no net/
           celebration confirmation).
        """
        confirmed = [v for v in verdicts
                     if v.result == VerificationResult.CONFIRMED]

        shot_types = {EventType.SHOT_ON_TARGET, EventType.SHOT_STOP_DIVING}
        shots = [v for v in confirmed if v.event_type in shot_types]
        goals = [v for v in confirmed if v.event_type == EventType.GOAL]
        kickoffs = [v for v in confirmed if v.event_type == EventType.KICKOFF]

        vlm_times = {v.candidate.timestamp for v in verdicts}

        # ── Step 1: Find candidates to rescan for kickoffs ────────────
        rescan_map: dict[float, VLMVerdict] = {}  # candidate_ts → shot verdict
        rescan_candidates: list[EventCandidate] = []

        shots_without_rescan: list[VLMVerdict] = []

        for shot in shots:
            t = shot.candidate.timestamp
            # Already have a kickoff after this shot?
            has_ko = any(
                self._KICKOFF_RESCAN_MIN_GAP
                < ko.candidate.timestamp - t
                < self._KICKOFF_RESCAN_MAX_GAP
                for ko in kickoffs
            )
            if has_ko:
                continue

            # Find first unclassified motion candidate 30-180s later
            found = False
            for cand in sorted(all_candidates, key=lambda c: c.timestamp):
                gap = cand.timestamp - t
                if gap < self._KICKOFF_RESCAN_MIN_GAP:
                    continue
                if gap > self._KICKOFF_RESCAN_MAX_GAP:
                    break
                if cand.timestamp not in vlm_times:
                    rescan_map[cand.timestamp] = shot
                    rescan_candidates.append(cand)
                    found = True
                    break

            if not found:
                shots_without_rescan.append(shot)

        # For shots with no motion candidate in the window, create
        # synthetic probe candidates at fixed offsets (direct extraction)
        from src.detection.visual_candidate import VisualContext
        for shot in shots_without_rescan:
            t = shot.candidate.timestamp
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
                rescan_map[probe_t] = shot
                rescan_candidates.append(probe)

        if shots_without_rescan:
            log.info("goal_inference.direct_probes",
                     shots=len(shots_without_rescan),
                     probes=sum(1 for s in shots_without_rescan
                                for _ in self._KICKOFF_DIRECT_PROBES))

        # ── Step 2: Rescan candidates via focused kickoff prompt ──────
        new_kickoffs: list[VLMVerdict] = []
        if rescan_candidates:
            log.info("goal_inference.kickoff_rescan",
                     shots=len(shots), rescanning=len(rescan_candidates))

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
                 "for_shot_at": (rescan_map[rv.candidate.timestamp]
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

        # ── Step 3: Upgrade shots that have a kickoff after them ──────
        modified = list(verdicts)
        upgrades = 0

        for shot in shots:
            t = shot.candidate.timestamp
            for ko in all_kickoffs:
                gap = ko.candidate.timestamp - t
                if self._KICKOFF_RESCAN_MIN_GAP < gap < self._KICKOFF_RESCAN_MAX_GAP:
                    idx = modified.index(shot)
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
                    break

        # ── Step 4: Deduplicate goals within 3 min window ─────────────
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
                    # Keep the one with higher confidence
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

        # ── Step 5: Downgrade goals with weak evidence ────────────────
        _STRONG_EVIDENCE = ["inside the net", "in the net", "into the net"]
        _CELEBRATION = ["celebrat", "arms raised", "group hug"]
        downgrades = 0

        for v in list(modified):
            if (v.result != VerificationResult.CONFIRMED
                    or v.event_type != EventType.GOAL):
                continue
            reasoning_lower = v.reasoning.lower()
            has_net = any(kw in reasoning_lower for kw in _STRONG_EVIDENCE)
            has_celebration = any(kw in reasoning_lower for kw in _CELEBRATION)
            has_kickoff_after = any(
                self._KICKOFF_RESCAN_MIN_GAP
                < ko.candidate.timestamp - v.candidate.timestamp
                < self._KICKOFF_RESCAN_MAX_GAP
                for ko in all_kickoffs
            )
            # Weak = no net evidence AND no celebration AND no kickoff
            if not has_net and not has_celebration and not has_kickoff_after:
                idx = modified.index(v)
                modified[idx] = VLMVerdict(
                    candidate=v.candidate,
                    result=VerificationResult.CONFIRMED,
                    event_type=EventType.SHOT_ON_TARGET,
                    confidence=v.confidence * 0.7,
                    reasoning=f"DOWNGRADED: weak goal evidence. "
                              f"Original: {v.reasoning}",
                    model_used=v.model_used,
                )
                downgrades += 1
                t = v.candidate.timestamp
                log.info("goal_inference.downgrade",
                         time=f"{int(t//60):02d}:{t%60:05.2f}")

        log.info("goal_inference.complete",
                 upgrades=upgrades,
                 deduped=len(goals_to_remove),
                 downgrades=downgrades,
                 final_goals=sum(
                     1 for v in modified
                     if v.result == VerificationResult.CONFIRMED
                     and v.event_type == EventType.GOAL))

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

    def _verdicts_to_events(self, verdicts: list[VLMVerdict]) -> list[Event]:
        """Convert VLM verdicts to Event objects."""
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
            events.append(self._make_event(
                event_type=event_type,
                timestamp=c.timestamp,
                clip_start=c.clip_start,
                clip_end=c.clip_end,
                confidence=confidence,
                metadata={
                    "source": c.source.value,
                    "vlm_result": v.result.value,
                    "vlm_reasoning": v.reasoning,
                    "vlm_model": v.model_used,
                    "audio_cue": c.audio_cue.cue_type.value if c.audio_cue else None,
                    "motion_magnitude": c.context.motion_magnitude,
                    "spike_duration": c.context.spike_duration_sec,
                    "audio_boost": c.context.audio_boost,
                },
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

    def _deduplicate(
        self, events: list[Event], window_sec: float = 10.0,
    ) -> list[Event]:
        """Remove near-duplicate events within *window_sec* of each other."""
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e.timestamp_start)
        deduped: list[Event] = [sorted_events[0]]

        for e in sorted_events[1:]:
            prev = deduped[-1]
            if (
                e.event_type == prev.event_type
                and abs(e.timestamp_start - prev.timestamp_start) < window_sec
            ):
                # Keep higher-confidence one
                if e.confidence > prev.confidence:
                    deduped[-1] = e
            else:
                deduped.append(e)

        return deduped
