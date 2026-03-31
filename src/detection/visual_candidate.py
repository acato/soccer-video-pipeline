"""
Visual candidate generator (Phase 1 — motion-first).

Primary trigger is dense frame-differencing motion scan.  Audio cues
(when available) boost co-located motion candidates but never gate them.

Produces EventCandidate objects for downstream VLM classification.

Dependencies: FFmpeg (subprocess), opencv-python (frame differencing).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import structlog

from src.detection.audio_detector import AudioCandidate, AudioCueType

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class CandidateSource(str, Enum):
    """How this candidate was generated."""
    AUDIO_WHISTLE = "audio_whistle"
    AUDIO_SURGE = "audio_surge"
    AUDIO_BOTH = "audio_both"
    SPOT_CHECK = "spot_check"           # Periodic full-scan fallback
    MOTION_SPIKE = "motion_spike"       # Frame-difference motion detection


@dataclass(frozen=True)
class VisualContext:
    """What motion/YOLO analysis found at the candidate timestamp."""
    player_count: int = 0
    ball_detected: bool = False
    ball_position: Optional[tuple[float, float]] = None   # Normalised (x, y)
    gk_detected: bool = False
    gk_position: Optional[tuple[float, float]] = None
    motion_magnitude: float = 0.0       # Frame-diff motion score 0-1
    near_goal: bool = False             # Ball or action in goal-area zone
    spike_duration_sec: float = 0.0     # How long the motion spike lasted
    audio_boost: bool = False           # Audio cue co-located within window


@dataclass(frozen=True)
class EventCandidate:
    """A moment flagged for VLM verification."""
    timestamp: float                    # Seconds from video start
    source: CandidateSource
    confidence: float                   # 0-1 — how likely this is a real event
    context: VisualContext
    audio_cue: Optional[AudioCandidate] = None
    clip_start: float = 0.0            # Suggested clip window start
    clip_end: float = 0.0              # Suggested clip window end


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class VisualCandidateGenerator:
    """Generate event candidates via dense motion scanning.

    Motion-first architecture: dense frame-differencing finds high-
    activity moments.  Audio cues are an optional confidence booster,
    never a gate.

    Usage::

        gen = VisualCandidateGenerator("/path/to/match.mp4")
        candidates = gen.motion_scan(duration)
        # Optionally boost with audio:
        candidates = gen.boost_with_audio(candidates, audio_candidates)
    """

    # Clip window for VLM verification
    _CLIP_PRE_SEC = 5.0
    _CLIP_POST_SEC = 15.0

    # Motion scan defaults
    _SAMPLE_INTERVAL = 0.5      # Sample every 0.5s for dense coverage
    _SIGMA_THRESHOLD = 1.0      # 1.0σ — adaptive per-window, lower to catch saves
    _MERGE_WINDOW_SEC = 8.0     # Merge spikes within 8s of each other
    _AUDIO_BOOST_WINDOW = 5.0   # Audio cue within ±5s boosts motion candidate
    _ADAPTIVE_WINDOW_SEC = 600  # 10-minute sliding window for adaptive threshold
    _GLOBAL_FLOOR_RATIO = 0.80  # Floor at 80% of global mean (prevent halftime triggers)

    def __init__(
        self,
        source_file: str | Path,
        *,
        yolo_model_path: Optional[str] = None,
        use_gpu: bool = False,
        inference_size: int = 640,
        fps: float = 30.0,
        working_dir: Optional[str | Path] = None,
    ):
        self._source = Path(source_file)
        self._yolo_path = yolo_model_path
        self._use_gpu = use_gpu
        self._inf_size = inference_size
        self._fps = fps

        if working_dir:
            self._work = Path(working_dir)
        else:
            self._work = Path("/tmp/soccer-pipeline")
        self._work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def motion_scan(
        self,
        video_duration: float,
        sample_interval: Optional[float] = None,
        sigma_threshold: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[EventCandidate]:
        """Dense motion scan — primary candidate generator.

        Samples every *sample_interval* seconds, computes frame-diff
        motion magnitude, finds spikes above *sigma_threshold* σ.
        """
        import cv2

        interval = sample_interval or self._SAMPLE_INTERVAL
        sigma = sigma_threshold or self._SIGMA_THRESHOLD

        cap = cv2.VideoCapture(str(self._source))
        if not cap.isOpened():
            log.error("visual_candidate.cannot_open", file=str(self._source))
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or self._fps
        frame_step = max(1, int(fps * interval))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        prev_gray = None
        motion_scores: list[tuple[float, float]] = []  # (timestamp, score)

        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))  # Small for speed

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                score = float(np.mean(diff)) / 255.0
                t = frame_idx / fps
                motion_scores.append((t, score))

            prev_gray = gray
            frame_idx += frame_step

            if progress_callback and total_frames > 0:
                progress_callback(min(frame_idx / total_frames, 1.0))

        cap.release()

        if not motion_scores:
            return []

        import math

        # Compute global stats (for logging and floor)
        scores_arr = np.array([s for _, s in motion_scores])
        global_mean = float(np.mean(scores_arr))
        global_std = float(np.std(scores_arr))
        global_threshold = global_mean + sigma * global_std

        log.info("visual_candidate.motion_stats",
                 mean=round(global_mean, 5),
                 std=round(global_std, 5),
                 global_threshold=round(global_threshold, 5),
                 sigma=sigma,
                 samples=len(motion_scores),
                 mode="adaptive_window")

        # Build per-sample adaptive threshold via sliding window
        window_samples = int(self._ADAPTIVE_WINDOW_SEC / interval)
        threshold_floor = global_mean * self._GLOBAL_FLOOR_RATIO
        thresholds: list[float] = []

        for i in range(len(motion_scores)):
            # Window centred on current sample
            half_w = window_samples // 2
            w_start = max(0, i - half_w)
            w_end = min(len(motion_scores), i + half_w + 1)
            window = scores_arr[w_start:w_end]
            local_mean = float(np.mean(window))
            local_std = float(np.std(window))
            local_thresh = local_mean + sigma * local_std
            # Floor: never below 80% of global mean (prevents halftime triggers)
            thresholds.append(max(local_thresh, threshold_floor))

        # Find spike regions using per-sample adaptive threshold
        results: list[EventCandidate] = []
        in_spike = False
        spike_start = 0.0
        spike_max_score = 0.0
        spike_max_t = 0.0
        spike_threshold = 0.0  # threshold at the spike peak

        for idx, (t, score) in enumerate(motion_scores):
            thresh = thresholds[idx]
            if score > thresh and not in_spike:
                in_spike = True
                spike_start = t
                spike_max_score = score
                spike_max_t = t
                spike_threshold = thresh
            elif score > thresh and in_spike:
                if score > spike_max_score:
                    spike_max_score = score
                    spike_max_t = t
                    spike_threshold = thresh
            elif score <= thresh and in_spike:
                in_spike = False
                spike_dur = t - spike_start
                clip_start = max(0, spike_max_t - self._CLIP_PRE_SEC)
                clip_end = min(video_duration, spike_max_t + self._CLIP_POST_SEC)

                ratio = spike_max_score / max(spike_threshold, 1e-8)
                raw_conf = min(0.3 + 0.35 * math.log2(max(ratio, 1.0)), 0.85)
                dur_bonus = min(spike_dur / 10.0, 0.15)
                confidence = min(raw_conf + dur_bonus, 1.0)

                results.append(EventCandidate(
                    timestamp=spike_max_t,
                    source=CandidateSource.MOTION_SPIKE,
                    confidence=confidence,
                    context=VisualContext(
                        motion_magnitude=spike_max_score,
                        spike_duration_sec=spike_dur,
                    ),
                    clip_start=clip_start,
                    clip_end=clip_end,
                ))

        # Handle spike at end of video
        if in_spike:
            t_last = motion_scores[-1][0]
            spike_dur = t_last - spike_start
            clip_start = max(0, spike_max_t - self._CLIP_PRE_SEC)
            clip_end = min(video_duration, spike_max_t + self._CLIP_POST_SEC)
            ratio = spike_max_score / max(spike_threshold, 1e-8)
            raw_conf = min(0.3 + 0.35 * math.log2(max(ratio, 1.0)), 0.85)
            dur_bonus = min(spike_dur / 10.0, 0.15)
            confidence = min(raw_conf + dur_bonus, 1.0)
            results.append(EventCandidate(
                timestamp=spike_max_t,
                source=CandidateSource.MOTION_SPIKE,
                confidence=confidence,
                context=VisualContext(
                    motion_magnitude=spike_max_score,
                    spike_duration_sec=spike_dur,
                ),
                clip_start=clip_start,
                clip_end=clip_end,
            ))

        # Merge candidates within merge window
        results = self._merge_motion_candidates(results)

        log.info("visual_candidate.motion_scan",
                 samples=len(motion_scores), spikes=len(results),
                 global_threshold=round(global_threshold, 5))
        return results

    def boost_with_audio(
        self,
        candidates: list[EventCandidate],
        audio_candidates: list[AudioCandidate],
    ) -> list[EventCandidate]:
        """Boost motion candidates that co-locate with audio cues.

        Audio never gates — it only increases confidence of existing
        motion candidates.  Whistles boost by +0.15, surges by +0.10.
        Also tags candidates with matching audio for VLM context.
        """
        if not audio_candidates or not candidates:
            return candidates

        boosted: list[EventCandidate] = []
        for c in candidates:
            best_audio: Optional[AudioCandidate] = None
            best_dist = float("inf")

            for ac in audio_candidates:
                dist = abs(c.timestamp - ac.timestamp)
                if dist < self._AUDIO_BOOST_WINDOW and dist < best_dist:
                    best_dist = dist
                    best_audio = ac

            if best_audio is not None:
                # Determine boost amount
                is_whistle = best_audio.cue_type in (
                    AudioCueType.WHISTLE, AudioCueType.WHISTLE_AND_SURGE,
                )
                boost = 0.15 if is_whistle else 0.10

                source = {
                    AudioCueType.WHISTLE: CandidateSource.AUDIO_WHISTLE,
                    AudioCueType.ENERGY_SURGE: CandidateSource.AUDIO_SURGE,
                    AudioCueType.WHISTLE_AND_SURGE: CandidateSource.AUDIO_BOTH,
                }[best_audio.cue_type]

                new_ctx = VisualContext(
                    motion_magnitude=c.context.motion_magnitude,
                    spike_duration_sec=c.context.spike_duration_sec,
                    audio_boost=True,
                )
                boosted.append(EventCandidate(
                    timestamp=c.timestamp,
                    source=source,  # Upgrade source to reflect audio
                    confidence=min(c.confidence + boost, 1.0),
                    context=new_ctx,
                    audio_cue=best_audio,
                    clip_start=c.clip_start,
                    clip_end=c.clip_end,
                ))

                mm = int(c.timestamp // 60)
                ss = c.timestamp % 60
                log.debug("visual_candidate.audio_boost",
                          time=f"{mm:02d}:{ss:05.2f}",
                          cue=best_audio.cue_type.value,
                          boost=boost,
                          new_conf=round(min(c.confidence + boost, 1.0), 3))
            else:
                boosted.append(c)

        audio_boosted = sum(1 for c in boosted if c.context.audio_boost)
        log.info("visual_candidate.audio_boost_done",
                 total=len(boosted), boosted=audio_boosted)
        return boosted

    # ------------------------------------------------------------------
    # Internal — merging
    # ------------------------------------------------------------------

    def _merge_motion_candidates(
        self, candidates: list[EventCandidate],
    ) -> list[EventCandidate]:
        """Merge motion candidates within _MERGE_WINDOW_SEC of each other."""
        if len(candidates) <= 1:
            return candidates

        sorted_c = sorted(candidates, key=lambda c: c.timestamp)
        merged: list[EventCandidate] = [sorted_c[0]]

        for c in sorted_c[1:]:
            if c.timestamp - merged[-1].timestamp < self._MERGE_WINDOW_SEC:
                # Keep the higher-confidence one, take widest clip window
                if c.confidence > merged[-1].confidence:
                    prev = merged[-1]
                    merged[-1] = EventCandidate(
                        timestamp=c.timestamp,
                        source=c.source,
                        confidence=c.confidence,
                        context=c.context,
                        audio_cue=c.audio_cue,
                        clip_start=min(c.clip_start, prev.clip_start),
                        clip_end=max(c.clip_end, prev.clip_end),
                    )
                else:
                    prev = merged[-1]
                    merged[-1] = EventCandidate(
                        timestamp=prev.timestamp,
                        source=prev.source,
                        confidence=prev.confidence,
                        context=prev.context,
                        audio_cue=prev.audio_cue,
                        clip_start=min(prev.clip_start, c.clip_start),
                        clip_end=max(prev.clip_end, c.clip_end),
                    )
            else:
                merged.append(c)

        return merged
