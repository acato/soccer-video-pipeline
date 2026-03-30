"""
Audio-based event candidate detection.

Extracts the audio track via FFmpeg, runs two parallel analyses:
  1. Whistle detection — bandpass 2-4 kHz, onset detection, min 200 ms
  2. Energy surge    — RMS rolling window, peaks > N std-dev above mean

Output: a timestamped list of AudioCandidate objects for downstream
visual confirmation.  Runs on CPU in < 60 s for a full 90-minute match.

When a video has no audio stream the detector returns an empty list
(fail-open).

Dependencies: FFmpeg (subprocess), librosa, numpy, soundfile.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class AudioCueType(str, Enum):
    WHISTLE = "whistle"
    ENERGY_SURGE = "energy_surge"
    WHISTLE_AND_SURGE = "whistle_and_surge"


@dataclass(frozen=True)
class AudioCandidate:
    """A moment in the match flagged by audio analysis."""
    timestamp: float                       # Seconds from video start
    cue_type: AudioCueType
    duration_sec: float                    # How long the cue lasted
    amplitude: float                       # Normalised peak amplitude 0-1
    frequency_hz: Optional[float] = None   # Dominant freq (whistles only)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class AudioDetector:
    """Detect referee whistles and crowd energy surges from match audio.

    Usage::

        detector = AudioDetector("/path/to/match.mp4")
        candidates = detector.detect()  # [] if no audio
    """

    def __init__(
        self,
        source_file: str | Path,
        *,
        bandpass_low_hz: int = 2000,
        bandpass_high_hz: int = 4000,
        min_whistle_sec: float = 0.2,
        rms_window_sec: float = 2.0,
        surge_stddev_threshold: float = 2.0,
        sample_rate: int = 22050,
        game_start_sec: float = 0.0,
        working_dir: Optional[str | Path] = None,
    ):
        self._source_file = Path(source_file)
        self._bp_low = bandpass_low_hz
        self._bp_high = bandpass_high_hz
        self._min_whistle = min_whistle_sec
        self._rms_window = rms_window_sec
        self._surge_std = surge_stddev_threshold
        self._sr = sample_rate
        self._game_start = game_start_sec

        if working_dir:
            self._work = Path(working_dir)
        else:
            self._work = Path("/tmp/soccer-pipeline")
        self._work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_audio(self) -> bool:
        """Return True if the source video contains an audio stream."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(self._source_file),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return b"audio" in result.stdout
        except Exception:
            return False

    def detect(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> list[AudioCandidate]:
        """Extract audio, detect whistles and surges.

        Returns an empty list when the video has no audio stream or
        extraction fails (fail-open).
        """
        import librosa  # lazy import — not needed at module level

        if not self.has_audio():
            log.info("audio_detector.no_audio", file=str(self._source_file))
            return []

        # Step 1: extract audio via FFmpeg
        wav_path = self._extract_audio()
        if wav_path is None:
            return []

        if progress_callback:
            progress_callback(0.1)

        # Step 2: load with librosa
        try:
            audio, sr = librosa.load(str(wav_path), sr=self._sr, mono=True)
        except Exception as exc:
            log.error("audio_detector.load_failed", error=str(exc))
            return []

        if progress_callback:
            progress_callback(0.2)

        # Trim to game start
        start_sample = int(self._game_start * sr)
        if start_sample > 0 and start_sample < len(audio):
            audio_trimmed = audio[start_sample:]
            offset = self._game_start
        else:
            audio_trimmed = audio
            offset = 0.0

        # Step 3: parallel detection
        whistles = self._detect_whistles(audio_trimmed, sr, offset)
        if progress_callback:
            progress_callback(0.6)

        surges = self._detect_energy_surges(audio_trimmed, sr, offset)
        if progress_callback:
            progress_callback(0.8)

        # Step 4: merge co-occurring cues
        merged = self._merge_cues(whistles, surges)

        log.info("audio_detector.complete",
                 whistles=len(whistles), surges=len(surges),
                 merged=len(merged))

        if progress_callback:
            progress_callback(1.0)

        # Clean up temp file
        wav_path.unlink(missing_ok=True)

        return merged

    # ------------------------------------------------------------------
    # Internal — extraction
    # ------------------------------------------------------------------

    def _extract_audio(self) -> Optional[Path]:
        """Extract audio to mono WAV via FFmpeg."""
        wav_path = self._work / f"audio_{self._source_file.stem}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self._source_file),
            "-vn",                        # no video
            "-ac", "1",                   # mono
            "-ar", str(self._sr),         # target sample rate
            "-acodec", "pcm_s16le",       # 16-bit PCM
            str(wav_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0 and wav_path.exists():
                log.info("audio_detector.extracted",
                         path=str(wav_path),
                         size_mb=wav_path.stat().st_size / 1e6)
                return wav_path
            log.warning("audio_detector.ffmpeg_failed",
                        returncode=result.returncode,
                        stderr=(result.stderr or b"")[-300:].decode(errors="replace"))
            return None
        except subprocess.TimeoutExpired:
            log.warning("audio_detector.ffmpeg_timeout")
            return None

    # ------------------------------------------------------------------
    # Internal — whistle detection
    # ------------------------------------------------------------------

    def _detect_whistles(
        self,
        audio: np.ndarray,
        sr: int,
        time_offset: float,
    ) -> list[AudioCandidate]:
        """Bandpass 2-4 kHz → onset detection → group into whistle events."""
        import librosa

        # Bandpass filter via STFT: zero out bins outside 2-4 kHz
        n_fft = 2048
        hop = 512
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Create bandpass mask
        mask = np.zeros(len(freqs))
        band_idx = np.where((freqs >= self._bp_low) & (freqs <= self._bp_high))[0]
        mask[band_idx] = 1.0

        # Apply mask and compute energy in band
        band_energy = np.sum(np.abs(stft[band_idx, :]) ** 2, axis=0)

        # Normalise
        if band_energy.max() > 0:
            band_energy_norm = band_energy / band_energy.max()
        else:
            return []

        # Threshold: frames where band energy is above a dynamic threshold
        median_energy = np.median(band_energy_norm)
        std_energy = np.std(band_energy_norm)
        threshold = median_energy + 3.0 * std_energy  # conservative

        active = band_energy_norm > threshold

        # Group consecutive active frames into whistle events
        whistles: list[AudioCandidate] = []
        times = librosa.frames_to_time(
            np.arange(len(band_energy)), sr=sr, hop_length=hop,
        )

        in_whistle = False
        start_idx = 0

        for i, is_active in enumerate(active):
            if is_active and not in_whistle:
                start_idx = i
                in_whistle = True
            elif not is_active and in_whistle:
                duration = times[i] - times[start_idx]
                if duration >= self._min_whistle:
                    peak_amp = float(band_energy_norm[start_idx:i].max())
                    center_time = (times[start_idx] + times[i]) / 2.0 + time_offset
                    # Estimate dominant frequency in the band
                    segment = np.abs(stft[band_idx, start_idx:i])
                    dom_bin = band_idx[segment.sum(axis=1).argmax()]
                    dom_freq = float(freqs[dom_bin])

                    whistles.append(AudioCandidate(
                        timestamp=center_time,
                        cue_type=AudioCueType.WHISTLE,
                        duration_sec=duration,
                        amplitude=peak_amp,
                        frequency_hz=dom_freq,
                    ))
                in_whistle = False

        # Handle whistle at end of audio
        if in_whistle:
            duration = times[-1] - times[start_idx]
            if duration >= self._min_whistle:
                peak_amp = float(band_energy_norm[start_idx:].max())
                center_time = (times[start_idx] + times[-1]) / 2.0 + time_offset
                segment = np.abs(stft[band_idx, start_idx:])
                dom_bin = band_idx[segment.sum(axis=1).argmax()]
                dom_freq = float(freqs[dom_bin])

                whistles.append(AudioCandidate(
                    timestamp=center_time,
                    cue_type=AudioCueType.WHISTLE,
                    duration_sec=duration,
                    amplitude=peak_amp,
                    frequency_hz=dom_freq,
                ))

        log.info("audio_detector.whistles",
                 count=len(whistles), threshold=threshold)
        return whistles

    # ------------------------------------------------------------------
    # Internal — energy surge detection
    # ------------------------------------------------------------------

    def _detect_energy_surges(
        self,
        audio: np.ndarray,
        sr: int,
        time_offset: float,
    ) -> list[AudioCandidate]:
        """RMS rolling window, flag peaks > N std-dev above mean."""
        import librosa

        hop = 512
        rms = librosa.feature.rms(
            y=audio, frame_length=int(self._rms_window * sr),
            hop_length=hop,
        )[0]

        if len(rms) == 0:
            return []

        # Compute rolling statistics with a larger window for local baseline
        # Use 60-second windows for the baseline to handle varying crowd levels
        baseline_frames = int(60.0 * sr / hop)
        if baseline_frames < 1:
            baseline_frames = 1

        mean_rms = np.convolve(rms, np.ones(baseline_frames) / baseline_frames, mode="same")
        # Std dev with same window
        rms_sq = np.convolve(rms ** 2, np.ones(baseline_frames) / baseline_frames, mode="same")
        std_rms = np.sqrt(np.maximum(rms_sq - mean_rms ** 2, 0))
        std_rms = np.maximum(std_rms, 1e-8)  # avoid division by zero

        # Z-score
        z_scores = (rms - mean_rms) / std_rms

        # Find peaks above threshold
        surge_mask = z_scores > self._surge_std

        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

        # Group consecutive surge frames, merge within 1s
        surges: list[AudioCandidate] = []
        in_surge = False
        start_idx = 0

        for i, is_surge in enumerate(surge_mask):
            if is_surge and not in_surge:
                start_idx = i
                in_surge = True
            elif not is_surge and in_surge:
                duration = times[i] - times[start_idx]
                if duration >= 0.5:  # minimum 0.5s surge
                    peak_amp = float(rms[start_idx:i].max())
                    max_amp = float(rms.max()) if rms.max() > 0 else 1.0
                    norm_amp = peak_amp / max_amp
                    center_time = (times[start_idx] + times[i]) / 2.0 + time_offset

                    surges.append(AudioCandidate(
                        timestamp=center_time,
                        cue_type=AudioCueType.ENERGY_SURGE,
                        duration_sec=duration,
                        amplitude=norm_amp,
                    ))
                in_surge = False

        # Merge surges within 3s of each other
        if len(surges) > 1:
            merged: list[AudioCandidate] = [surges[0]]
            for s in surges[1:]:
                prev = merged[-1]
                if s.timestamp - prev.timestamp < 3.0:
                    # Merge: keep earlier timestamp, extend duration, max amplitude
                    merged[-1] = AudioCandidate(
                        timestamp=prev.timestamp,
                        cue_type=AudioCueType.ENERGY_SURGE,
                        duration_sec=(s.timestamp + s.duration_sec / 2)
                                     - (prev.timestamp - prev.duration_sec / 2),
                        amplitude=max(prev.amplitude, s.amplitude),
                    )
                else:
                    merged.append(s)
            surges = merged

        log.info("audio_detector.surges", count=len(surges))
        return surges

    # ------------------------------------------------------------------
    # Internal — merge co-occurring cues
    # ------------------------------------------------------------------

    def _merge_cues(
        self,
        whistles: list[AudioCandidate],
        surges: list[AudioCandidate],
        co_occur_window: float = 3.0,
    ) -> list[AudioCandidate]:
        """Merge co-occurring whistle + surge within *co_occur_window* seconds.

        Returns a unified, sorted, deduplicated candidate list.
        """
        merged: list[AudioCandidate] = []
        used_surges: set[int] = set()

        for w in whistles:
            matched = False
            for j, s in enumerate(surges):
                if j in used_surges:
                    continue
                if abs(w.timestamp - s.timestamp) <= co_occur_window:
                    merged.append(AudioCandidate(
                        timestamp=min(w.timestamp, s.timestamp),
                        cue_type=AudioCueType.WHISTLE_AND_SURGE,
                        duration_sec=max(w.duration_sec, s.duration_sec),
                        amplitude=max(w.amplitude, s.amplitude),
                        frequency_hz=w.frequency_hz,
                    ))
                    used_surges.add(j)
                    matched = True
                    break
            if not matched:
                merged.append(w)

        # Add unmatched surges
        for j, s in enumerate(surges):
            if j not in used_surges:
                merged.append(s)

        merged.sort(key=lambda c: c.timestamp)
        return merged
