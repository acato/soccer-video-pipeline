"""QL2 Mode B + C: audio-based goal fusion.

Phase A measurement (scripts/analyze_audio_yamnet.py) showed that ~50% of
GT goals across our two test games have a detectable AudioSet-class peak
in the celebration cluster (Cheering/Applause/Crowd/Shout/Childshout) or
referee Whistle. The other ~50% are silent (no crowd, muted commentary,
or recording style without ambient pickup).

Two-mode fusion:
  - Mode B (recall booster): for each shot_on_target candidate, if an
    audio celebration peak exists in [shot.start, shot.start+25s] AND no
    goal already covers that interval, promote the shot to a goal candidate.
  - Mode C (confidence annotation): attach audio_celebration_score to
    every goal event so downstream (e.g. highlights reel composer) can
    rank goals by audio strength.

Mode A (silent-goal precision filter) is NOT implemented — the 50% of
goals without audio would be discarded, which is recall-destructive.

Audio scoring is expensive (~20 min CPU per game). Scores are cached by
video SHA at WORKING_DIR/audio_cache/{sha}.npz so subsequent runs of the
same video reuse them.
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)


# AudioSet class indices for celebration / referee signals.
# Phase A showed Rush goals fire on Cheering/Applause/Crowd/Shout/Childshout;
# sporting goals fire on Whistle (different recording style).
_CELEBRATION_CLASSES = {
    66: "Cheering",
    67: "Applause",
    69: "Crowd",
    8: "Shout",
    13: "Children shouting",
    402: "Whistle",
}


@dataclass
class _AudioData:
    timestamps: np.ndarray  # (T,) seconds
    scores: np.ndarray      # (T, 527) PANN/AudioSet softmax scores
    p99_per_class: dict[int, float]


_audio_cache_mem: dict[str, _AudioData] = {}


def _compute_video_sha(video_path: str) -> str:
    """SHA-1 of first 1MB + size for fast cache key (full SHA is overkill for cache)."""
    p = Path(video_path).resolve()
    h = hashlib.sha1()
    h.update(str(p).encode())
    h.update(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


def _ensure_audio_extracted(video_path: str, sr: int = 32000) -> str:
    """Extract mono PCM at sr Hz to a temp WAV. Caller deletes."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", video_path, "-ac", "1", "-ar", str(sr), "-y", out],
        check=True,
    )
    return out


def _run_pann_inference(audio_path: str, window_sec: float = 1.0,
                       hop_sec: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Run PANN's Cnn14 over sliding windows. Returns (timestamps, [T, 527])."""
    import librosa
    from panns_inference import AudioTagging

    sr = 32000
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    n_windows = (len(y) - win) // hop + 1

    at = AudioTagging(
        checkpoint_path=str(Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"),
        device="cpu",
    )

    batch = 32
    timestamps = np.array([i * hop / sr for i in range(n_windows)],
                           dtype=np.float32)
    scores = np.zeros((n_windows, 527), dtype=np.float32)

    for b0 in range(0, n_windows, batch):
        b1 = min(b0 + batch, n_windows)
        chunks = np.stack([y[i * hop : i * hop + win] for i in range(b0, b1)])
        clipwise, _ = at.inference(chunks)
        scores[b0:b1] = clipwise

    return timestamps, scores


def get_or_compute_audio_data(
    video_path: str,
    cache_dir: Optional[Path] = None,
) -> _AudioData:
    """Lazy-compute (or load from cache) per-video audio scores.

    Cache key: SHA-1 prefix of (path, size, first 1MB). Cached as npz at
    cache_dir/{sha}.npz. In-memory cache also kept across calls in the
    same process.
    """
    sha = _compute_video_sha(video_path)
    if sha in _audio_cache_mem:
        return _audio_cache_mem[sha]

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{sha}.npz"

    if cache_path is not None and cache_path.exists():
        log.info("audio_fusion.cache_hit", sha=sha, path=str(cache_path))
        z = np.load(str(cache_path))
        timestamps = z["timestamps"]
        scores = z["scores"]
    else:
        log.info("audio_fusion.compute_start", sha=sha, video=video_path)
        wav = _ensure_audio_extracted(video_path)
        try:
            timestamps, scores = _run_pann_inference(wav)
        finally:
            try:
                Path(wav).unlink()
            except Exception:
                pass
        log.info("audio_fusion.compute_done",
                 sha=sha, n_windows=int(len(timestamps)))
        if cache_path is not None:
            np.savez_compressed(str(cache_path), timestamps=timestamps, scores=scores)
            log.info("audio_fusion.cache_write", path=str(cache_path))

    p99 = {c: float(np.percentile(scores[:, c], 99))
           for c in _CELEBRATION_CLASSES}
    data = _AudioData(timestamps=timestamps, scores=scores, p99_per_class=p99)
    _audio_cache_mem[sha] = data
    return data


def compute_celebration_score(
    data: _AudioData, t_start: float, t_end: float,
) -> tuple[float, str]:
    """Max (window_max / global_p99) across celebration classes within [t_start, t_end].

    Returns (score, dominant_class_name). Score >= 1.0 = at-or-above-99th
    percentile. Score >= 2.0 = clearly above ambient.
    """
    win = (data.timestamps >= t_start) & (data.timestamps <= t_end)
    if not win.any():
        return 0.0, ""
    best_score = 0.0
    best_name = ""
    for cidx, cname in _CELEBRATION_CLASSES.items():
        win_max = float(data.scores[win, cidx].max())
        p99 = data.p99_per_class[cidx]
        ratio = win_max / max(1e-9, p99)
        if ratio > best_score:
            best_score = ratio
            best_name = cname
    return best_score, best_name


def apply_audio_fusion(
    events: list,
    video_path: str,
    cache_dir: Optional[Path] = None,
    promotion_threshold: float = 2.5,
    promotion_lookahead_sec: float = 25.0,
    job_id: Optional[str] = None,
) -> tuple[list, dict]:
    """Run Mode B (shot+celebration → goal promotion) and Mode C (annotate
    goals with audio_celebration_score).

    Args:
        events: detector output, list of Event with event_type / timestamp_start
            / timestamp_end / confidence / metadata.
        video_path: source video for audio extraction.
        cache_dir: where to cache PANN scores per video SHA.
        promotion_threshold: minimum audio celebration score (ratio over global
            99th percentile) to promote a shot to a goal.
        promotion_lookahead_sec: window after shot.start to scan for audio peak.
        job_id: for log correlation.

    Returns:
        (updated_events, stats_dict)
    """
    from src.detection.models import Event, EventType  # local import to avoid cycle

    data = get_or_compute_audio_data(video_path, cache_dir=cache_dir)

    stats = {
        "goals_annotated": 0,
        "goals_with_audio": 0,
        "shots_examined": 0,
        "shots_promoted": 0,
        "promotion_blocked_existing_goal": 0,
    }

    # Mode C: annotate every existing goal
    for e in events:
        etype = e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type)
        if etype != "goal":
            continue
        score, dom = compute_celebration_score(
            data, e.timestamp_start, e.timestamp_start + promotion_lookahead_sec
        )
        if e.metadata is None:
            e.metadata = {}
        e.metadata["audio_celebration_score"] = round(score, 3)
        e.metadata["audio_dominant_class"] = dom
        stats["goals_annotated"] += 1
        if score >= promotion_threshold:
            stats["goals_with_audio"] += 1

    # Mode B: scan shot_on_target → promote to goal if audio peak nearby
    # Compute existing goal coverage intervals for collision check.
    existing_goal_intervals = []
    for e in events:
        etype = e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type)
        if etype == "goal":
            existing_goal_intervals.append(
                (e.timestamp_start - 5.0, e.timestamp_end + 25.0)
            )

    new_goals = []
    for e in events:
        etype = e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type)
        if etype != "shot_on_target":
            continue
        stats["shots_examined"] += 1
        score, dom = compute_celebration_score(
            data,
            e.timestamp_start,
            e.timestamp_start + promotion_lookahead_sec,
        )
        if score < promotion_threshold:
            continue
        # Collision check — already a goal here?
        if any(s <= e.timestamp_start <= en for s, en in existing_goal_intervals):
            stats["promotion_blocked_existing_goal"] += 1
            continue

        # Promote: clone the shot's spatial info, change type to goal,
        # keep a reasonable end time covering shot + reaction window
        new_goal = Event(
            event_type=EventType.GOAL,
            timestamp_start=e.timestamp_start,
            timestamp_end=max(e.timestamp_end, e.timestamp_start + 15.0),
            confidence=min(0.9, 0.5 + score / 10.0),  # score-proportional
            metadata={
                "audio_celebration_score": round(score, 3),
                "audio_dominant_class": dom,
                "promoted_from_shot": True,
                "source_event_type": "shot_on_target",
                "vlm_reasoning": (
                    f"Audio fusion: shot_on_target at {e.timestamp_start:.1f}s "
                    f"with audio celebration peak ({dom}) score "
                    f"{score:.1f}× p99 baseline."
                ),
            },
        )
        new_goals.append(new_goal)
        # Add this new goal to coverage so we don't double-promote overlapping shots
        existing_goal_intervals.append(
            (new_goal.timestamp_start - 5.0, new_goal.timestamp_end + 25.0)
        )
        stats["shots_promoted"] += 1

    log.info("audio_fusion.summary",
             job_id=job_id,
             **stats,
             total_events_in=len(events),
             new_goals=len(new_goals))

    return events + new_goals, stats
