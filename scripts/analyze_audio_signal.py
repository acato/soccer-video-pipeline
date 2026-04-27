#!/usr/bin/env python3
"""QL2 Phase A: do soccer audio events align with GT event timestamps?

Heuristic whistle detector via spectral bandpass (2.5–4.5 kHz, the dominant
fundamental band of pea/coach/referee whistles). For each detected whistle,
compute alignment with each GT event type — if a meaningful fraction of GT
events have a whistle within ±5s, the signal is real and worth fusing.

Usage:
    python analyze_audio_signal.py <video_path> --gt-file <h1> --gt-file <h2> \
        [--video-offset 418.0 --half2-start 3916.0 --half2-game-offset 2700.0]

If --gt-file is omitted, falls back to evaluate_detection's defaults (Rush).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def extract_mono_audio(video_path: str, sr: int = 22050) -> str:
    """ffmpeg-extract a mono float32 WAV at sr Hz."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", video_path, "-ac", "1", "-ar", str(sr), "-y", out],
        check=True,
    )
    return out


def detect_whistles(audio_path: str, sr: int = 22050) -> list[dict]:
    """Return list of {start, end, peak_energy, mean_energy} for whistle bursts.

    Whistle detector:
      - STFT (n_fft=2048, hop=512) -> ~46 ms time resolution
      - Bandpass energy 2500–4500 Hz (referee whistle fundamental)
      - Threshold: mean energy >= 95th percentile of bandpass energy
      - Group consecutive frames within 0.2 s into bursts
      - Filter bursts shorter than 0.15 s (transient noise) or longer than
        2.0 s (sustained tones, not pea whistles)
    """
    import librosa
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= 2500) & (freqs <= 4500)
    band_energy = S[band, :].mean(axis=0)

    threshold = np.percentile(band_energy, 95.0)
    above = band_energy >= threshold
    times = librosa.frames_to_time(np.arange(len(band_energy)),
                                    sr=sr, hop_length=512)

    bursts = []
    in_burst = False
    burst_start_idx = -1
    last_above_idx = -1
    gap_frames = int(0.2 * sr / 512)  # bridge ≤200ms gaps

    for i, a in enumerate(above):
        if a:
            if not in_burst:
                in_burst = True
                burst_start_idx = i
            last_above_idx = i
        else:
            if in_burst and (i - last_above_idx) > gap_frames:
                # Burst ended
                start_t = times[burst_start_idx]
                end_t = times[last_above_idx]
                dur = end_t - start_t
                if 0.15 <= dur <= 2.0:
                    seg = band_energy[burst_start_idx:last_above_idx + 1]
                    bursts.append({
                        "start": float(start_t),
                        "end": float(end_t),
                        "duration": float(dur),
                        "peak_energy": float(seg.max()),
                        "mean_energy": float(seg.mean()),
                    })
                in_burst = False

    # Close any open burst at end
    if in_burst:
        start_t = times[burst_start_idx]
        end_t = times[last_above_idx]
        dur = end_t - start_t
        if 0.15 <= dur <= 2.0:
            seg = band_energy[burst_start_idx:last_above_idx + 1]
            bursts.append({
                "start": float(start_t),
                "end": float(end_t),
                "duration": float(dur),
                "peak_energy": float(seg.max()),
                "mean_energy": float(seg.mean()),
            })

    return bursts


def alignment_stats(bursts: list[dict], gt_events: list,
                    tolerance_sec: float = 5.0) -> dict:
    """For each GT event type, compute % of GT events with a whistle within ±tolerance_sec."""
    by_type = defaultdict(list)
    for g in gt_events:
        by_type[g.event_type].append(g.video_time_sec)

    burst_centers = sorted(((b["start"] + b["end"]) / 2 for b in bursts))

    out = {}
    for etype, ts_list in by_type.items():
        n = len(ts_list)
        n_aligned = 0
        for gt_t in ts_list:
            # binary search for nearest burst
            lo, hi = 0, len(burst_centers) - 1
            best_dist = float("inf")
            while lo <= hi:
                mid = (lo + hi) // 2
                bt = burst_centers[mid]
                d = abs(bt - gt_t)
                if d < best_dist:
                    best_dist = d
                if bt < gt_t:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_dist <= tolerance_sec:
                n_aligned += 1
        out[etype] = {
            "gt_count": n,
            "aligned": n_aligned,
            "alignment_pct": (100.0 * n_aligned / n) if n else 0.0,
        }
    return out


def baseline_alignment(bursts: list[dict], gt_events: list,
                       video_duration: float,
                       tolerance_sec: float = 5.0) -> dict:
    """Baseline: if whistles were uniformly random in time, what alignment would
    we see by chance? Computes expected alignment % as P(any whistle within ±tol)
    given the whistle rate.
    P_chance ≈ 1 - (1 - p_per_sec)^(2*tolerance_sec)
    where p_per_sec = n_bursts / video_duration approximated as Poisson rate."""
    rate = len(bursts) / max(1.0, video_duration)
    expected_pct = 100.0 * (1.0 - np.exp(-rate * 2 * tolerance_sec))
    return {"chance_aligned_pct": float(expected_pct),
            "whistle_rate_per_sec": float(rate),
            "n_bursts": len(bursts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--gt-file", action="append", default=None)
    ap.add_argument("--video-offset", type=float, default=418.0)
    ap.add_argument("--half2-start", type=float, default=3916.0)
    ap.add_argument("--half2-game-offset", type=float, default=2700.0)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--audio-out", default=None,
                    help="Optional: keep extracted WAV at this path for re-runs")
    args = ap.parse_args()

    print(f"Extracting audio from {args.video}...")
    if args.audio_out and Path(args.audio_out).exists():
        audio_path = args.audio_out
        print(f"  reusing existing audio: {audio_path}")
    else:
        audio_path = extract_mono_audio(args.video)
        if args.audio_out:
            import shutil
            shutil.copy(audio_path, args.audio_out)
        print(f"  audio extracted: {audio_path}")

    print("Detecting whistles (bandpass 2.5–4.5 kHz, 95th-percentile threshold)...")
    bursts = detect_whistles(audio_path)
    print(f"  detected {len(bursts)} whistle bursts")

    # Get video duration
    p = subprocess.run(
        ["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
         args.video], capture_output=True, text=True,
    )
    duration = float(p.stdout.strip())

    # Load GT
    from scripts import evaluate_detection
    if args.gt_file:
        evaluate_detection.GT_FILES = list(args.gt_file)
        evaluate_detection.VIDEO_OFFSET = args.video_offset
    gt_events = evaluate_detection.load_ground_truth(
        half2_video_start=args.half2_start,
        half2_game_offset=args.half2_game_offset,
    )
    print(f"  GT events: {len(gt_events)}")
    print()

    # Baseline (random whistle alignment by chance)
    baseline = baseline_alignment(bursts, gt_events, duration, args.tolerance)
    print(f"Whistle rate: {baseline['whistle_rate_per_sec']:.3f}/sec "
          f"({baseline['n_bursts']} bursts / {duration:.0f}s)")
    print(f"Random-chance alignment ±{args.tolerance:.0f}s: "
          f"{baseline['chance_aligned_pct']:.1f}%")
    print()

    # Per-type alignment
    stats = alignment_stats(bursts, gt_events, args.tolerance)
    print(f"Per-type whistle alignment (within ±{args.tolerance:.0f}s):")
    print(f"  {'type':<20} {'GT':>4} {'aligned':>8} {'pct':>7} {'lift_vs_chance':>14}")
    chance = baseline["chance_aligned_pct"]
    for etype in sorted(stats.keys()):
        s = stats[etype]
        lift = s["alignment_pct"] - chance
        print(f"  {etype:<20} {s['gt_count']:>4} {s['aligned']:>8} "
              f"{s['alignment_pct']:>6.1f}% {lift:>+13.1f}pp")


if __name__ == "__main__":
    main()
