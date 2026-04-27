#!/usr/bin/env python3
"""QL2 Phase A step 2: AudioSet-class alignment with GT events.

Replaces the crude 2.5–4.5kHz bandpass with PANN's Cnn14 model trained on
AudioSet (527 classes, same labels as YAMNet). Computes per-class detection
over 1s windows, then per-class alignment with GT events.

If specific classes (Whistle, Cheering, Applause, Crowd, Shout) align with
GT event types consistently across both Rush and sporting_ac, the audio
signal is real and a fusion model is worth building.

Usage:
    python analyze_audio_yamnet.py <video_path> --gt-file h1.json --gt-file h2.json \
        [--video-offset 418 --half2-start 3916 --half2-game-offset 2700] \
        [--audio-out /tmp/x.wav]
"""
import argparse
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Classes we care about for soccer events
INTERESTING = {
    402: "Whistle",
    66: "Cheering",
    67: "Applause",
    69: "Crowd",
    8: "Shout",
    53: "Walk, footsteps",
    13: "Children shouting",
}


def extract_mono(video_path: str, sr: int = 32000) -> str:
    """PANN's Cnn14 expects 32kHz mono."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", video_path, "-ac", "1", "-ar", str(sr), "-y", out],
        check=True,
    )
    return out


def yamnet_detect(audio_path: str, window_sec: float = 1.0,
                  hop_sec: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Run Cnn14 over sliding windows. Returns (timestamps, scores [T, 527])."""
    import librosa
    from panns_inference import AudioTagging

    sr = 32000
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    n_windows = (len(y) - win) // hop + 1
    print(f"  audio {len(y) / sr:.1f}s, {n_windows} windows of {window_sec}s @ {hop_sec}s hop")

    at = AudioTagging(checkpoint_path=str(Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"),
                      device="cuda" if False else "cpu")  # CPU is fine; mac has MPS

    # Batch inference for speed
    batch = 32
    timestamps = np.array([i * hop / sr for i in range(n_windows)])
    scores = np.zeros((n_windows, 527), dtype=np.float32)

    for b0 in range(0, n_windows, batch):
        b1 = min(b0 + batch, n_windows)
        chunks = np.stack([y[i * hop : i * hop + win] for i in range(b0, b1)])
        clipwise, _ = at.inference(chunks)
        scores[b0:b1] = clipwise
        if (b0 // batch) % 10 == 0:
            print(f"    {b0}/{n_windows} windows...", end="\r")
    print(f"    done: {n_windows} windows                ")
    return timestamps, scores


def event_times_for_class(timestamps: np.ndarray, scores: np.ndarray,
                          class_idx: int, threshold: float = 0.3,
                          merge_gap_sec: float = 1.0) -> list[float]:
    """Return centers of contiguous frames where class_idx >= threshold."""
    above = scores[:, class_idx] >= threshold
    centers = []
    in_event = False
    start_t = 0.0
    last_t = 0.0
    for t, a in zip(timestamps, above):
        if a:
            if not in_event:
                start_t = float(t)
                in_event = True
            last_t = float(t)
        else:
            if in_event and (t - last_t) > merge_gap_sec:
                centers.append((start_t + last_t) / 2)
                in_event = False
    if in_event:
        centers.append((start_t + last_t) / 2)
    return centers


def alignment_pct(detections: list[float], gt_times: list[float],
                  tolerance: float) -> tuple[int, int]:
    """Return (n_aligned, n_gt) — GT events with any detection within ±tolerance."""
    if not detections:
        return 0, len(gt_times)
    det = sorted(detections)
    n_aligned = 0
    for gt_t in gt_times:
        # Binary search nearest
        lo, hi = 0, len(det) - 1
        best = float("inf")
        while lo <= hi:
            mid = (lo + hi) // 2
            d = abs(det[mid] - gt_t)
            if d < best: best = d
            if det[mid] < gt_t: lo = mid + 1
            else: hi = mid - 1
        if best <= tolerance:
            n_aligned += 1
    return n_aligned, len(gt_times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--gt-file", action="append", default=None)
    ap.add_argument("--video-offset", type=float, default=418.0)
    ap.add_argument("--half2-start", type=float, default=3916.0)
    ap.add_argument("--half2-game-offset", type=float, default=2700.0)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--audio-out", default=None)
    ap.add_argument("--scores-out", default=None,
                    help="Save (timestamps, scores) npz for re-runs")
    ap.add_argument("--scores-in", default=None,
                    help="Reuse a previous scores npz")
    args = ap.parse_args()

    # Get audio + scores
    if args.scores_in and Path(args.scores_in).exists():
        print(f"Loading cached scores from {args.scores_in}")
        z = np.load(args.scores_in)
        timestamps = z["timestamps"]
        scores = z["scores"]
    else:
        if args.audio_out and Path(args.audio_out).exists():
            audio_path = args.audio_out
            print(f"Reusing audio at {audio_path}")
        else:
            print(f"Extracting audio from {args.video}...")
            audio_path = extract_mono(args.video)
            if args.audio_out:
                import shutil; shutil.copy(audio_path, args.audio_out)
        print("Running PANN Cnn14 inference...")
        timestamps, scores = yamnet_detect(audio_path)
        if args.scores_out:
            np.savez_compressed(args.scores_out, timestamps=timestamps, scores=scores)
            print(f"Saved scores to {args.scores_out}")

    # Get video duration for chance baseline
    p = subprocess.run(["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                        args.video], capture_output=True, text=True)
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

    gt_by_type = defaultdict(list)
    for g in gt_events:
        gt_by_type[g.event_type].append(g.video_time_sec)

    # Per-class detection rate + per-(class, gt-type) alignment
    print("\nClass detection counts (threshold=%.2f):" % args.threshold)
    for cidx, cname in INTERESTING.items():
        dets = event_times_for_class(timestamps, scores, cidx, args.threshold)
        rate = len(dets) / max(1.0, duration)
        chance_pct = 100.0 * (1.0 - np.exp(-rate * 2 * args.tolerance))
        print(f"  [{cidx}] {cname:<20s}: {len(dets):>4} events  ({rate:.3f}/s, chance ±{args.tolerance:.0f}s = {chance_pct:.1f}%)")

    # Per-(audio class, gt type) alignment matrix
    print(f"\nAlignment matrix: % of GT events with audio detection within ±{args.tolerance:.0f}s")
    headers = ["GT type"] + [INTERESTING[c] for c in INTERESTING]
    col_w = 10
    print(f"  {'GT type':<20s}" + "".join(f"{h:>{col_w+2}s}" for h in headers[1:]))
    for etype in sorted(gt_by_type.keys()):
        gt_t = gt_by_type[etype]
        n = len(gt_t)
        row = [f"{etype}({n})".ljust(20)]
        for cidx in INTERESTING:
            dets = event_times_for_class(timestamps, scores, cidx, args.threshold)
            aligned, _ = alignment_pct(dets, gt_t, args.tolerance)
            pct = 100 * aligned / n if n else 0
            row.append(f"{pct:>{col_w}.0f}%")
        print(f"  {''.join(row)}")

    # Lift vs chance (helpful summary)
    print(f"\nLift vs chance (alignment% − random-chance%) — strong positives = signal:")
    print(f"  {'GT type':<20s}" + "".join(f"{INTERESTING[c]:>{col_w+2}s}" for c in INTERESTING))
    for etype in sorted(gt_by_type.keys()):
        gt_t = gt_by_type[etype]
        n = len(gt_t)
        row = [f"{etype}({n})".ljust(20)]
        for cidx in INTERESTING:
            dets = event_times_for_class(timestamps, scores, cidx, args.threshold)
            rate = len(dets) / max(1.0, duration)
            chance = 100.0 * (1.0 - np.exp(-rate * 2 * args.tolerance))
            aligned, _ = alignment_pct(dets, gt_t, args.tolerance)
            pct = 100 * aligned / n if n else 0
            lift = pct - chance
            row.append(f"{lift:>+{col_w}.0f}pp")
        print(f"  {''.join(row)}")


if __name__ == "__main__":
    main()
