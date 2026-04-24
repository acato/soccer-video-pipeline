#!/usr/bin/env python3
"""Smoke-test the field-crop algorithm on both games.

Computes the per-video field bbox for sporting_ac and Rush, saves
an original + cropped sample frame for each so we can visually
verify the bbox excludes non-pitch content without tightening into
the field.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.dual_pass_detector import DualPassConfig, DualPassDetector

OUT = Path("/tmp/field_crop_test")
OUT.mkdir(parents=True, exist_ok=True)

GAMES = [
    ("sporting", "/Users/aless/soccer-working/2025-01-01-ga-vs-sporting-ac.mp4", [428.2, 552.9, 2157.7]),
    ("rush",     "/Users/aless/soccer-working/2026-02-07 - Rush - GA2008.mp4",   [213.5, 303.5, 4000.0]),
]


def main():
    for name, video, samples in GAMES:
        print(f"\n=== {name} ===")
        cfg = DualPassConfig(field_crop_enabled=True, frame_width=1280)
        # Mock duration — we just need it for sampling stride
        import subprocess, json
        p = subprocess.run(
            ["/opt/homebrew/bin/ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video],
            capture_output=True, text=True,
        )
        duration = float(p.stdout.strip())
        print(f"  duration: {duration:.1f}s")

        det = DualPassDetector(
            config=cfg,
            source_file=video,
            video_duration=duration,
            job_id=f"test_{name}",
            working_dir="/tmp",
        )

        bbox = det._compute_field_bbox()
        if bbox is None:
            print(f"  FIELD DETECTION FAILED")
            continue
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        print(f"  bbox (normalized): x=[{x1:.3f}, {x2:.3f}]  y=[{y1:.3f}, {y2:.3f}]  area_frac={area:.3f}")

        # Sample 3 frames and crop
        from src.detection.frame_sampler import FrameSampler, SampledFrame
        sampler = FrameSampler(video, frame_width=1280)
        for t in samples:
            jpeg = sampler._extract_single_frame(t)
            if not jpeg:
                print(f"    t={t}: extract failed")
                continue
            orig = OUT / f"{name}_t{t}_orig.jpg"
            orig.write_bytes(jpeg)
            sf = SampledFrame(timestamp_sec=t, jpeg_bytes=jpeg)
            out_sf = det._field_crop_frames([sf])[0]
            crop = OUT / f"{name}_t{t}_fieldcrop.jpg"
            crop.write_bytes(out_sf.jpeg_bytes)
            ok = out_sf.jpeg_bytes != jpeg
            print(f"    t={t:.1f}  orig={len(jpeg)//1024}KB  crop={len(out_sf.jpeg_bytes)//1024}KB  cropped={ok}")

        print(f"  stats: {det._field_crop_stats}")

    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
