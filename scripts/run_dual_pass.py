"""CLI wrapper for the dual-pass event detector.

Replaces the legacy Celery worker as the way to invoke the detector. Reads
configuration from environment variables (the same names the worker used);
writes events to <out_dir>/events.jsonl.

Usage:
  python scripts/run_dual_pass.py \\
      --video /path/to/match.mp4 \\
      --out-dir /tmp/soccer-pipeline/<job_id>/ \\
      --vllm-url http://inference-host:8000/v1 \\
      [--single-pass]

Required env or CLI flags:
  VLLM_URL                          OpenAI-compatible endpoint
  DUAL_PASS_TIER1_NAME (optional)   default: qwen3-vl-32b
  DUAL_PASS_TIER2_NAME (optional)   default: qwen3-vl-32b
  SINGLE_PASS_32B                   if "true", runs single-pass 32B (default in production)

The full set of env vars matches the legacy worker; see
docs/legacy/runbook_new_match.md for the canonical list, or inspect
src/detection/dual_pass_detector.py:DualPassConfig.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

# Allow running from repo root with no install step
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.dual_pass_detector import DualPassConfig, DualPassDetector


def _truthy(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes")


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def _float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v else default


def _csv_int(name: str) -> tuple[int, ...]:
    s = os.environ.get(name, "").strip()
    if not s:
        return ()
    return tuple(int(x) for x in s.split(",") if x.strip())


def probe_duration(video: Path) -> float:
    """Use ffprobe to get the video duration in seconds."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--vllm-url", default=os.environ.get("VLLM_URL"))
    p.add_argument("--job-id", default=None,
                   help="job-id used in working-dir paths; defaults to a fresh UUID")
    p.add_argument("--single-pass", action="store_true",
                   default=_truthy(os.environ.get("SINGLE_PASS_32B"), default=True),
                   help="single-pass 32B (current production default)")
    args = p.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not args.vllm_url:
        print("ERROR: --vllm-url or VLLM_URL env must be set", file=sys.stderr)
        sys.exit(1)

    job_id = args.job_id or str(uuid.uuid4())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(args.video)

    dp_config = DualPassConfig(
        vllm_url=args.vllm_url,
        single_pass=args.single_pass,
        single_pass_step_sec=_float("SINGLE_PASS_STEP_SEC", 10.0),
        single_pass_window_sec=_float("SINGLE_PASS_WINDOW_SEC", 15.0),
        single_pass_frames=_int("SINGLE_PASS_FRAMES", 5),
        yolo_crop_enabled=_truthy(os.environ.get("YOLO_CROP_ENABLED")),
        field_crop_enabled=_truthy(os.environ.get("FIELD_CROP_ENABLED")),
        field_crop_upscale_long_edge=_int("FIELD_CROP_UPSCALE_LONG_EDGE", 0),
        ball_crop_enabled=_truthy(os.environ.get("BALL_CROP_ENABLED")),
        refinement_enabled=_truthy(os.environ.get("REFINEMENT_ENABLED")),
        audio_fusion_enabled=_truthy(os.environ.get("AUDIO_FUSION_ENABLED")),
        audio_cache_dir=str(args.out_dir / "audio_cache"),
        temporal_fusion_enabled=_truthy(os.environ.get("TEMPORAL_FUSION_ENABLED")),
        kickoff_verifier_enabled=_truthy(os.environ.get("KICKOFF_VERIFIER_ENABLED")),
        ball_presence_verifier_enabled=_truthy(os.environ.get("BALL_PRESENCE_VERIFIER_ENABLED")),
        ball_presence_verifier_model_path=os.environ.get("BALL_PRESENCE_VERIFIER_MODEL_PATH", ""),
        ball_presence_verifier_conf=_float("BALL_PRESENCE_VERIFIER_CONF", 0.10),
        ball_presence_verifier_inference_size=_int("BALL_PRESENCE_VERIFIER_INFERENCE_SIZE", 1920),
        ball_presence_verifier_n_frames=_int("BALL_PRESENCE_VERIFIER_N_FRAMES", 4),
        ball_context_enabled=_truthy(os.environ.get("BALL_CONTEXT_ENABLED")),
        ball_context_model_path=os.environ.get("BALL_CONTEXT_MODEL_PATH", ""),
        ball_context_conf=_float("BALL_CONTEXT_CONF", 0.05),
        ball_context_inference_size=_int("BALL_CONTEXT_INFERENCE_SIZE", 1920),
        ball_context_max_dets=_int("BALL_CONTEXT_MAX_DETS", 3),
        ball_trajectory_enabled=_truthy(os.environ.get("BALL_TRAJECTORY_ENABLED")),
        ball_trajectory_acceleration_enabled=_truthy(os.environ.get("BALL_TRAJECTORY_ACCELERATION_ENABLED")),
        tier1_model_name=os.environ.get("DUAL_PASS_TIER1_NAME", "qwen3-vl-32b"),
        tier1_model_path=os.environ.get("DUAL_PASS_TIER1_PATH", ""),
        tier2_model_name=os.environ.get("DUAL_PASS_TIER2_NAME", "qwen3-vl-32b"),
        tier2_model_path=os.environ.get("DUAL_PASS_TIER2_PATH", ""),
        step_sec=_float("DUAL_PASS_TRIAGE_STEP", 10.0),
        swap_script=os.environ.get("DUAL_PASS_SWAP_SCRIPT", ""),
        yolo_grounding_enabled=_truthy(os.environ.get("YOLO_GROUNDING_ENABLED")),
        yolo_grounding_fail_open=_truthy(os.environ.get("YOLO_GROUNDING_FAIL_OPEN"), default=True),
        yolo_grounding_frames=_int("YOLO_GROUNDING_FRAMES", 5),
        yolo_grounding_frame_span_sec=_float("YOLO_GROUNDING_FRAME_SPAN_SEC", 2.0),
        yolo_grounding_inference_size=_int("YOLO_GROUNDING_INFERENCE_SIZE", 640),
        yolo_grounding_ball_conf=_float("YOLO_GROUNDING_BALL_CONF", 0.15),
        yolo_model_path=os.environ.get("YOLO_MODEL_PATH", ""),
        yolo_use_gpu=_truthy(os.environ.get("USE_GPU")),
        yolo_ball_class_id=_int("YOLO_BALL_CLASS_ID", 32),
        yolo_person_class_ids=_csv_int("YOLO_PERSON_CLASS_IDS") or (0,),
        yolo_gk_class_ids=_csv_int("YOLO_GK_CLASS_IDS"),
        yolo_gk_proximity_threshold=_float("YOLO_GK_PROXIMITY_THRESHOLD", 0.20),
        yolo_gk_frames=_int("YOLO_GK_FRAMES", 10),
        yolo_gk_min_span_sec=_float("YOLO_GK_MIN_SPAN_SEC", 6.0),
        yolo_gk_inference_size=_int("YOLO_GK_INFERENCE_SIZE", 1280),
        yolo_trajectory_enabled=_truthy(os.environ.get("YOLO_TRAJECTORY_ENABLED"), default=True),
        yolo_parry_angle_threshold=_float("YOLO_PARRY_ANGLE_THRESHOLD", 90.0),
        yolo_deflection_angle_threshold=_float("YOLO_DEFLECTION_ANGLE_THRESHOLD", 30.0),
        yolo_catch_speed_ratio_threshold=_float("YOLO_CATCH_SPEED_RATIO_THRESHOLD", 0.3),
        yolo_missed_speed_ratio_threshold=_float("YOLO_MISSED_SPEED_RATIO_THRESHOLD", 0.8),
        yolo_fks_lookback_sec=_float("YOLO_FKS_LOOKBACK_SEC", 5.0),
        yolo_fks_n_frames=_int("YOLO_FKS_N_FRAMES", 4),
        yolo_fks_stillness_std_threshold=_float("YOLO_FKS_STILLNESS_STD_THRESHOLD", 0.04),
        yolo_fks_motion_std_threshold=_float("YOLO_FKS_MOTION_STD_THRESHOLD", 0.08),
        yolo_ball_chain_enabled=_truthy(os.environ.get("YOLO_BALL_CHAIN_ENABLED"), default=True),
        yolo_ball_max_speed_per_sec=_float("YOLO_BALL_MAX_SPEED_PER_SEC", 0.3),
    )

    detector = DualPassDetector(
        config=dp_config,
        source_file=str(args.video),
        video_duration=duration,
        job_id=job_id,
        working_dir=str(args.out_dir),
    )

    def on_progress(pct: float) -> None:
        sys.stderr.write(f"\rdetect: {pct * 100:5.1f}%")
        sys.stderr.flush()

    events = detector.detect(progress_callback=on_progress)
    sys.stderr.write("\n")

    events_path = args.out_dir / "events.jsonl"
    with events_path.open("w") as f:
        for e in events:
            f.write(e.model_dump_json() + "\n")

    print(f"wrote {len(events)} events to {events_path}")
    print(f"job_id={job_id}")


if __name__ == "__main__":
    main()
