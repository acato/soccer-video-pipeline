"""
YOLOv8 player and ball detector.

Processes video in sliding chunks using FFmpeg frame extraction.
Frames are decoded at reduced resolution for inference, but all timestamps
reference the original 4K source.

GPU/CPU fallback is automatic via PyTorch device selection.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import structlog

from src.detection.base import BaseDetector
from src.detection.models import BoundingBox, Detection, Event, Track

log = structlog.get_logger(__name__)

# YOLO class names for soccer domain
YOLO_CLASS_PLAYER     = "player"
YOLO_CLASS_BALL       = "ball"
YOLO_CLASS_GOALKEEPER = "goalkeeper"
YOLO_CLASS_REFEREE    = "referee"

# Map COCO class names → pipeline class names.
# A fine-tuned soccer model would output "player"/"ball" directly and bypass this.
COCO_TO_PIPELINE_CLASS = {
    "person": YOLO_CLASS_PLAYER,
    "sports ball": YOLO_CLASS_BALL,
}


class PlayerDetector(BaseDetector):
    """
    Detects players, ball, and goalkeepers using YOLOv8.

    Frame extraction uses FFmpeg (not OpenCV VideoCapture) to ensure
    NAS-tolerant, buffered reads. Frames are extracted to a temp directory
    in WORKING_DIR (local SSD) before inference.
    """

    def __init__(
        self,
        job_id: str,
        source_file: str,
        model_path: str,
        use_gpu: bool = True,
        inference_size: int = 1280,
        frame_step: int = 3,
        working_dir: str = "/tmp/soccer-pipeline",
    ):
        super().__init__(job_id, source_file)
        self.model_path = model_path
        self.inference_size = inference_size
        self.frame_step = frame_step
        self.working_dir = Path(working_dir)
        self._model = None  # Lazy-loaded on first use
        self._use_gpu = use_gpu
        self._device = self._select_device(use_gpu)

    @property
    def reel_targets(self) -> list[str]:
        return ["goalkeeper", "highlights"]

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
    ) -> list[Detection]:
        """Run YOLO inference on a single frame (BGR numpy array)."""
        model = self._get_model()
        results = model.predict(
            frame,
            imgsz=self.inference_size,
            device=self._device,
            verbose=False,
            conf=0.3,  # Low threshold here; confidence filtering at event level
        )

        detections = []
        if not results:
            return detections

        h, w = frame.shape[:2]
        for box in results[0].boxes:
            cls_idx = int(box.cls[0])
            raw_name = results[0].names.get(cls_idx, "unknown")
            cls_name = COCO_TO_PIPELINE_CLASS.get(raw_name, raw_name)
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(Detection(
                frame_number=frame_number,
                timestamp=timestamp,
                class_name=cls_name,
                confidence=conf,
                bbox=BoundingBox(
                    x=x1 / w,
                    y=y1 / h,
                    width=(x2 - x1) / w,
                    height=(y2 - y1) / h,
                ),
            ))
        return detections

    def detect_chunk(
        self,
        source_path: str,
        chunk_start_sec: float,
        chunk_duration_sec: float,
        source_fps: float,
    ) -> list[Detection]:
        """
        Extract and run detection on one chunk of video.
        Frames are extracted to local SSD temp dir — NAS is read once per chunk.
        """
        frames_dir = self.working_dir / self.job_id / "frames_tmp"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            frame_paths = self._extract_frames(
                source_path, chunk_start_sec, chunk_duration_sec,
                self.frame_step, frames_dir
            )

            all_detections = []
            for idx, frame_path in enumerate(sorted(frames_dir.glob("frame_*.jpg"))):
                frame = self._load_frame(str(frame_path))
                if frame is None:
                    continue
                # Compute true frame number and timestamp in source video
                frame_number = int(chunk_start_sec * source_fps) + idx * self.frame_step
                timestamp = chunk_start_sec + (idx * self.frame_step / source_fps)
                dets = self.detect_frame(frame, frame_number, timestamp)
                all_detections.extend(dets)

            return all_detections
        finally:
            # Clean up temp frames immediately — don't accumulate on SSD
            for f in frames_dir.glob("frame_*.jpg"):
                f.unlink(missing_ok=True)

    def finalize_chunk(self, tracks: list[Track]) -> list[Event]:
        """Player detector doesn't classify events — handled by event_classifier."""
        return []

    # ── Internal ──────────────────────────────────────────────────────────

    def _extract_frames(
        self,
        source_path: str,
        start_sec: float,
        duration_sec: float,
        frame_step: int,
        output_dir: Path,
    ) -> list[Path]:
        """
        Use FFmpeg to extract frames at 1/frame_step rate to output_dir.
        Returns sorted list of extracted frame paths.
        """
        select_expr = f"not(mod(n\\,{frame_step}))"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-i", source_path,
            "-vf", f"select='{select_expr}',scale={self.inference_size}:-1",
            "-vsync", "vfr",
            "-q:v", "3",          # JPEG quality
            str(output_dir / "frame_%06d.jpg"),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            log.error(
                "frame_extraction.failed",
                chunk_start=start_sec,
                error=result.stderr.decode()[:500],
            )
            return []
        return sorted(output_dir.glob("frame_*.jpg"))

    def _load_frame(self, path: str):
        """Load a JPEG frame as numpy array (BGR). Returns None on error."""
        try:
            import cv2
            frame = cv2.imread(path)
            return frame
        except Exception as exc:
            log.warning("frame_load.error", path=path, error=str(exc))
            return None

    def _get_model(self):
        """Lazy-load YOLO model on first call."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                log.info("player_detector.model_loaded", path=self.model_path, device=self._device)
            except Exception as exc:
                raise RuntimeError(f"Failed to load YOLO model from {self.model_path}: {exc}")
        return self._model

    @staticmethod
    def _select_device(use_gpu: bool) -> str:
        """Select best available device: cuda:0 → mps → cpu."""
        if not use_gpu:
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                log.info("player_detector.using_gpu", device="cuda:0")
                return "cuda:0"
            if torch.backends.mps.is_available():
                log.info("player_detector.using_gpu", device="mps")
                return "mps"
        except (ImportError, AttributeError):
            pass
        log.info("player_detector.using_cpu", reason="GPU not available or torch not installed")
        return "cpu"
