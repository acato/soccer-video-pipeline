"""
High-level event classifier that orchestrates all detectors for a full video.

Processes video in overlapping chunks, accumulates events per reel type,
and writes the final event log to disk.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np
import structlog

from src.detection.event_log import EventLog
from src.detection.goalkeeper_detector import GoalkeeperDetector
from src.detection.models import Event, EventType, Track, EVENT_REEL_MAP
from src.detection.player_detector import PlayerDetector
from src.ingestion.models import VideoFile
from src.tracking.gk_tracker import MatchDualGoalkeeperTracker
from src.tracking.tracker import PlayerTracker

log = structlog.get_logger(__name__)


def classify_highlights_events(
    tracks: list[Track],
    job_id: str,
    source_file: str,
    source_fps: float,
) -> list[Event]:
    """
    Detect highlights-reel events from tracked player/ball data.
    
    Key heuristics:
    - Shot: ball track with high velocity toward goal region
    - Goal: ball enters goal bounding box and disappears
    - Dribble: one player track passes through multiple other player bboxes
    - Tackle: rapid proximity + sudden velocity change between two tracks
    """
    events = []
    ball_tracks = [t for t in tracks if any(d.class_name == "ball" for d in t.detections)]
    player_tracks = [t for t in tracks if any(d.class_name in ("player", "goalkeeper") for d in t.detections)]

    # ── Shot detection via ball velocity ──────────────────────────────────
    for ball_track in ball_tracks:
        events.extend(_detect_shots(ball_track, job_id, source_file, source_fps))

    # ── Tackle detection via player proximity ─────────────────────────────
    events.extend(_detect_tackles(player_tracks, job_id, source_file, source_fps))

    # ── Dribble sequence detection ────────────────────────────────────────
    events.extend(_detect_dribbles(player_tracks, job_id, source_file, source_fps))

    return events


def _detect_shots(
    ball_track: Track, job_id: str, source_file: str, fps: float
) -> list[Event]:
    """Detect shots: ball with increasing horizontal velocity near goal area."""
    events = []
    dets = ball_track.detections
    if len(dets) < 4:
        return events

    for i in range(2, len(dets) - 1):
        prev, curr = dets[i-2], dets[i]
        dt = curr.timestamp - prev.timestamp
        if dt <= 0:
            continue

        vx = abs(curr.bbox.center_x - prev.bbox.center_x) / dt
        vy = abs(curr.bbox.center_y - prev.bbox.center_y) / dt
        speed = (vx**2 + vy**2) ** 0.5

        # High speed ball near goal (top or bottom 15% of frame = near goal)
        near_goal = curr.bbox.center_y < 0.2 or curr.bbox.center_y > 0.8
        near_sides = curr.bbox.center_x < 0.1 or curr.bbox.center_x > 0.9

        if speed > 0.4 and (near_goal or near_sides):
            event_type = EventType.SHOT_ON_TARGET if near_goal else EventType.SHOT_OFF_TARGET
            events.append(Event(
                job_id=job_id,
                source_file=source_file,
                event_type=event_type,
                timestamp_start=max(0, prev.timestamp - 1.0),
                timestamp_end=curr.timestamp + 3.0,
                confidence=min(0.90, 0.55 + speed * 0.5),
                reel_targets=EVENT_REEL_MAP[event_type],
                frame_start=prev.frame_number,
                frame_end=curr.frame_number + int(3 * fps),
                bounding_box=curr.bbox,
                metadata={"ball_speed_px_per_sec": speed},
            ))

    return _merge_nearby(events, 2.0)


def _detect_tackles(
    player_tracks: list[Track], job_id: str, source_file: str, fps: float
) -> list[Event]:
    """Detect tackles: two players in very close proximity with rapid movement."""
    events = []
    if len(player_tracks) < 2:
        return events

    # Build frame-indexed lookup
    frame_to_players: dict[int, list[tuple[int, object]]] = {}
    for track in player_tracks:
        for det in track.detections:
            frame_to_players.setdefault(det.frame_number, []).append((track.track_id, det))

    for frame_num, players in frame_to_players.items():
        if len(players) < 2:
            continue
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                tid1, det1 = players[i]
                tid2, det2 = players[j]
                dist = (
                    (det1.bbox.center_x - det2.bbox.center_x) ** 2 +
                    (det1.bbox.center_y - det2.bbox.center_y) ** 2
                ) ** 0.5

                if dist < 0.05:  # Very close proximity (normalized units)
                    ts = det1.timestamp
                    events.append(Event(
                        job_id=job_id,
                        source_file=source_file,
                        event_type=EventType.TACKLE,
                        timestamp_start=max(0, ts - 1.0),
                        timestamp_end=ts + 3.0,
                        confidence=0.65,
                        reel_targets=EVENT_REEL_MAP[EventType.TACKLE],
                        frame_start=max(0, frame_num - int(fps)),
                        frame_end=frame_num + int(3 * fps),
                        bounding_box=det1.bbox,
                    ))

    return _merge_nearby(events, 3.0)


def _detect_dribbles(
    player_tracks: list[Track], job_id: str, source_file: str, fps: float
) -> list[Event]:
    """Detect dribble sequences: a player track with sustained high velocity."""
    events = []
    for track in player_tracks:
        dets = track.detections
        if len(dets) < 10:
            continue

        speeds = []
        for i in range(1, len(dets)):
            dt = dets[i].timestamp - dets[i-1].timestamp
            if dt <= 0:
                continue
            dx = dets[i].bbox.center_x - dets[i-1].bbox.center_x
            dy = dets[i].bbox.center_y - dets[i-1].bbox.center_y
            speeds.append(((dx**2 + dy**2) ** 0.5) / dt)

        if not speeds:
            continue

        # Find windows of sustained high speed (dribble = moving fast for > 1.5s)
        import numpy as np
        threshold = 0.08
        high_speed_frames = [i for i, s in enumerate(speeds) if s > threshold]

        if len(high_speed_frames) >= int(1.5 * fps / 3):  # Sustained for ~1.5s
            start_det = dets[high_speed_frames[0]]
            end_det = dets[high_speed_frames[-1]]
            if end_det.timestamp - start_det.timestamp >= 1.5:
                events.append(Event(
                    job_id=job_id,
                    source_file=source_file,
                    event_type=EventType.DRIBBLE_SEQUENCE,
                    timestamp_start=max(0, start_det.timestamp - 0.5),
                    timestamp_end=end_det.timestamp + 2.0,
                    confidence=0.67,
                    reel_targets=EVENT_REEL_MAP[EventType.DRIBBLE_SEQUENCE],
                    player_track_id=track.track_id,
                    frame_start=start_det.frame_number,
                    frame_end=end_det.frame_number,
                    bounding_box=start_det.bbox,
                ))

    return events


def _merge_nearby(events: list[Event], gap_sec: float) -> list[Event]:
    """Merge temporally close events of the same type."""
    if len(events) <= 1:
        return events
    events = sorted(events, key=lambda e: e.timestamp_start)
    merged = [events[0]]
    for ev in events[1:]:
        prev = merged[-1]
        if ev.event_type == prev.event_type and ev.timestamp_start - prev.timestamp_end < gap_sec:
            data = prev.model_dump()
            data["timestamp_end"] = max(prev.timestamp_end, ev.timestamp_end)
            data["frame_end"] = max(prev.frame_end, ev.frame_end)
            data["confidence"] = max(prev.confidence, ev.confidence)
            merged[-1] = Event(**data)
        else:
            merged.append(ev)
    return merged


def _aggregate_jersey_colors(tracks: list[Track]) -> None:
    """
    Populate Track.jersey_color_hsv from per-detection jersey_hsv metadata.

    PlayerDetector tags each detection with its jersey HSV colour while the
    frame is still in memory.  Here we aggregate those per-detection colours
    into a single per-track colour using the median (robust to single-frame noise).
    """
    for track in tracks:
        if track.jersey_color_hsv is not None:
            continue
        colors = [
            d.metadata["jersey_hsv"]
            for d in track.detections
            if "jersey_hsv" in d.metadata
        ]
        if colors:
            arr = np.array(colors)
            track.jersey_color_hsv = tuple(
                float(np.median(arr[:, i])) for i in range(3)
            )


class PipelineRunner:
    """
    Orchestrates the full detection pipeline for one video file.

    Processes in overlapping chunks, accumulates events, and writes event log.
    Called by the Celery worker task.
    """

    def __init__(
        self,
        job_id: str,
        video_file: VideoFile,
        player_detector: PlayerDetector,
        gk_detector: GoalkeeperDetector,
        event_log: EventLog,
        chunk_sec: int = 30,
        overlap_sec: float = 2.0,
        min_confidence: float = 0.65,
    ):
        self.job_id = job_id
        self.video_file = video_file
        self.player_detector = player_detector
        self.gk_detector = gk_detector
        self.event_log = event_log
        self.chunk_sec = chunk_sec
        self.overlap_sec = overlap_sec
        self.min_confidence = min_confidence
        self._tracker = PlayerTracker()
        self._gk_tracker = MatchDualGoalkeeperTracker(job_id)

    def run(self, progress_callback=None) -> int:
        """
        Run full detection pipeline. Returns total events detected.
        progress_callback(pct: float) called periodically if provided.
        """
        duration = self.video_file.duration_sec
        fps = self.video_file.fps
        total_events = 0
        consecutive_empty = 0
        max_consecutive_empty = 10

        chunk_starts = self._chunk_starts(duration)
        total_chunks = len(chunk_starts)

        for chunk_idx, start_sec in enumerate(chunk_starts):
            chunk_dur = min(self.chunk_sec + self.overlap_sec, duration - start_sec)
            if chunk_dur <= 0:
                break

            log.info(
                "pipeline.chunk_start",
                chunk=chunk_idx + 1,
                total=total_chunks,
                start_sec=start_sec,
                duration_sec=chunk_dur,
            )

            t0 = time.monotonic()
            self._tracker.reset()

            # Detect players/ball in this chunk
            detections = self.player_detector.detect_chunk(
                self.video_file.path, start_sec, chunk_dur, fps
            )

            # Abort early if frame extraction keeps failing (NAS/disk issue)
            if not detections:
                consecutive_empty += 1
                if consecutive_empty >= max_consecutive_empty:
                    raise RuntimeError(
                        f"Frame extraction failed for {consecutive_empty} consecutive "
                        f"chunks (from {start_sec - (consecutive_empty - 1) * self.chunk_sec:.0f}s). "
                        f"Source file may be inaccessible: {self.video_file.path}"
                    )
            else:
                consecutive_empty = 0

            # Group detections by frame and feed to tracker
            frame_groups: dict[int, list] = {}
            for det in detections:
                frame_groups.setdefault(det.frame_number, []).append(det)

            for frame_num in sorted(frame_groups):
                self._tracker.update(
                    frame_groups[frame_num],
                    frame_shape=(self.video_file.height, self.video_file.width),
                )

            tracks = self._tracker.get_all_tracks()

            # Aggregate jersey colors from per-detection metadata onto tracks
            _aggregate_jersey_colors(tracks)

            # Build track_colors dict from tracks with jersey_color_hsv
            track_colors = {
                t.track_id: t.jersey_color_hsv
                for t in tracks if t.jersey_color_hsv is not None
            }

            # Identify both GKs using jersey-first approach
            gk_ids = self.gk_detector.identify_goalkeepers(
                tracks,
                (self.video_file.height, self.video_file.width),
                track_colors,
            )
            self._gk_tracker.register_chunk_gks(chunk_idx, gk_ids)

            # Classify events
            chunk_events: list[Event] = []

            # GK events — only for the team's GK (opponent's label is None)
            for keeper_role in ("keeper_a", "keeper_b"):
                gk_id = gk_ids.get(keeper_role)
                if gk_id is None:
                    continue
                reel_label = self.gk_detector.reel_label_for(keeper_role)
                if reel_label is None:
                    continue
                gk_tracks = [t for t in tracks if t.track_id == gk_id]
                if gk_tracks:
                    gk_track = gk_tracks[0]
                    gk_track.is_goalkeeper = True
                    gk_events = self.gk_detector.classify_gk_events(
                        gk_track, tracks, fps, keeper_role=reel_label,
                    )
                    chunk_events.extend(gk_events)

            # Highlights events
            hl_events = classify_highlights_events(tracks, self.job_id, self.video_file.path, fps)
            chunk_events.extend(hl_events)

            # Filter by confidence and write to log
            passing = [e for e in chunk_events if e.should_include(self.min_confidence)]
            self.event_log.append_many(passing)
            total_events += len(passing)

            elapsed = time.monotonic() - t0
            speedup = chunk_dur / elapsed if elapsed > 0 else 0
            log.info(
                "pipeline.chunk_done",
                chunk=chunk_idx + 1,
                events_found=len(passing),
                elapsed_sec=round(elapsed, 1),
                speedup=f"{speedup:.1f}x",
            )

            if progress_callback:
                progress_callback((chunk_idx + 1) / total_chunks * 100)

        gk_summary = self._gk_tracker.summary()
        log.info(
            "pipeline.complete",
            total_events=total_events,
            job_id=self.job_id,
            gk_identification_rate=gk_summary["identification_rate"],
            gk_summary=gk_summary,
        )
        return total_events

    def _chunk_starts(self, duration: float) -> list[float]:
        starts = []
        t = 0.0
        step = self.chunk_sec
        while t < duration:
            starts.append(t)
            t += step
        return starts


class PipelineRunnerV2(PipelineRunner):
    """
    Extended pipeline runner that adds:
    1. Jersey-based GK identification (fallback when homography unavailable)
    2. Action recognition confirmation of candidate events
    3. Confidence calibration before writing to event log

    Drop-in replacement for PipelineRunner with the same interface.
    """

    def __init__(self, *args, action_model_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_model_path = action_model_path

    def run(self, progress_callback=None) -> int:
        from src.detection.action_classifier import ActionClassifier
        from src.detection.confidence_calibration import calibrate_events

        duration = self.video_file.duration_sec
        fps = self.video_file.fps
        total_events = 0
        chunk_starts = self._chunk_starts(duration)
        total_chunks = len(chunk_starts)
        action_clf = ActionClassifier(model_path=self._action_model_path, use_gpu=True)

        for chunk_idx, start_sec in enumerate(chunk_starts):
            chunk_dur = min(self.chunk_sec + self.overlap_sec, duration - start_sec)
            if chunk_dur <= 0:
                break

            import time
            t0 = time.monotonic()
            self._tracker.reset()

            detections = self.player_detector.detect_chunk(
                self.video_file.path, start_sec, chunk_dur, fps
            )

            frame_groups: dict[int, list] = {}
            for det in detections:
                frame_groups.setdefault(det.frame_number, []).append(det)

            for frame_num in sorted(frame_groups):
                self._tracker.update(
                    frame_groups[frame_num],
                    frame_shape=(self.video_file.height, self.video_file.width),
                )

            tracks = self._tracker.get_all_tracks()

            # V2: Aggregate jersey colors, then identify keepers
            _aggregate_jersey_colors(tracks)
            track_colors = {
                t.track_id: t.jersey_color_hsv
                for t in tracks if t.jersey_color_hsv is not None
            }

            gk_ids = self.gk_detector.identify_goalkeepers(
                tracks,
                (self.video_file.height, self.video_file.width),
                track_colors,
            )
            self._gk_tracker.register_chunk_gks(chunk_idx, gk_ids)

            chunk_events: list[Event] = []

            # GK events for each identified keeper
            for keeper_role in ("keeper_a", "keeper_b"):
                gk_id = gk_ids.get(keeper_role)
                if gk_id is None:
                    continue
                gk_tracks = [t for t in tracks if t.track_id == gk_id]
                if gk_tracks:
                    gk_track = gk_tracks[0]
                    gk_track.is_goalkeeper = True
                    gk_events = self.gk_detector.classify_gk_events(
                        gk_track, tracks, fps, keeper_role=keeper_role
                    )
                    chunk_events.extend(gk_events)

            hl_events = classify_highlights_events(tracks, self.job_id, self.video_file.path, fps)
            chunk_events.extend(hl_events)

            # V2: Action recognition confirmation
            if action_clf._ensure_loaded():
                chunk_events = action_clf.confirm_events(
                    chunk_events, self.video_file.path, self.job_id
                )

            # V2: Calibrate before threshold filter
            chunk_events = calibrate_events(chunk_events, "motion_heuristic")

            passing = [e for e in chunk_events if e.should_include(self.min_confidence)]
            self.event_log.append_many(passing)
            total_events += len(passing)

            elapsed = time.monotonic() - t0
            log.info(
                "pipeline_v2.chunk_done",
                chunk=chunk_idx + 1,
                total=total_chunks,
                events=len(passing),
                speedup=f"{chunk_dur / elapsed:.1f}x" if elapsed > 0 else "∞",
            )

            if progress_callback:
                progress_callback((chunk_idx + 1) / total_chunks * 100)

        log.info("pipeline_v2.complete", total_events=total_events)
        return total_events
