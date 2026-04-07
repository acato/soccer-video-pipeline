"""
8B triage scanner — fast multi-frame sliding window for candidate generation.

Scans the entire game with the 8B model using a sliding window of N frames
spanning a configurable time span. Produces candidate time windows where
"something is happening" for detailed 32B review.

Uses a simple 3-label taxonomy (EVENT/PLAY/DEAD) because 8B models can't
reliably distinguish specific set piece types. The 32B pass does fine-grained
classification. The 8B just needs to separate "something happened" from
"open midfield play" and "dead ball / stoppage".
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import structlog

from src.detection.frame_sampler import FrameSampler, SampledFrame

log = structlog.get_logger(__name__)

# Simple 3-label taxonomy — 8B only decides "is something happening?"
TRIAGE_LABELS = ["EVENT", "PLAY", "DEAD"]

# Labels that trigger a candidate window for 32B review
ACTIVE_LABELS = frozenset({"EVENT"})


VALID_BALL_ZONES = frozenset({"left_third", "middle", "right_third"})


@dataclass
class TriageFlag:
    """A single triage flag from the 8B model."""
    center_sec: float   # Center timestamp of the window
    label: str          # 8B classification label
    window_start: float # Start of the multi-frame window
    window_end: float   # End of the multi-frame window
    ball_zone: str = "middle"  # Where the ball/action is on the pitch


@dataclass
class CandidateWindow:
    """Merged candidate window for 32B review."""
    start_sec: float
    end_sec: float
    labels: list[str] = field(default_factory=list)  # All triage labels in this window
    flags: list[TriageFlag] = field(default_factory=list)

    @property
    def center_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


TRIAGE_PROMPT = """\
You are analyzing a soccer match from a sideline camera. \
Here are {n_frames} frames spanning {span_sec:.0f} seconds of play.

Classify what is happening using ONE label:

- EVENT: Something specific is happening — a set piece (corner kick, throw-in, \
goal kick, free kick, penalty), a shot, a save, a goal, a goalkeeper catch, \
a kickoff, or any restart of play. Look for: players lined up at the sideline \
for a throw-in, a player at the corner flag, the ball in the goalkeeper's \
hands, players forming a wall for a free kick, the ball in or near the goal, \
celebrations, or the ball being placed on the ground for a restart.
- PLAY: Normal open play — midfield passing, dribbling, running with the ball. \
No set piece, shot, or goalkeeper action visible.
- DEAD: Dead ball with no restart visible, stoppage, referee talking, replays, \
graphics overlay, halftime, or players milling around.

Respond with ONLY a JSON object:
{{"label": "PLAY", "ball_zone": "middle"}}

ball_zone should be one of: "left_third", "middle", "right_third" \
(based on where the ball/action is on the pitch from the camera's perspective).
"""


class TriageScanner:
    """Scan an entire game with the 8B model using multi-frame sliding windows."""

    def __init__(
        self,
        vllm_url: str,
        vllm_model: str,
        source_file: str,
        video_duration: float,
        frame_width: int = 960,
        frames_per_window: int = 5,
        window_span_sec: float = 10.0,
        step_sec: float = 6.0,
    ):
        self._vllm_url = vllm_url.rstrip("/")
        self._vllm_model = vllm_model
        self._source_file = source_file
        self._video_duration = video_duration
        self._frame_width = frame_width
        self._frames_per_window = frames_per_window
        self._window_span_sec = window_span_sec
        self._step_sec = step_sec

        self._sampler = FrameSampler(source_file, frame_width=frame_width)

    def scan(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> list[TriageFlag]:
        """Scan the game with sliding windows, returning triage flags.

        Args:
            progress_callback: Called with fraction 0.0-1.0 as scan progresses.
            start_sec: Start scanning from this timestamp.
            end_sec: Stop scanning at this timestamp (default: video end).

        Returns:
            List of TriageFlag for windows classified as active events.
        """
        if end_sec is None:
            end_sec = self._video_duration

        # Pre-extract all frames at 1fps for the entire range
        log.info("triage_scanner.extracting_frames",
                 start=start_sec, end=end_sec, duration=end_sec - start_sec)
        all_frames = self._sampler.sample(
            duration_sec=end_sec,
            interval_sec=1.0,
            start_sec=start_sec,
        )
        log.info("triage_scanner.frames_extracted", count=len(all_frames))

        # Build a lookup: timestamp → frame
        frame_by_ts = {round(f.timestamp_sec): f for f in all_frames}

        # Sliding window
        flags: list[TriageFlag] = []
        window_start = start_sec
        total_windows = int((end_sec - start_sec - self._window_span_sec) / self._step_sec) + 1
        window_idx = 0

        while window_start + self._window_span_sec <= end_sec:
            window_end = window_start + self._window_span_sec
            center = window_start + self._window_span_sec / 2

            # Pick N evenly-spaced frames from this window
            frame_times = [
                round(window_start + i * self._window_span_sec / (self._frames_per_window - 1))
                for i in range(self._frames_per_window)
            ]
            frames = [frame_by_ts[t] for t in frame_times if t in frame_by_ts]

            if len(frames) >= 3:  # Need at least 3 frames
                label, ball_zone = self._classify_window(frames)
                if label in ACTIVE_LABELS:
                    flags.append(TriageFlag(
                        center_sec=center,
                        label=label,
                        window_start=window_start,
                        window_end=window_end,
                        ball_zone=ball_zone,
                    ))

            window_idx += 1
            if progress_callback and total_windows > 0:
                progress_callback(min(1.0, window_idx / total_windows))

            window_start += self._step_sec

        log.info("triage_scanner.scan_complete",
                 total_windows=window_idx, flags=len(flags))
        return flags

    def _classify_window(self, frames: list[SampledFrame]) -> tuple[str, str]:
        """Send a multi-frame window to the 8B model for classification.

        Returns:
            (label, ball_zone) tuple.
        """
        import httpx

        prompt = TRIAGE_PROMPT.format(
            n_frames=len(frames),
            span_sec=self._window_span_sec,
        )

        content: list[dict] = []
        for frame in frames:
            b64 = base64.b64encode(frame.jpeg_bytes).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self._vllm_model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 100,
            "temperature": 0,
        }

        try:
            r = httpx.post(
                f"{self._vllm_url}/v1/chat/completions",
                json=payload,
                timeout=60,
            )
            if r.status_code != 200:
                log.warning("triage_scanner.vllm_error",
                            status=r.status_code, body=r.text[:200])
                return ("PLAY", "middle")

            text = r.json()["choices"][0]["message"]["content"].strip()
            return self._parse_response(text)

        except Exception as exc:
            log.warning("triage_scanner.request_error", error=str(exc))
            return ("PLAY", "middle")

    def _parse_response(self, text: str) -> tuple[str, str]:
        """Parse the 8B model's JSON response.

        Returns:
            (label, ball_zone) tuple.
        """
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            label = data.get("label", "PLAY").upper()
            ball_zone = data.get("ball_zone", "middle")
            if label not in TRIAGE_LABELS:
                label = "PLAY"
            if ball_zone not in VALID_BALL_ZONES:
                ball_zone = "middle"
            return (label, ball_zone)
        except (json.JSONDecodeError, ValueError, TypeError):
            # Try to extract label from free text
            text_upper = text.upper()
            for label in TRIAGE_LABELS:
                if label in text_upper:
                    return (label, "middle")
            return ("PLAY", "middle")


def merge_flags(
    flags: list[TriageFlag],
    merge_gap_sec: float = 4.0,
    pad_sec: float = 3.0,
    max_window_sec: float = 60.0,
) -> list[CandidateWindow]:
    """Merge nearby triage flags into candidate windows for 32B review.

    Merging rules:
      - Flags closer than ``merge_gap_sec`` are merged.
      - Ball-zone change forces a window break (different area of the pitch
        likely means a different event).
      - Windows exceeding ``max_window_sec`` are split at the largest gap.

    Args:
        flags: List of triage flags from the 8B scan.
        merge_gap_sec: Maximum gap between flags to merge them.
        pad_sec: Padding added before first and after last flag in a window.
        max_window_sec: Hard ceiling on window duration (split if exceeded).

    Returns:
        List of CandidateWindow with merged time ranges.
    """
    if not flags:
        return []

    sorted_flags = sorted(flags, key=lambda f: f.center_sec)

    # First pass: group flags, breaking on gap or ball_zone change
    groups: list[list[TriageFlag]] = [[sorted_flags[0]]]

    for flag in sorted_flags[1:]:
        prev = groups[-1][-1]
        gap = flag.window_start - prev.window_end
        zone_changed = flag.ball_zone != prev.ball_zone

        if gap > merge_gap_sec or zone_changed:
            groups.append([flag])
        else:
            groups[-1].append(flag)

    # Second pass: convert groups to windows, split oversized ones
    windows: list[CandidateWindow] = []
    for group in groups:
        win = _group_to_window(group, pad_sec)
        if win.duration_sec <= max_window_sec:
            windows.append(win)
        else:
            windows.extend(_split_oversized(group, pad_sec, max_window_sec))

    log.info("triage_scanner.merge_complete",
             input_flags=len(flags), output_windows=len(windows),
             total_review_sec=sum(w.duration_sec for w in windows))
    return windows


def _group_to_window(
    group: list[TriageFlag], pad_sec: float
) -> CandidateWindow:
    """Convert a list of flags into a single CandidateWindow."""
    return CandidateWindow(
        start_sec=max(0, group[0].window_start - pad_sec),
        end_sec=group[-1].window_end + pad_sec,
        labels=[f.label for f in group],
        flags=list(group),
    )


def _split_oversized(
    group: list[TriageFlag],
    pad_sec: float,
    max_window_sec: float,
) -> list[CandidateWindow]:
    """Split an oversized group of flags at the largest internal gap.

    Recursively splits until all windows are under max_window_sec.
    """
    if len(group) <= 1:
        return [_group_to_window(group, pad_sec)]

    # Find largest gap
    best_gap = -1.0
    best_idx = 0
    for i in range(1, len(group)):
        gap = group[i].window_start - group[i - 1].window_end
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    left = group[:best_idx]
    right = group[best_idx:]
    result: list[CandidateWindow] = []

    for sub in (left, right):
        if not sub:
            continue
        win = _group_to_window(sub, pad_sec)
        if win.duration_sec <= max_window_sec:
            result.append(win)
        else:
            result.extend(_split_oversized(sub, pad_sec, max_window_sec))

    return result
