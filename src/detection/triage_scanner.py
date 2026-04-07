"""
8B triage scanner — fast multi-frame sliding window for candidate generation.

Scans the entire game with the 8B model using a sliding window of N frames
spanning a configurable time span. Produces candidate time windows where
"something is happening" (ATTACK, SHOT, SAVE, etc.) for detailed 32B review.

The 8B model can't reliably classify specific set pieces (corners, throw-ins)
but detects attacking play and shots near goals/saves with good recall.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import structlog

from src.detection.frame_sampler import FrameSampler, SampledFrame

log = structlog.get_logger(__name__)

# Labels the 8B model can assign. Labels that indicate "something happening"
# will be flagged as candidates for the 32B pass.
TRIAGE_LABELS = [
    "GOAL", "SAVE", "SHOT", "CORNER", "FREE_KICK",
    "THROW_IN", "GOAL_KICK", "KICKOFF", "PENALTY",
    "ATTACK", "PLAY", "DEAD",
]

# Labels that trigger a candidate window for 32B review
ACTIVE_LABELS = frozenset({
    "GOAL", "SAVE", "SHOT", "CORNER", "FREE_KICK",
    "THROW_IN", "GOAL_KICK", "KICKOFF", "PENALTY", "ATTACK",
})


@dataclass
class TriageFlag:
    """A single triage flag from the 8B model."""
    center_sec: float   # Center timestamp of the window
    label: str          # 8B classification label
    window_start: float # Start of the multi-frame window
    window_end: float   # End of the multi-frame window


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

Classify what is happening in these frames. Choose ONE label:

- GOAL: Ball clearly entering the net, or immediate celebration after scoring
- SAVE: Goalkeeper diving/catching/parrying a shot
- SHOT: A player shooting toward goal (but no goal or save visible)
- CORNER: Corner kick being set up or taken (player at corner flag)
- FREE_KICK: Free kick being set up or taken
- THROW_IN: Throw-in being taken from the sideline
- GOAL_KICK: Goal kick being taken from the 6-yard box
- KICKOFF: Kick-off from the center circle
- PENALTY: Penalty kick setup or execution
- ATTACK: Attacking play, ball moving toward a goal, dangerous buildup
- PLAY: Normal midfield play, nothing dangerous
- DEAD: Dead ball, stoppage, replays, graphics, halftime

Respond with ONLY a JSON object:
{{"label": "PLAY", "confidence": 0.8}}
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
                label, conf = self._classify_window(frames)
                if label in ACTIVE_LABELS:
                    flags.append(TriageFlag(
                        center_sec=center,
                        label=label,
                        window_start=window_start,
                        window_end=window_end,
                    ))

            window_idx += 1
            if progress_callback and total_windows > 0:
                progress_callback(min(1.0, window_idx / total_windows))

            window_start += self._step_sec

        log.info("triage_scanner.scan_complete",
                 total_windows=window_idx, flags=len(flags))
        return flags

    def _classify_window(self, frames: list[SampledFrame]) -> tuple[str, float]:
        """Send a multi-frame window to the 8B model for classification.

        Returns:
            (label, confidence) tuple.
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
                return ("PLAY", 0.0)

            text = r.json()["choices"][0]["message"]["content"].strip()
            return self._parse_response(text)

        except Exception as exc:
            log.warning("triage_scanner.request_error", error=str(exc))
            return ("PLAY", 0.0)

    def _parse_response(self, text: str) -> tuple[str, float]:
        """Parse the 8B model's JSON response."""
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            label = data.get("label", "PLAY").upper()
            conf = float(data.get("confidence", 0.5))
            if label not in TRIAGE_LABELS:
                label = "PLAY"
            return (label, conf)
        except (json.JSONDecodeError, ValueError, TypeError):
            # Try to extract label from free text
            text_upper = text.upper()
            for label in TRIAGE_LABELS:
                if label in text_upper:
                    return (label, 0.5)
            return ("PLAY", 0.0)


def merge_flags(
    flags: list[TriageFlag],
    merge_gap_sec: float = 15.0,
    pad_sec: float = 10.0,
) -> list[CandidateWindow]:
    """Merge nearby triage flags into candidate windows for 32B review.

    Args:
        flags: Sorted list of triage flags from the 8B scan.
        merge_gap_sec: Maximum gap between flags to merge them.
        pad_sec: Padding added before first and after last flag in a window.

    Returns:
        List of CandidateWindow with merged time ranges.
    """
    if not flags:
        return []

    sorted_flags = sorted(flags, key=lambda f: f.center_sec)
    windows: list[CandidateWindow] = []
    current = CandidateWindow(
        start_sec=max(0, sorted_flags[0].window_start - pad_sec),
        end_sec=sorted_flags[0].window_end + pad_sec,
        labels=[sorted_flags[0].label],
        flags=[sorted_flags[0]],
    )

    for flag in sorted_flags[1:]:
        if flag.window_start - current.end_sec <= merge_gap_sec:
            # Merge: extend the window
            current.end_sec = flag.window_end + pad_sec
            current.labels.append(flag.label)
            current.flags.append(flag)
        else:
            # Gap too large: start new window
            windows.append(current)
            current = CandidateWindow(
                start_sec=max(0, flag.window_start - pad_sec),
                end_sec=flag.window_end + pad_sec,
                labels=[flag.label],
                flags=[flag],
            )

    windows.append(current)

    log.info("triage_scanner.merge_complete",
             input_flags=len(flags), output_windows=len(windows),
             total_review_sec=sum(w.duration_sec for w in windows))
    return windows
