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

# Triage labels — 8B identifies broad categories, active ones trigger 32B review
TRIAGE_LABELS = [
    "SET_PIECE", "SHOT_SAVE", "GOAL",
    "ATTACK", "PLAY", "DEAD",
]

# Labels that trigger a candidate window for 32B review
ACTIVE_LABELS = frozenset({"SET_PIECE", "SHOT_SAVE", "GOAL", "ATTACK"})


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

Classify what is happening. Choose ONE label:

- SET_PIECE: Any restart of play OR the setup before a restart. Use this \
whenever you see ANY of these: a player near the corner flag, a player \
walking to the sideline with a ball, a player holding a ball with both \
hands at the sideline, a GK placing or about to kick the ball from inside \
the penalty area, a ball on the ground with a defensive wall forming, \
players gathering around the center circle for kickoff, a player standing \
over a stationary ball outside the run of play. SET_PIECE covers the \
WHOLE restart phase (setup through kick), not only the instant of contact.
- SHOT_SAVE: A shot on goal, the ball flying toward/hitting the goal frame, \
or the goalkeeper diving/catching/punching/parrying a shot.
- GOAL: Ball clearly inside the net, or players celebrating with arms \
raised and running toward teammates.
- ATTACK: Ball in the attacking third with ongoing play — dangerous build-up, \
crosses into the box, or players converging on goal. No restart is happening.
- PLAY: Normal flowing midfield play — passing, dribbling, the ball is \
visibly on the pitch between the two penalty areas and no restart or \
shot is happening.
- DEAD: The ball is not visibly in play and no player is about to restart \
it — referee stoppage, halftime, substitution, replay graphic, or players \
milling around waiting. Use DEAD only if no SET_PIECE signal is visible.

Priority rules (most important first):
1. If you see ANY restart signal (corner flag, sideline throw-in, GK with \
ball in the six-yard box, wall forming, kickoff circle) → SET_PIECE.
2. If you see a shot or a GK contacting the ball → SHOT_SAVE.
3. If the ball is out of play with no one about to restart → DEAD.
4. If play is flowing in the attacking third → ATTACK.
5. Otherwise → PLAY.

Tie-breaker: prefer SET_PIECE > SHOT_SAVE > ATTACK > DEAD > PLAY. \
It is better to flag a moment than to miss it.

Respond with ONLY a JSON object:
{{"label": "SET_PIECE", "ball_zone": "left_third"}}

ball_zone: "left_third", "middle", or "right_third" (where the action is).
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
        # Run #6: track EVERY window (including PLAY/DEAD) to detect
        # DEAD -> non-DEAD transitions that signal set-piece restarts.
        all_window_records: list[TriageFlag] = []
        label_counts: dict[str, int] = {}
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
                label_counts[label] = label_counts.get(label, 0) + 1
                record = TriageFlag(
                    center_sec=center,
                    label=label,
                    window_start=window_start,
                    window_end=window_end,
                    ball_zone=ball_zone,
                )
                all_window_records.append(record)
                if label in ACTIVE_LABELS:
                    flags.append(record)

            window_idx += 1
            if progress_callback and total_windows > 0:
                progress_callback(min(1.0, window_idx / total_windows))

            # Log progress every 100 windows
            if window_idx % 100 == 0:
                log.info("triage_scanner.progress",
                         window=window_idx, total=total_windows,
                         flags_so_far=len(flags),
                         label_dist=dict(label_counts))

            window_start += self._step_sec

        # Run #6: synthesize restart flags at DEAD -> non-DEAD transitions.
        # These are set-piece candidates the 8B is classifying as PLAY/DEAD
        # but which hold corner_kick / goal_kick / throw_in / free_kick events.
        restart_flags = _synthesize_restart_flags(all_window_records)
        if restart_flags:
            log.info("triage_scanner.restart_flags_synthesized",
                     count=len(restart_flags))
            flags.extend(restart_flags)

        log.info("triage_scanner.scan_complete",
                 total_windows=window_idx, flags=len(flags),
                 restart_flags=len(restart_flags),
                 label_distribution=dict(label_counts))
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


def _synthesize_restart_flags(
    records: list[TriageFlag],
    min_dead_windows: int = 2,
) -> list[TriageFlag]:
    """Find DEAD -> non-DEAD transitions and emit synthetic SET_PIECE flags.

    Soccer restarts (corner, throw-in, goal_kick, free_kick) start with the
    ball dead, then the kick/throw, then play resumes. The 8B triage often
    labels the dead portion as DEAD and the resume as PLAY, missing the
    set-piece itself. This helper walks the full label sequence and emits
    a SET_PIECE flag at every boundary where at least ``min_dead_windows``
    consecutive DEAD windows are followed by a non-DEAD window.

    Args:
        records: ALL window records from the scan (active + inactive),
            in chronological order.
        min_dead_windows: Minimum consecutive DEAD count required before
            a transition qualifies.

    Returns:
        List of synthetic TriageFlag with label SET_PIECE positioned at the
        transition moment, inheriting ball_zone from the first non-DEAD window.
    """
    if not records:
        return []

    synthetic: list[TriageFlag] = []
    dead_run = 0
    dead_start_record: Optional[TriageFlag] = None

    for rec in records:
        if rec.label == "DEAD":
            if dead_run == 0:
                dead_start_record = rec
            dead_run += 1
            continue

        # Non-DEAD: check if we just exited a DEAD run
        if dead_run >= min_dead_windows and dead_start_record is not None:
            # Span the transition: from start of DEAD run to end of current window.
            start = dead_start_record.window_start
            end = rec.window_end
            center = (start + end) / 2
            synthetic.append(TriageFlag(
                center_sec=center,
                label="SET_PIECE",
                window_start=start,
                window_end=end,
                ball_zone=rec.ball_zone,
            ))
        dead_run = 0
        dead_start_record = None

    return synthetic


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
