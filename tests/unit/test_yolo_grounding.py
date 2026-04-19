"""Unit tests for src/detection/yolo_grounding.py

Mocks YOLOv8 inference and FrameSampler output so tests run without
ultralytics, torch, or ffmpeg. Validates the per-type spatial rules and
the fail-open/fail-closed behavior.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.detection.frame_sampler import SampledFrame
from src.detection.models import Event, EventType
from src.detection.yolo_grounding import (
    _COCO_PERSON,
    _COCO_SPORTS_BALL,
    YoloGrounder,
)


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # Decoded-as-None in tests


@pytest.fixture
def sampler_stub():
    """A sampler that returns 3 fake frames around the requested center."""
    s = MagicMock()

    def _sample(center_sec, window_sec, interval_sec, duration_sec):
        return [
            SampledFrame(timestamp_sec=center_sec - window_sec, jpeg_bytes=FAKE_JPEG),
            SampledFrame(timestamp_sec=center_sec,              jpeg_bytes=FAKE_JPEG),
            SampledFrame(timestamp_sec=center_sec + window_sec, jpeg_bytes=FAKE_JPEG),
        ]
    s.sample_range.side_effect = _sample
    return s


@pytest.fixture
def sampler_stub_per_frame_positions():
    """Sampler + model pair that lets a test specify ball position per frame."""
    def _make(positions):
        # positions is a list of (x, y, conf) or None per frame
        s = MagicMock()

        def _sample(center_sec, window_sec, interval_sec, duration_sec):
            n = len(positions)
            if n == 1:
                return [SampledFrame(timestamp_sec=center_sec, jpeg_bytes=FAKE_JPEG)]
            step = (2 * window_sec) / (n - 1)
            return [
                SampledFrame(
                    timestamp_sec=center_sec - window_sec + i * step,
                    jpeg_bytes=FAKE_JPEG,
                )
                for i in range(n)
            ]
        s.sample_range.side_effect = _sample

        model = MagicMock()

        def _tensor(arr):
            t = MagicMock()
            t.cpu.return_value.numpy.return_value = np.array(arr)
            return t

        def _result_for(pos):
            boxes = MagicMock()
            if pos is None:
                boxes.cls = _tensor([])
                boxes.conf = _tensor([])
                boxes.xywhn = _tensor(np.zeros((0, 4)))
            else:
                x, y, conf = pos
                boxes.cls = _tensor([_COCO_SPORTS_BALL])
                boxes.conf = _tensor([conf])
                boxes.xywhn = _tensor(np.array([[x, y, 0.02, 0.02]]))
            r = MagicMock()
            r.boxes = boxes
            return r

        def _call(images, **kwargs):
            # One result per input frame, in order
            return [_result_for(p) for p in positions[: len(images)]]

        model.side_effect = _call
        return s, model
    return _make


def _make_yolo_model(ball_xywhn=None, ball_conf=0.9, person_boxes=()):
    """Build a fake ultralytics YOLO model whose call returns batch results."""
    def _result_for_frame():
        boxes = MagicMock()
        cls_list = []
        conf_list = []
        xywhn_list = []
        if ball_xywhn is not None:
            cls_list.append(_COCO_SPORTS_BALL)
            conf_list.append(ball_conf)
            xywhn_list.append(list(ball_xywhn))
        for p in person_boxes:
            cls_list.append(_COCO_PERSON)
            conf_list.append(p[4] if len(p) > 4 else 0.9)
            xywhn_list.append(list(p[:4]))

        def _tensor(arr):
            # Mimic torch tensor API that ultralytics returns
            t = MagicMock()
            t.cpu.return_value.numpy.return_value = np.array(arr)
            return t
        boxes.cls = _tensor(cls_list or [0])
        # Rebuild without the dummy 0 above if no classes were supplied
        if not cls_list:
            empty = _tensor([])
            boxes.cls = empty
            boxes.conf = empty
            boxes.xywhn = _tensor(np.zeros((0, 4)))
        else:
            boxes.conf = _tensor(conf_list)
            boxes.xywhn = _tensor(np.array(xywhn_list))
        r = MagicMock()
        r.boxes = boxes
        return r

    model = MagicMock()

    def _call(images, **kwargs):
        return [_result_for_frame() for _ in images]

    model.side_effect = _call
    return model


def _make_event(event_type: EventType, t: float = 120.0) -> Event:
    return Event(
        event_id=str(uuid.uuid4()),
        job_id="job-x",
        source_file="match.mp4",
        event_type=event_type,
        timestamp_start=t,
        timestamp_end=t + 3.0,
        confidence=0.8,
        reel_targets=[],
        is_goalkeeper_event=False,
        frame_start=0,
        frame_end=30,
    )


def _make_grounder(sampler, model, tmp_path=None, **kwargs):
    diag = tmp_path / "yolo_grounding.jsonl" if tmp_path else None
    defaults = dict(
        sampler=sampler,
        video_duration=600.0,
        model_path="/fake/yolov8m.pt",
        inference_size=640,
        use_gpu=False,
        ball_conf_threshold=0.15,
        n_frames=3,
        frame_span_sec=2.0,
        fail_open=True,
        diagnostics_path=diag,
        model=model,
    )
    defaults.update(kwargs)
    return YoloGrounder(**defaults)


# ---------------------------------------------------------------------------
# Monkey-patch cv2.imdecode so JPEG decoding returns a fake image array.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def stub_imdecode(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, "imdecode", lambda buf, flags: np.zeros((720, 1280, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Non-gated event types pass through unchanged.
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPassthrough:

    def test_goal_passes_through_untouched(self, sampler_stub):
        model = _make_yolo_model()  # Not called
        g = _make_grounder(sampler_stub, model)
        events = [_make_event(EventType.GOAL)]
        out = g.filter(events)
        assert len(out) == 1
        # Model should NOT have been called — goal is not a gated type
        model.assert_not_called()

    def test_shot_on_target_passes_through(self, sampler_stub):
        model = _make_yolo_model()
        g = _make_grounder(sampler_stub, model)
        events = [_make_event(EventType.SHOT_ON_TARGET)]
        out = g.filter(events)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# throw_in spatial rule
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestThrowIn:

    def test_ball_near_top_touchline_kept(self, sampler_stub):
        # y = 0.10 is within top 22% band
        model = _make_yolo_model(ball_xywhn=[0.5, 0.10, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert len(out) == 1

    def test_ball_near_bottom_touchline_kept(self, sampler_stub):
        model = _make_yolo_model(ball_xywhn=[0.5, 0.90, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert len(out) == 1

    def test_ball_in_midfield_rejected(self, sampler_stub):
        # y = 0.50 — nowhere near a touchline
        model = _make_yolo_model(ball_xywhn=[0.5, 0.50, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert out == []


# ---------------------------------------------------------------------------
# corner_kick spatial rule
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCornerKick:

    def test_ball_in_top_left_corner_kept(self, sampler_stub):
        model = _make_yolo_model(ball_xywhn=[0.05, 0.05, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.CORNER_KICK)])
        assert len(out) == 1

    def test_ball_in_bottom_right_corner_kept(self, sampler_stub):
        model = _make_yolo_model(ball_xywhn=[0.95, 0.95, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.CORNER_KICK)])
        assert len(out) == 1

    def test_ball_in_midfield_rejected(self, sampler_stub):
        model = _make_yolo_model(ball_xywhn=[0.5, 0.5, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.CORNER_KICK)])
        assert out == []

    def test_ball_near_touchline_but_not_corner_rejected(self, sampler_stub):
        # x centered, y near top: touchline yes, corner no.
        model = _make_yolo_model(ball_xywhn=[0.5, 0.10, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.CORNER_KICK)])
        assert out == []


# ---------------------------------------------------------------------------
# goal_kick spatial rule
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGoalKick:

    def test_ball_near_left_goal_line_middle_kept(self, sampler_stub):
        model = _make_yolo_model(ball_xywhn=[0.08, 0.50, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.GOAL_KICK)])
        assert len(out) == 1

    def test_ball_near_right_goal_line_middle_kept(self, sampler_stub):
        model = _make_yolo_model(ball_xywhn=[0.92, 0.50, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.GOAL_KICK)])
        assert len(out) == 1

    def test_ball_in_midfield_rejected(self, sampler_stub):
        model = _make_yolo_model(ball_xywhn=[0.5, 0.50, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.GOAL_KICK)])
        assert out == []

    def test_ball_at_corner_rejected(self, sampler_stub):
        # Near goal line but at top — outside vertical middle band (0.25–0.75)
        model = _make_yolo_model(ball_xywhn=[0.05, 0.10, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.GOAL_KICK)])
        assert out == []


# ---------------------------------------------------------------------------
# Fail-open semantics when YOLO can't find a ball
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFailOpen:

    def test_no_ball_detected_fail_open_keeps(self, sampler_stub):
        model = _make_yolo_model(ball_xywhn=None)
        g = _make_grounder(sampler_stub, model, fail_open=True)
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert len(out) == 1

    def test_no_ball_detected_fail_closed_drops(self, sampler_stub):
        model = _make_yolo_model(ball_xywhn=None)
        g = _make_grounder(sampler_stub, model, fail_open=False)
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert out == []

    def test_no_model_path_fails_open(self, sampler_stub):
        # model=None + model_path=None → load() returns None → no features
        g = _make_grounder(sampler_stub, None, model_path=None, fail_open=True)
        out = g.filter([_make_event(EventType.CORNER_KICK)])
        assert len(out) == 1  # fail-open


# ---------------------------------------------------------------------------
# Diagnostics output
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnyFrameMatches:
    """Event is kept if ANY sampled frame satisfies the spatial rule.

    This is Option A of the Run #33 post-mortem fix: the landmark moment
    (ball at touchline / corner / goal line) is fleeting, so a single
    frame matching is sufficient even if the others show the ball in
    open play.
    """

    def test_throw_in_kept_if_any_frame_near_touchline(
        self, sampler_stub_per_frame_positions
    ):
        # Frame 0: midfield; frame 1: midfield; frame 2: ball at touchline.
        # Old code picked best-confidence ball (all equal here → last wins
        # by ordering, which is touchline), but this test pins ANY-frame
        # semantics by putting the touchline moment in a non-last frame.
        positions = [
            (0.50, 0.50, 0.80),   # midfield, high conf
            (0.50, 0.10, 0.40),   # touchline, LOW conf
            (0.50, 0.50, 0.80),   # midfield, high conf
        ]
        sampler, model = sampler_stub_per_frame_positions(positions)
        g = _make_grounder(sampler, model, n_frames=3)
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert len(out) == 1

    def test_goal_kick_kept_if_any_frame_near_goal_line(
        self, sampler_stub_per_frame_positions
    ):
        positions = [
            (0.50, 0.50, 0.80),   # midfield
            (0.08, 0.50, 0.30),   # near left goal line (low conf)
            (0.60, 0.50, 0.80),   # midfield
        ]
        sampler, model = sampler_stub_per_frame_positions(positions)
        g = _make_grounder(sampler, model, n_frames=3)
        out = g.filter([_make_event(EventType.GOAL_KICK)])
        assert len(out) == 1

    def test_throw_in_rejected_if_no_frame_near_touchline(
        self, sampler_stub_per_frame_positions
    ):
        positions = [
            (0.50, 0.50, 0.80),
            (0.50, 0.45, 0.70),
            (0.50, 0.60, 0.70),
        ]
        sampler, model = sampler_stub_per_frame_positions(positions)
        g = _make_grounder(sampler, model, n_frames=3)
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert out == []


@pytest.mark.unit
class TestSpanSampling:
    """Sampling spans the whole event window, not ±1s around the start."""

    def test_span_sampler_called_with_event_center_and_half_span(self, sampler_stub):
        model = _make_yolo_model(ball_xywhn=[0.5, 0.10, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model)
        # Event: start=100, end=106 → span=6, center=103, half_window=3
        event = Event(
            event_id=str(uuid.uuid4()),
            job_id="j",
            source_file="m.mp4",
            event_type=EventType.THROW_IN,
            timestamp_start=100.0,
            timestamp_end=106.0,
            confidence=0.8,
            reel_targets=[],
            is_goalkeeper_event=False,
            frame_start=0,
            frame_end=180,
        )
        g.filter([event])
        call = sampler_stub.sample_range.call_args
        assert call.kwargs["center_sec"] == pytest.approx(103.0)
        assert call.kwargs["window_sec"] == pytest.approx(3.0)


@pytest.mark.unit
class TestDiagnostics:

    def test_writes_per_event_record(self, sampler_stub, tmp_path):
        import json
        model = _make_yolo_model(ball_xywhn=[0.5, 0.10, 0.02, 0.02])
        g = _make_grounder(sampler_stub, model, tmp_path=tmp_path)
        events = [
            _make_event(EventType.THROW_IN, t=100.0),   # kept
            _make_event(EventType.CORNER_KICK, t=200.0),  # rejected (midfield)
        ]
        # Second event has ball at top middle — corner rule fails
        g.filter(events)
        g.close()
        lines = (tmp_path / "yolo_grounding.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        recs = [json.loads(l) for l in lines]
        assert recs[0]["event_type"] == "throw_in"
        assert recs[0]["keep"] is True
        assert recs[1]["event_type"] == "corner_kick"
        assert recs[1]["keep"] is False
