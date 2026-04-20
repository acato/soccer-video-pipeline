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
    TrajectorySignature,
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
class TestCustomClassIds:
    """Soccer-tuned models use ball=0 and dedicated goalkeeper class."""

    def _soccer_model(self, *, ball_xy, gk_xy=None):
        """Build a fake YOLO result where ball=class 0 and gk=class 1."""
        def _tensor(arr):
            t = MagicMock()
            t.cpu.return_value.numpy.return_value = np.array(arr)
            return t
        boxes = MagicMock()
        cls_list, conf_list, xywhn_list = [0], [0.9], [list(ball_xy)]
        if gk_xy is not None:
            cls_list.append(1)
            conf_list.append(0.8)
            xywhn_list.append(list(gk_xy))
        boxes.cls = _tensor(cls_list)
        boxes.conf = _tensor(conf_list)
        boxes.xywhn = _tensor(np.array(xywhn_list))
        r = MagicMock()
        r.boxes = boxes
        model = MagicMock()
        model.side_effect = lambda images, **kw: [r for _ in images]
        return model

    def test_ball_at_class_id_zero_detected(self, sampler_stub):
        # Ball near top touchline, at class ID 0 (soccer model schema).
        model = self._soccer_model(ball_xy=[0.5, 0.10, 0.02, 0.02])
        g = _make_grounder(
            sampler_stub, model,
            ball_class_id=0,
            person_class_ids=(1, 2, 3),
        )
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert len(out) == 1

    def test_ball_at_coco_id_ignored_when_class_overridden(self, sampler_stub):
        # COCO sports_ball=32 at touchline, but we only accept class 0.
        # Without a class-0 detection, gate sees no ball → fail-open keeps.
        model = _make_yolo_model(ball_xywhn=[0.5, 0.10, 0.02, 0.02])
        g = _make_grounder(
            sampler_stub, model,
            ball_class_id=0,  # Reject COCO schema
            person_class_ids=(1, 2, 3),
            fail_open=False,  # Prove the ball was NOT picked up
        )
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert out == []

    def test_gk_detections_captured_when_gk_class_configured(
        self, sampler_stub, tmp_path
    ):
        import json
        model = self._soccer_model(
            ball_xy=[0.5, 0.10, 0.02, 0.02],
            gk_xy=[0.9, 0.5, 0.05, 0.12],
        )
        g = _make_grounder(
            sampler_stub, model, tmp_path=tmp_path,
            ball_class_id=0,
            person_class_ids=(1, 2, 3),
            gk_class_ids=(1,),
        )
        g.filter([_make_event(EventType.THROW_IN)])
        g.close()
        recs = [
            json.loads(l)
            for l in (tmp_path / "yolo_grounding.jsonl").read_text().splitlines()
        ]
        assert recs[0]["features"]["n_gk_detections"] > 0
        assert recs[0]["features"]["gk_positions"][0]["x"] == pytest.approx(0.9)

    def test_gk_detections_empty_when_gk_class_not_configured(
        self, sampler_stub, tmp_path
    ):
        import json
        model = self._soccer_model(
            ball_xy=[0.5, 0.10, 0.02, 0.02],
            gk_xy=[0.9, 0.5, 0.05, 0.12],
        )
        g = _make_grounder(
            sampler_stub, model, tmp_path=tmp_path,
            ball_class_id=0,
            person_class_ids=(1, 2, 3),
            # gk_class_ids left at default () — no GK tracking
        )
        g.filter([_make_event(EventType.THROW_IN)])
        g.close()
        recs = [
            json.loads(l)
            for l in (tmp_path / "yolo_grounding.jsonl").read_text().splitlines()
        ]
        assert recs[0]["features"]["n_gk_detections"] == 0


@pytest.mark.unit
class TestGkProximityGate:
    """Run #36: GK events kept only if ball comes close to a GK in-frame.

    Uses the soccer-tuned class schema (ball=0, goalkeeper=1).
    """

    def _build_model(self, per_frame_boxes):
        """per_frame_boxes: list[list[(cls, x, y, conf)]] — one list per frame."""
        def _tensor(arr):
            t = MagicMock()
            t.cpu.return_value.numpy.return_value = np.array(arr)
            return t

        def _result_for(boxes_list):
            boxes = MagicMock()
            if not boxes_list:
                boxes.cls = _tensor([])
                boxes.conf = _tensor([])
                boxes.xywhn = _tensor(np.zeros((0, 4)))
            else:
                boxes.cls = _tensor([b[0] for b in boxes_list])
                boxes.conf = _tensor([b[3] for b in boxes_list])
                boxes.xywhn = _tensor(
                    np.array([[b[1], b[2], 0.05, 0.05] for b in boxes_list])
                )
            r = MagicMock()
            r.boxes = boxes
            return r

        model = MagicMock()
        model.side_effect = lambda images, **kw: [
            _result_for(per_frame_boxes[i]) for i in range(len(images))
        ]
        return model

    def _gk_grounder(self, sampler, model, **extra):
        return _make_grounder(
            sampler, model,
            ball_class_id=0,
            person_class_ids=(1, 2, 3),
            gk_class_ids=(1,),
            **extra,
        )

    def test_ball_next_to_gk_kept(self, sampler_stub):
        # Both ball and GK at (0.5, 0.5) in all 3 frames → distance = 0 → keep.
        per_frame = [
            [(0, 0.5, 0.5, 0.9), (1, 0.5, 0.5, 0.8)],
        ] * 3
        model = self._build_model(per_frame)
        g = self._gk_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.CATCH)])
        assert len(out) == 1

    def test_ball_far_from_gk_rejected(self, sampler_stub):
        # Ball at (0.9, 0.9), GK at (0.1, 0.1) → dist ≈ 1.13 > 0.20 → reject.
        per_frame = [
            [(0, 0.9, 0.9, 0.9), (1, 0.1, 0.1, 0.8)],
        ] * 3
        model = self._build_model(per_frame)
        g = self._gk_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.SHOT_STOP_DIVING)])
        assert out == []

    def test_no_gk_detected_fails_open(self, sampler_stub):
        # Ball detected, no GK at all → fail-open keep.
        per_frame = [
            [(0, 0.5, 0.5, 0.9)],
        ] * 3
        model = self._build_model(per_frame)
        g = self._gk_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.CATCH)])
        assert len(out) == 1

    def test_ball_and_gk_never_same_frame_fails_open(self, sampler_stub):
        # Frame 1: ball only. Frame 2: GK only. Frame 3: nothing. → fail-open.
        per_frame = [
            [(0, 0.5, 0.5, 0.9)],           # ball only
            [(1, 0.5, 0.5, 0.8)],           # gk only
            [(0, 0.5, 0.5, 0.9)],           # ball only (so ball_detected=True)
        ]
        model = self._build_model(per_frame)
        g = self._gk_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.CATCH)])
        assert len(out) == 1

    def test_multiple_gks_picks_nearest(self, sampler_stub):
        # Ball at (0.5, 0.5). Two GKs: (0.9, 0.9) far, (0.55, 0.55) close.
        # Nearest dist ≈ 0.071 < 0.20 → keep.
        per_frame = [
            [(0, 0.5, 0.5, 0.9),
             (1, 0.9, 0.9, 0.8),
             (1, 0.55, 0.55, 0.8)],
        ] * 3
        model = self._build_model(per_frame)
        g = self._gk_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.CATCH)])
        assert len(out) == 1

    def test_tight_threshold_rejects_moderately_close(self, sampler_stub):
        # Distance ≈ 0.141 — between 0.10 and 0.20. Tight threshold rejects.
        per_frame = [
            [(0, 0.5, 0.5, 0.9), (1, 0.6, 0.6, 0.8)],
        ] * 3
        model = self._build_model(per_frame)
        g = self._gk_grounder(
            sampler_stub, model, gk_proximity_threshold=0.10
        )
        out = g.filter([_make_event(EventType.SHOT_STOP_DIVING)])
        assert out == []

    def test_non_gk_event_not_affected_by_proximity_rule(self, sampler_stub):
        # An event that's a landmark type (throw_in) should route through
        # the touchline rule, not the GK rule — even with GK visible.
        per_frame = [
            [(0, 0.5, 0.10, 0.9), (1, 0.5, 0.5, 0.8)],  # ball near top touchline
        ] * 3
        model = self._build_model(per_frame)
        g = self._gk_grounder(sampler_stub, model)
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert len(out) == 1


@pytest.mark.unit
class TestGkSpecificSampling:
    """Run #36b: GK events sample more frames over a wider window at higher
    inference resolution than landmark events."""

    def _sampler_and_model(self):
        s = MagicMock()
        # Record call args so tests can inspect them
        captured = {"calls": []}

        def _sample(center_sec, window_sec, interval_sec, duration_sec):
            captured["calls"].append({
                "center_sec": center_sec,
                "window_sec": window_sec,
                "interval_sec": interval_sec,
            })
            # Return 3 dummy frames regardless (quantity doesn't matter for
            # sampling assertion — we inspect call args, not return length)
            return [
                SampledFrame(timestamp_sec=center_sec, jpeg_bytes=FAKE_JPEG)
                for _ in range(3)
            ]
        s.sample_range.side_effect = _sample

        def _tensor(arr):
            t = MagicMock()
            t.cpu.return_value.numpy.return_value = np.array(arr)
            return t
        boxes = MagicMock()
        boxes.cls = _tensor([])
        boxes.conf = _tensor([])
        boxes.xywhn = _tensor(np.zeros((0, 4)))
        r = MagicMock()
        r.boxes = boxes
        model = MagicMock()
        model_captured = {"imgsz": []}

        def _call(images, **kw):
            model_captured["imgsz"].append(kw.get("imgsz"))
            return [r for _ in images]
        model.side_effect = _call
        return s, model, captured, model_captured

    def test_gk_event_uses_wider_window(self):
        sampler, model, captured, _ = self._sampler_and_model()
        g = _make_grounder(
            sampler, model,
            ball_class_id=0, person_class_ids=(1, 2, 3), gk_class_ids=(1,),
            n_frames=5, frame_span_sec=2.0,
            gk_n_frames=10, gk_min_span_sec=6.0,
        )
        # Event span of 1s — shorter than both min_span (2s) and gk_min_span (6s).
        # Non-GK event (throw_in) uses 2s min → window_sec = 1.0
        # GK event (catch) uses 6s min → window_sec = 3.0
        event_tin = _make_event(EventType.THROW_IN, t=100.0)
        event_tin = event_tin.__class__(**{**event_tin.__dict__,
                                           "timestamp_end": 101.0})
        event_gk = _make_event(EventType.CATCH, t=200.0)
        event_gk = event_gk.__class__(**{**event_gk.__dict__,
                                         "timestamp_end": 201.0})
        g.filter([event_tin, event_gk])
        assert captured["calls"][0]["window_sec"] == pytest.approx(1.0)
        assert captured["calls"][1]["window_sec"] == pytest.approx(3.0)

    def test_gk_event_uses_higher_inference_size(self):
        sampler, model, _, model_captured = self._sampler_and_model()
        g = _make_grounder(
            sampler, model,
            ball_class_id=0, person_class_ids=(1, 2, 3), gk_class_ids=(1,),
            inference_size=640,
            gk_inference_size=1280,
        )
        g.filter([
            _make_event(EventType.THROW_IN, t=10.0),
            _make_event(EventType.CATCH, t=20.0),
        ])
        # First event (throw_in) at 640, second (catch) at 1280.
        assert model_captured["imgsz"][0] == 640
        assert model_captured["imgsz"][1] == 1280

    def test_non_gk_event_uses_default_sampling(self):
        sampler, model, captured, model_captured = self._sampler_and_model()
        g = _make_grounder(
            sampler, model,
            ball_class_id=0, person_class_ids=(1, 2, 3), gk_class_ids=(1,),
            n_frames=5, frame_span_sec=2.0, inference_size=640,
            gk_n_frames=10, gk_min_span_sec=6.0, gk_inference_size=1280,
        )
        # _make_event default span is 3s (t → t+3). With min_span=2.0 for
        # landmark events, span stays at 3.0 → window_sec = 1.5. This is
        # NOT 3.0 (which would indicate gk_min_span=6.0 was applied).
        g.filter([_make_event(EventType.GOAL_KICK, t=100.0)])
        assert captured["calls"][0]["window_sec"] == pytest.approx(1.5)
        assert model_captured["imgsz"][0] == 640


@pytest.mark.unit
class TestTrajectorySignature:
    """Run #37: ball motion pre/post GK contact classifies the save type."""

    def _sampler_with_timed_frames(self, n, interval=0.5):
        """Returns a sampler that produces n frames at fixed intervals."""
        s = MagicMock()

        def _sample(center_sec, window_sec, interval_sec, duration_sec):
            return [
                SampledFrame(
                    timestamp_sec=center_sec - window_sec + i * interval,
                    jpeg_bytes=FAKE_JPEG,
                )
                for i in range(n)
            ]
        s.sample_range.side_effect = _sample
        return s

    def _model_per_frame(self, positions_by_frame):
        """positions_by_frame: list[list[(cls,x,y,conf)]] — one per frame."""
        def _tensor(arr):
            t = MagicMock()
            t.cpu.return_value.numpy.return_value = np.array(arr)
            return t

        def _result_for(boxes_list):
            boxes = MagicMock()
            if not boxes_list:
                boxes.cls = _tensor([])
                boxes.conf = _tensor([])
                boxes.xywhn = _tensor(np.zeros((0, 4)))
            else:
                boxes.cls = _tensor([b[0] for b in boxes_list])
                boxes.conf = _tensor([b[3] for b in boxes_list])
                boxes.xywhn = _tensor(
                    np.array([[b[1], b[2], 0.05, 0.05] for b in boxes_list])
                )
            r = MagicMock()
            r.boxes = boxes
            return r

        model = MagicMock()
        model.side_effect = lambda images, **kw: [
            _result_for(positions_by_frame[i]) for i in range(len(images))
        ]
        return model

    def _gk_grounder(self, sampler, model, **extra):
        # Trajectory tests feed 5 positions per event; make sure all 5 reach
        # the model by setting n_frames=5 (default is 3, which would clip).
        defaults = dict(
            ball_class_id=0, person_class_ids=(1, 2, 3), gk_class_ids=(1,),
            n_frames=5,
        )
        defaults.update(extra)
        return _make_grounder(sampler, model, **defaults)

    def test_parry_signature_keeps_and_tags(self):
        # 5 frames, GK at (0.9, 0.5). Ball approaches GK from left, then
        # reverses back to left after contact — sharp direction change.
        sampler = self._sampler_with_timed_frames(5)
        positions = [
            [(0, 0.50, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
            [(0, 0.70, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
            [(0, 0.88, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],  # contact
            [(0, 0.70, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],  # reversed
            [(0, 0.50, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
        ]
        model = self._model_per_frame(positions)
        g = self._gk_grounder(sampler, model)
        out = g.filter([_make_event(EventType.SHOT_STOP_DIVING)])
        assert len(out) == 1  # kept
        # Can't directly read the signature from the output Event, but we
        # can open the diag file or inspect via the grounder's last call.

    def test_catch_signature_keeps(self):
        # Ball flies in, then stops near GK (speed ratio << 0.3).
        sampler = self._sampler_with_timed_frames(5)
        positions = [
            [(0, 0.30, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
            [(0, 0.50, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
            [(0, 0.88, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],  # contact
            [(0, 0.89, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],  # held
            [(0, 0.89, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
        ]
        model = self._model_per_frame(positions)
        g = self._gk_grounder(sampler, model)
        out = g.filter([_make_event(EventType.CATCH)])
        assert len(out) == 1

    def test_missed_signature_rejects_despite_proximity(self):
        # Ball flies through the GK location at constant speed — no touch.
        # Proximity would keep (ball is near GK at contact frame) but
        # trajectory MISSED should override and REJECT.
        sampler = self._sampler_with_timed_frames(5)
        positions = [
            [(0, 0.10, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
            [(0, 0.30, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
            [(0, 0.89, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],  # near GK
            [(0, 0.70, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],  # ball continues
            [(0, 0.50, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],  # (wrapped back)
        ]
        # Actually, a "pass through" test needs same direction continuation.
        # Use a cleanly constant-direction pass: ball moves left→right
        # through the GK location at steady speed.
        positions = [
            [(0, 0.10, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
            [(0, 0.30, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
            [(0, 0.89, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],  # at GK
            [(0, 0.70, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
            # ... hmm reversed. Let me do a real pass-through.
        ]
        # Real pass-through: ball continues at constant speed across GK.
        positions = [
            [(0, 0.10, 0.50, 0.9), (1, 0.50, 0.50, 0.8)],
            [(0, 0.30, 0.50, 0.9), (1, 0.50, 0.50, 0.8)],
            [(0, 0.49, 0.50, 0.9), (1, 0.50, 0.50, 0.8)],  # contact
            [(0, 0.70, 0.50, 0.9), (1, 0.50, 0.50, 0.8)],
            [(0, 0.90, 0.50, 0.9), (1, 0.50, 0.50, 0.8)],
        ]
        model = self._model_per_frame(positions)
        g = self._gk_grounder(sampler, model)
        out = g.filter([_make_event(EventType.SHOT_STOP_DIVING)])
        assert out == []  # rejected by trajectory MISSED override

    def test_insufficient_data_falls_back_to_proximity(self):
        # Only 2 ball positions → not enough for trajectory. Proximity is
        # within threshold → should still KEEP via proximity path.
        sampler = self._sampler_with_timed_frames(3)
        positions = [
            [(0, 0.89, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
            [(1, 0.90, 0.50, 0.8)],  # GK only, no ball in frame 2
            [(0, 0.91, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
        ]
        model = self._model_per_frame(positions)
        g = self._gk_grounder(sampler, model)
        out = g.filter([_make_event(EventType.CATCH)])
        assert len(out) == 1  # kept via proximity

    def test_stationary_ball_is_insufficient_data(self):
        # Ball at exactly same position every frame — no motion. Should NOT
        # be classified as MISSED. Proximity keeps.
        sampler = self._sampler_with_timed_frames(5)
        positions = [
            [(0, 0.89, 0.50, 0.9), (1, 0.90, 0.50, 0.8)],
        ] * 5
        model = self._model_per_frame(positions)
        g = self._gk_grounder(sampler, model)
        out = g.filter([_make_event(EventType.CATCH)])
        assert len(out) == 1  # kept by proximity (trajectory is INSUFFICIENT_DATA)

    def test_trajectory_disabled_uses_only_proximity(self):
        # Constant-speed pass-through would normally be MISSED-rejected.
        # With trajectory disabled, proximity-keep stands.
        sampler = self._sampler_with_timed_frames(5)
        positions = [
            [(0, 0.10, 0.50, 0.9), (1, 0.50, 0.50, 0.8)],
            [(0, 0.30, 0.50, 0.9), (1, 0.50, 0.50, 0.8)],
            [(0, 0.49, 0.50, 0.9), (1, 0.50, 0.50, 0.8)],
            [(0, 0.70, 0.50, 0.9), (1, 0.50, 0.50, 0.8)],
            [(0, 0.90, 0.50, 0.9), (1, 0.50, 0.50, 0.8)],
        ]
        model = self._model_per_frame(positions)
        g = self._gk_grounder(sampler, model, trajectory_enabled=False)
        out = g.filter([_make_event(EventType.SHOT_STOP_DIVING)])
        assert len(out) == 1  # kept — trajectory override disabled

    def test_non_gk_event_not_affected_by_trajectory(self):
        # Even a clean "MISSED" signature wouldn't affect a throw_in event
        # since it's not a GK type.
        sampler = self._sampler_with_timed_frames(5)
        positions = [
            # ball right on the touchline — should be kept by landmark rule
            [(0, 0.50, 0.05, 0.9), (1, 0.90, 0.50, 0.8)],
        ] * 5
        model = self._model_per_frame(positions)
        g = self._gk_grounder(sampler, model)
        out = g.filter([_make_event(EventType.THROW_IN)])
        assert len(out) == 1


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
