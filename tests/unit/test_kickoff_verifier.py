"""Unit tests for src/detection/kickoff_verifier.py.

Mocks the YOLO model + FrameSampler since the real ones need GPU + video files.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.detection import kickoff_verifier as kv
from src.detection.models import Event, EventType


# ── Fakes ──────────────────────────────────────────────────────────────────
class _FakeBoxes:
    """Mimics the slice of ultralytics Boxes used by the verifier."""

    def __init__(self, classes: list[int], confs: list[float],
                 xywhn: list[tuple[float, float, float, float]]):
        self.cls = MagicMock()
        self.cls.cpu.return_value.numpy.return_value = np.array(classes)
        self.conf = MagicMock()
        self.conf.cpu.return_value.numpy.return_value = np.array(confs)
        self.xywhn = MagicMock()
        self.xywhn.cpu.return_value.numpy.return_value = np.array(xywhn)


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def _kickoff_scene_boxes() -> _FakeBoxes:
    """Ball at frame center, 10 players ~5/5 across halves."""
    classes = [32] + [0] * 10
    confs = [0.95] + [0.85] * 10
    xywhn = [(0.51, 0.50, 0.02, 0.02)]
    # 5 players in left half, 5 in right
    for x in (0.1, 0.2, 0.3, 0.4, 0.45):
        xywhn.append((x, 0.5, 0.05, 0.10))
    for x in (0.55, 0.6, 0.7, 0.8, 0.9):
        xywhn.append((x, 0.5, 0.05, 0.10))
    return _FakeBoxes(classes, confs, xywhn)


def _saved_shot_scene_boxes() -> _FakeBoxes:
    """Ball at one penalty area; 10 players bunched in left half."""
    classes = [32] + [0] * 10
    confs = [0.85] + [0.85] * 10
    xywhn = [(0.15, 0.55, 0.02, 0.02)]
    for x in (0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35):
        xywhn.append((x, 0.5, 0.05, 0.10))
    return _FakeBoxes(classes, confs, xywhn)


def _close_up_scene_boxes() -> _FakeBoxes:
    """Replay close-up: only 3 players, no ball visible."""
    classes = [0, 0, 0]
    confs = [0.85, 0.80, 0.78]
    xywhn = [(0.5, 0.5, 0.4, 0.6), (0.45, 0.5, 0.3, 0.5), (0.55, 0.5, 0.3, 0.5)]
    return _FakeBoxes(classes, confs, xywhn)


def _make_sampler(yields_a_frame: bool = True) -> MagicMock:
    """FrameSampler stub. sample_range() returns a fake jpeg."""
    sampler = MagicMock()
    if yields_a_frame:
        frame = MagicMock()
        frame.jpeg_bytes = b"\xff\xd8\xff\xe0"
        frame.timestamp_sec = 0.0
        sampler.sample_range.return_value = [frame]
    else:
        sampler.sample_range.return_value = []
    return sampler


def _make_yolo_model(per_call_results: list[_FakeResult]) -> MagicMock:
    """YOLO model stub: returns the next _FakeResult per call."""
    model = MagicMock()
    calls = iter(per_call_results)

    def _call(images, **kwargs):
        try:
            return [next(calls)]
        except StopIteration:
            return [_FakeResult(None)]
    model.side_effect = _call
    return model


def _outcome_goal(t_start: float = 100.0, t_end: float | None = None) -> Event:
    return Event(
        event_id=f"e_{int(t_start)}",
        job_id="j1",
        source_file="m.mp4",
        event_type=EventType.GOAL,
        timestamp_start=t_start,
        timestamp_end=t_end if t_end is not None else t_start + 15.0,
        confidence=0.9,
        reel_targets=["highlights"],
        frame_start=int(t_start * 30),
        frame_end=int((t_end if t_end is not None else t_start + 15.0) * 30),
        metadata={"detection_method": "shot_outcome",
                  "promoted_from_shot": True,
                  "source_event_type": "shot_on_target"},
    )


# ── Direct probe-frame tests (don't load YOLO) ─────────────────────────────
@pytest.mark.unit
class TestProbeFrame:

    def test_kickoff_scene_passes_all_three_signals(self, monkeypatch):
        sampler = _make_sampler()
        result = _FakeResult(_kickoff_scene_boxes())
        model = _make_yolo_model([result])
        # Bypass cv2 imdecode by stubbing _yolo_inference
        monkeypatch.setattr(kv, "_yolo_inference",
                            lambda model, jpeg, **kw: result)
        probe = kv._probe_frame(
            model, sampler, video_duration=200.0, target_sec=130.0,
            ball_class_id=32, person_class_ids=frozenset({0}),
            ball_conf=0.15, inference_size=640, use_gpu=False,
            central_box=0.10,
        )
        assert probe.has_yolo_read is True
        assert probe.ball_central is True
        assert probe.n_persons == 10
        assert probe.half_imbalance == 0.0  # 5/5

    def test_saved_shot_fails_half_imbalance(self, monkeypatch):
        sampler = _make_sampler()
        result = _FakeResult(_saved_shot_scene_boxes())
        monkeypatch.setattr(kv, "_yolo_inference",
                            lambda model, jpeg, **kw: result)
        probe = kv._probe_frame(
            MagicMock(), sampler, video_duration=200.0, target_sec=130.0,
            ball_class_id=32, person_class_ids=frozenset({0}),
            ball_conf=0.15, inference_size=640, use_gpu=False,
            central_box=0.10,
        )
        assert probe.has_yolo_read is True
        assert probe.ball_central is False  # ball at 0.15
        assert probe.n_persons == 10
        assert probe.half_imbalance == 1.0  # all 10 in left half

    def test_close_up_fails_min_persons(self, monkeypatch):
        sampler = _make_sampler()
        result = _FakeResult(_close_up_scene_boxes())
        monkeypatch.setattr(kv, "_yolo_inference",
                            lambda model, jpeg, **kw: result)
        probe = kv._probe_frame(
            MagicMock(), sampler, video_duration=200.0, target_sec=130.0,
            ball_class_id=32, person_class_ids=frozenset({0}),
            ball_conf=0.15, inference_size=640, use_gpu=False,
            central_box=0.10,
        )
        assert probe.has_yolo_read is True
        assert probe.n_persons == 3  # below typical min_persons=8

    def test_no_frame_returns_no_read(self):
        sampler = _make_sampler(yields_a_frame=False)
        probe = kv._probe_frame(
            MagicMock(), sampler, video_duration=200.0, target_sec=130.0,
            ball_class_id=32, person_class_ids=frozenset({0}),
            ball_conf=0.15, inference_size=640, use_gpu=False,
            central_box=0.10,
        )
        assert probe.has_yolo_read is False


# ── End-to-end verify_outcome_goals tests (mock YOLO load + inference) ─────
@pytest.mark.unit
class TestVerifyOutcomeGoals:

    def test_no_targets_passes_through(self):
        events = [
            Event(event_id="s1", job_id="j", source_file="m.mp4",
                  event_type=EventType.SHOT_ON_TARGET, timestamp_start=10,
                  timestamp_end=11, confidence=0.7, reel_targets=["highlights"],
                  frame_start=300, frame_end=330),
        ]
        out, stats = kv.verify_outcome_goals(
            events, sampler=_make_sampler(), video_duration=100.0,
            model_path=None,
        )
        assert out == events
        assert stats["checked"] == 0

    def test_no_model_path_keeps_targets_unchanged(self):
        events = [_outcome_goal()]
        out, stats = kv.verify_outcome_goals(
            events, sampler=_make_sampler(), video_duration=200.0,
            model_path=None,  # No model = no verification
        )
        assert len(out) == 1
        assert out[0].event_type == EventType.GOAL

    def test_kickoff_match_keeps_goal(self, monkeypatch):
        events = [_outcome_goal(t_start=100.0)]
        # Inject YOLO load + inference stubs
        match_result = _FakeResult(_kickoff_scene_boxes())
        monkeypatch.setattr(kv, "_yolo_inference",
                            lambda model, jpeg, **kw: match_result)

        # Stub the YOLO loader inside verify_outcome_goals (ultralytics import)
        class _FakeYOLO:
            def __init__(self, *a, **kw): pass
        monkeypatch.setitem(__import__("sys").modules, "ultralytics",
                            type("M", (), {"YOLO": _FakeYOLO}))

        out, stats = kv.verify_outcome_goals(
            events, sampler=_make_sampler(), video_duration=300.0,
            model_path="/fake/yolo.pt", min_persons=8, max_half_imbalance=0.30,
        )
        assert stats["checked"] == 1
        assert stats["kept_match"] == 1
        assert stats["dropped_no_kickoff"] == 0
        assert len(out) == 1

    def test_saved_shot_drops_goal_keeps_shot(self, monkeypatch):
        shot = Event(event_id="s1", job_id="j", source_file="m.mp4",
                     event_type=EventType.SHOT_ON_TARGET, timestamp_start=100,
                     timestamp_end=101, confidence=0.8,
                     reel_targets=["highlights"],
                     frame_start=3000, frame_end=3030)
        goal = _outcome_goal(t_start=100.0)
        events = [shot, goal]
        result = _FakeResult(_saved_shot_scene_boxes())
        monkeypatch.setattr(kv, "_yolo_inference",
                            lambda model, jpeg, **kw: result)

        class _FakeYOLO:
            def __init__(self, *a, **kw): pass
        monkeypatch.setitem(__import__("sys").modules, "ultralytics",
                            type("M", (), {"YOLO": _FakeYOLO}))

        out, stats = kv.verify_outcome_goals(
            events, sampler=_make_sampler(), video_duration=300.0,
            model_path="/fake/yolo.pt", fail_open=False,
        )
        assert stats["dropped_no_kickoff"] == 1
        assert stats["kept_match"] == 0
        # Shot survives, goal does not
        types = [e.event_type for e in out]
        assert EventType.SHOT_ON_TARGET in types
        assert EventType.GOAL not in types

    def test_fail_open_keeps_goal_when_no_yolo_read(self, monkeypatch):
        events = [_outcome_goal()]
        # YOLO returns None for every call → no_read on every probe
        monkeypatch.setattr(kv, "_yolo_inference",
                            lambda model, jpeg, **kw: None)

        class _FakeYOLO:
            def __init__(self, *a, **kw): pass
        monkeypatch.setitem(__import__("sys").modules, "ultralytics",
                            type("M", (), {"YOLO": _FakeYOLO}))

        out, stats = kv.verify_outcome_goals(
            events, sampler=_make_sampler(), video_duration=300.0,
            model_path="/fake/yolo.pt", fail_open=True,
        )
        assert stats["kept_fail_open"] == 1
        assert len(out) == 1

    def test_direct_goal_bypasses_verifier(self, monkeypatch):
        # A goal with detection_method != "shot_outcome" must pass through.
        direct_goal = Event(
            event_id="g_direct", job_id="j", source_file="m.mp4",
            event_type=EventType.GOAL, timestamp_start=200, timestamp_end=215,
            confidence=0.9, reel_targets=["highlights"],
            frame_start=6000, frame_end=6450,
            metadata={"detection_method": "dual_pass"},
        )
        out, stats = kv.verify_outcome_goals(
            [direct_goal], sampler=_make_sampler(), video_duration=300.0,
            model_path="/fake/yolo.pt",
        )
        # No checked, no dropped — the verifier didn't even consider it
        assert stats["checked"] == 0
        assert len(out) == 1
