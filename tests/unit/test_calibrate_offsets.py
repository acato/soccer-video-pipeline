"""Unit tests for scripts/calibrate_offsets.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# calibrate_offsets lives in scripts/, not on the default import path.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import calibrate_offsets as calib  # noqa: E402


def _write_gt(tmp_path: Path, half_idx: int, events: list[dict]) -> str:
    """Write a minimal GT JSON in the expected structure."""
    data = []
    for ev in events:
        data.append({
            "period_type": "Half",
            "period_order": half_idx,
            "event_time": ev["event_time"],
            "events": [{
                "event_name": ev["event_name"],
                "property": {"Type": ev.get("type")} if ev.get("type") else {},
            }],
        })
    p = tmp_path / f"gt_half{half_idx + 1}.json"
    p.write_text(json.dumps({"data": data}))
    return str(p)


def _write_detections(tmp_path: Path, events: list[dict]) -> str:
    path = tmp_path / "detected.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events))
    return str(path)


@pytest.mark.unit
class TestCalibrate:
    def test_recovers_planted_offsets_with_goals(self, tmp_path):
        # Plant: VIDEO_OFFSET=20, half2_start=3500 (halftime 780s > 300 min).
        gt1 = _write_gt(tmp_path, 0, [
            {"event_time": 500_000, "event_name": "Goals Conceded"},
            {"event_time": 2_415_000, "event_name": "Goals Conceded"},
        ])
        gt2 = _write_gt(tmp_path, 1, [
            {"event_time": 3_000_000, "event_name": "Goals Conceded"},
        ])
        det = _write_detections(tmp_path, [
            {"event_type": "goal", "start_sec": 520.0},
            {"event_type": "goal", "start_sec": 2435.0},
            {"event_type": "goal", "start_sec": 3800.0},
        ])

        result = calib.calibrate(
            events_path=det, gt_paths=[gt1, gt2],
            half2_game_offset=2700.0, tolerance=45.0,
        )

        assert "error" not in result
        assert "goal" in result["signals_used"]
        assert result["matches"]["goal"] == 3
        assert abs(result["video_offset"] - 20.0) <= 5
        assert abs(result["half2_video_start"] - 3500.0) <= 10

    def test_uses_shots_alone_when_no_goals(self, tmp_path):
        # No goals — shots only
        gt1_events = [
            {"event_time": int((i * 300 + 60) * 1000),
             "event_name": "Shots & Goals"}
            for i in range(5)
        ]
        gt2_events = [
            {"event_time": int((3000 + i * 300) * 1000),
             "event_name": "Shots & Goals"}
            for i in range(5)
        ]
        gt1 = _write_gt(tmp_path, 0, gt1_events)
        gt2 = _write_gt(tmp_path, 1, gt2_events)
        # Plant: VIDEO_OFFSET=50, half2_start=3550 (halftime 800s)
        det_events = [
            {"event_type": "shot_on_target",
             "start_sec": (i * 300 + 60) + 50}
            for i in range(5)
        ] + [
            {"event_type": "shot_on_target",
             "start_sec": (3000 + i * 300 - 2700) + 3550}
            for i in range(5)
        ]
        det = _write_detections(tmp_path, det_events)

        result = calib.calibrate(
            events_path=det, gt_paths=[gt1, gt2],
            half2_game_offset=2700.0, tolerance=45.0,
        )

        assert "error" not in result
        assert result["signals_used"] == ["shot_on_target"]
        assert result["matches"]["shot_on_target"] >= 8

    def test_combines_goals_and_shots_for_half2_localization(self, tmp_path):
        # Goals only in H1 — shots should drive H2 offset finding.
        # Plant: VIDEO_OFFSET=20, half2_start=3600 (halftime 880s).
        gt1 = _write_gt(tmp_path, 0, [
            {"event_time": 500_000, "event_name": "Goals Conceded"},
            {"event_time": 2_415_000, "event_name": "Goals Conceded"},
            {"event_time": 700_000, "event_name": "Shots & Goals"},
            {"event_time": 1_500_000, "event_name": "Shots & Goals"},
        ])
        gt2 = _write_gt(tmp_path, 1, [
            {"event_time": 3_000_000, "event_name": "Shots & Goals"},
            {"event_time": 3_300_000, "event_name": "Shots & Goals"},
            {"event_time": 3_900_000, "event_name": "Shots & Goals"},
        ])
        det = _write_detections(tmp_path, [
            {"event_type": "goal", "start_sec": 520.0},
            {"event_type": "goal", "start_sec": 2435.0},
            # H1 shots
            {"event_type": "shot_on_target", "start_sec": 720.0},
            {"event_type": "shot_on_target", "start_sec": 1520.0},
            # H2 shots at the planted half2_start=3600
            # game 3000 → elapsed 300 → video 3900
            # game 3300 → elapsed 600 → video 4200
            # game 3900 → elapsed 1200 → video 4800
            {"event_type": "shot_on_target", "start_sec": 3900.0},
            {"event_type": "shot_on_target", "start_sec": 4200.0},
            {"event_type": "shot_on_target", "start_sec": 4800.0},
        ])

        result = calib.calibrate(
            events_path=det, gt_paths=[gt1, gt2],
            half2_game_offset=2700.0, tolerance=45.0,
        )

        assert "error" not in result
        assert set(result["signals_used"]) == {"goal", "shot_on_target"}
        # H1 goals drive vo; H2 shots drive half2_start
        assert abs(result["video_offset"] - 20.0) <= 5
        assert abs(result["half2_video_start"] - 3600.0) <= 15

    def test_enforces_min_halftime(self, tmp_path):
        # If naive grid-search prefers halftime=0, constraint should push
        # half2_start to >= video_offset + half2_game_offset + 300.
        gt1 = _write_gt(tmp_path, 0, [
            {"event_time": 500_000, "event_name": "Goals Conceded"},
        ])
        gt2 = _write_gt(tmp_path, 1, [])
        det = _write_detections(tmp_path, [
            {"event_type": "goal", "start_sec": 520.0},
        ])
        result = calib.calibrate(
            events_path=det, gt_paths=[gt1, gt2],
            half2_game_offset=2700.0, tolerance=45.0,
        )
        # half2_video_start must be >= video_offset + 2700 + 300
        assert result["half2_video_start"] >= (
            result["video_offset"] + 2700 + 300 - 1
        )

    def test_insufficient_signal_returns_error(self, tmp_path):
        gt1 = _write_gt(tmp_path, 0, [])
        gt2 = _write_gt(tmp_path, 1, [])
        det = _write_detections(tmp_path, [
            {"event_type": "throw_in", "start_sec": 100.0},
        ])
        result = calib.calibrate(
            events_path=det, gt_paths=[gt1, gt2],
            half2_game_offset=2700.0, tolerance=45.0,
        )
        assert "error" in result
