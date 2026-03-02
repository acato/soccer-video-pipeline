"""
Unit tests for the extracted spatial filter.

Tests spatial false-positive filter and sim gate from
src/segmentation/spatial_filter.py.
"""
from __future__ import annotations

import pytest

from src.detection.models import BoundingBox, Event, EventType
from src.segmentation.spatial_filter import (
    filter_wrong_side_events,
    passes_sim_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_type: EventType,
    start: float = 10.0,
    end: float = 11.0,
    confidence: float = 0.85,
    is_gk: bool = True,
    bbox_center_x: float | None = None,
    sim_team_gk: float = 0.90,
) -> Event:
    bbox = None
    if bbox_center_x is not None:
        bbox = BoundingBox(x=bbox_center_x - 0.025, y=0.4, width=0.05, height=0.1)
    return Event(
        event_id=f"evt-{event_type.value}-{start}",
        job_id="test-job",
        source_file="match.mp4",
        event_type=event_type,
        timestamp_start=start,
        timestamp_end=end,
        confidence=confidence,
        reel_targets=[],
        is_goalkeeper_event=is_gk,
        frame_start=int(start * 30),
        frame_end=int(end * 30),
        bounding_box=bbox,
        metadata={"sim_team_gk": sim_team_gk} if is_gk else {},
    )


def _make_keeper_events_on_side(
    side: str,
    count: int,
    start_offset: float = 0.0,
) -> list[Event]:
    cx = 0.2 if side == "left" else 0.8
    return [
        _make_event(
            EventType.CATCH,
            start=start_offset + i * 10.0,
            end=start_offset + i * 10.0 + 1.0,
            bbox_center_x=cx,
        )
        for i in range(count)
    ]


# ===========================================================================
# Sim gate
# ===========================================================================

@pytest.mark.unit
class TestSimGate:
    def test_high_sim_passes(self):
        e = _make_event(EventType.CATCH, sim_team_gk=0.80)
        assert passes_sim_gate(e) is True

    def test_low_sim_fails(self):
        e = _make_event(EventType.GOAL_KICK, sim_team_gk=0.60)
        assert passes_sim_gate(e) is False

    def test_borderline_sim_passes(self):
        e = _make_event(EventType.DISTRIBUTION_SHORT, sim_team_gk=0.75)
        assert passes_sim_gate(e) is True

    def test_penalty_exempt(self):
        e = _make_event(EventType.PENALTY, sim_team_gk=0.0)
        assert passes_sim_gate(e) is True

    def test_corner_kick_exempt(self):
        e = _make_event(EventType.CORNER_KICK, sim_team_gk=0.0)
        assert passes_sim_gate(e) is True

    def test_save_lower_threshold(self):
        """Save events use a lower sim threshold (0.60)."""
        e = _make_event(EventType.CATCH, sim_team_gk=0.65)
        assert passes_sim_gate(e) is True

    def test_save_below_lower_threshold_fails(self):
        e = _make_event(EventType.CATCH, sim_team_gk=0.55)
        assert passes_sim_gate(e) is False


# ===========================================================================
# Spatial filter
# ===========================================================================

@pytest.mark.unit
class TestSpatialFilter:
    def test_correct_side_events_kept(self):
        all_events = _make_keeper_events_on_side("left", 5, start_offset=10.0)
        selected = all_events[:3]
        result = filter_wrong_side_events(selected, all_events, 1000.0)
        assert len(result) == 3

    def test_wrong_side_events_removed(self):
        left_events = _make_keeper_events_on_side("left", 5, start_offset=10.0)
        wrong_event = _make_event(
            EventType.CATCH, start=100.0, end=101.0, bbox_center_x=0.8,
        )
        all_events = left_events + [wrong_event]
        selected = [wrong_event]
        result = filter_wrong_side_events(selected, all_events, 1000.0)
        assert result == []

    def test_halftime_sides_independent(self):
        video_dur = 1000.0
        first_half = _make_keeper_events_on_side("left", 5, start_offset=10.0)
        second_half = _make_keeper_events_on_side("right", 5, start_offset=600.0)
        all_events = first_half + second_half

        wrong_first = _make_event(EventType.CATCH, start=100.0, bbox_center_x=0.8)
        wrong_second = _make_event(EventType.CATCH, start=700.0, bbox_center_x=0.2)
        all_events_with_wrong = all_events + [wrong_first, wrong_second]

        result = filter_wrong_side_events(
            [wrong_first, wrong_second], all_events_with_wrong, video_dur,
        )
        assert result == []

    def test_midfield_events_rejected(self):
        voters = _make_keeper_events_on_side("left", 5, start_offset=10.0)
        midfield_event = _make_event(EventType.CATCH, start=100.0, bbox_center_x=0.50)
        all_events = voters + [midfield_event]
        result = filter_wrong_side_events([midfield_event], all_events, 1000.0)
        assert result == []

    def test_no_bbox_events_always_kept(self):
        voters = _make_keeper_events_on_side("left", 5, start_offset=10.0)
        no_bbox = _make_event(EventType.CATCH, start=100.0)
        assert no_bbox.bounding_box is None
        all_events = voters + [no_bbox]
        result = filter_wrong_side_events([no_bbox], all_events, 1000.0)
        assert len(result) == 1

    def test_goal_kick_exempt_from_side_filter(self):
        """Goal kicks are exempt from side filtering."""
        voters = _make_keeper_events_on_side("left", 5, start_offset=10.0)
        gk_wrong_side = _make_event(EventType.GOAL_KICK, start=100.0, bbox_center_x=0.8)
        all_events = voters + [gk_wrong_side]
        result = filter_wrong_side_events([gk_wrong_side], all_events, 1000.0)
        assert len(result) == 1

    def test_corner_kick_exempt_from_side_filter(self):
        voters = _make_keeper_events_on_side("left", 5, start_offset=10.0)
        ck = _make_event(EventType.CORNER_KICK, start=100.0, bbox_center_x=0.8)
        all_events = voters + [ck]
        result = filter_wrong_side_events([ck], all_events, 1000.0)
        assert len(result) == 1

    def test_ambiguous_split_keeps_all(self):
        """50/50 split within one half — no majority, keep everything."""
        left = _make_keeper_events_on_side("left", 1, start_offset=10.0)
        result = filter_wrong_side_events(left, left, 1000.0)
        assert len(result) == 1

    def test_single_event_no_filtering(self):
        left = _make_keeper_events_on_side("left", 1, start_offset=10.0)
        result = filter_wrong_side_events(left, left, 1000.0)
        assert len(result) == 1

    def test_halftime_switch_inferred(self):
        """When both halves vote same side, flip the weaker half.

        Teams switch sides at halftime.  If first-half voters say 'right'
        and second-half voters also say 'right', the half with fewer
        voters gets flipped to 'left'.
        """
        video_dur = 1000.0
        # First half: 5 events on right (strong consensus)
        first_half = _make_keeper_events_on_side("right", 5, start_offset=10.0)
        # Second half: 3 events on right (weaker consensus — FPs)
        # + 1 real event on left (the actual GK side in 2nd half)
        second_half_fps = _make_keeper_events_on_side("right", 3, start_offset=600.0)
        real_left = _make_event(EventType.CATCH, start=700.0, bbox_center_x=0.15)
        all_events = first_half + second_half_fps + [real_left]

        # The left event in the 2nd half should be KEPT (GK switches to left)
        result = filter_wrong_side_events([real_left], all_events, video_dur)
        assert len(result) == 1

        # The right-side FPs in the 2nd half should be REMOVED
        result = filter_wrong_side_events(second_half_fps, all_events, video_dur)
        assert len(result) == 0

    def test_halftime_switch_one_half_known(self):
        """When only one half has a majority, infer the opposite for the other."""
        video_dur = 1000.0
        # First half: 5 events on left (clear majority)
        first_half = _make_keeper_events_on_side("left", 5, start_offset=10.0)
        # Second half: 1 event on right (no majority by itself)
        second_right = _make_event(EventType.CATCH, start=700.0, bbox_center_x=0.8)
        all_events = first_half + [second_right]

        # The right event in 2nd half should be KEPT (inferred: GK on right)
        result = filter_wrong_side_events([second_right], all_events, video_dur)
        assert len(result) == 1

    def test_explicit_gk_side_left(self):
        """When gk_first_half_side='left', use it directly."""
        video_dur = 1000.0
        # Put all voters on right (would normally vote right as GK side)
        all_events = _make_keeper_events_on_side("right", 5, start_offset=10.0)
        # A left-side event in first half — would normally be filtered
        left_event = _make_event(EventType.CATCH, start=100.0, bbox_center_x=0.15)
        all_events.append(left_event)

        # With explicit side='left', the left event is KEPT
        result = filter_wrong_side_events(
            [left_event], all_events, video_dur, gk_first_half_side="left",
        )
        assert len(result) == 1

        # And right events are REMOVED
        result = filter_wrong_side_events(
            all_events[:5], all_events, video_dur, gk_first_half_side="left",
        )
        assert len(result) == 0

    def test_explicit_gk_side_second_half_inferred(self):
        """Explicit side for 1st half → opposite for 2nd half."""
        video_dur = 1000.0
        all_events = _make_keeper_events_on_side("left", 3, start_offset=10.0)
        # Second half event on right (should be kept — GK switches)
        right_2nd = _make_event(EventType.CATCH, start=700.0, bbox_center_x=0.8)
        # Second half event on left (should be filtered — opponent end)
        left_2nd = _make_event(EventType.CATCH, start=800.0, bbox_center_x=0.2)
        all_events.extend([right_2nd, left_2nd])

        result = filter_wrong_side_events(
            [right_2nd, left_2nd], all_events, video_dur,
            gk_first_half_side="left",
        )
        # right_2nd kept, left_2nd removed
        assert len(result) == 1
        assert result[0] is right_2nd
