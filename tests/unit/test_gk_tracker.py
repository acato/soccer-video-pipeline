"""Unit tests for src/tracking/gk_tracker.py"""
import pytest
from src.tracking.gk_tracker import MatchGoalkeeperTracker


@pytest.mark.unit
class TestMatchGoalkeeperTracker:
    def test_not_identified_initially(self):
        t = MatchGoalkeeperTracker("job-1")
        assert not t.identified
        assert t.identification_rate == 0.0

    def test_identified_after_registration(self):
        t = MatchGoalkeeperTracker("job-1")
        t.register_chunk_gk(0, gk_track_id=5)
        assert t.identified
        assert 5 in t._confirmed_gk_track_ids

    def test_none_registration_not_counted_as_identified(self):
        t = MatchGoalkeeperTracker("job-1")
        t.register_chunk_gk(0, gk_track_id=None)
        assert not t.identified

    def test_identification_rate_calculation(self):
        t = MatchGoalkeeperTracker("job-1")
        t.register_chunk_gk(0, gk_track_id=1)
        t.register_chunk_gk(1, gk_track_id=None)
        t.register_chunk_gk(2, gk_track_id=1)
        t.register_chunk_gk(3, gk_track_id=None)
        assert abs(t.identification_rate - 0.5) < 0.01

    def test_summary_returns_dict(self):
        t = MatchGoalkeeperTracker("job-1")
        t.register_chunk_gk(0, gk_track_id=7)
        t.register_chunk_gk(1, gk_track_id=None)
        s = t.summary()
        assert s["job_id"] == "job-1"
        assert s["total_chunks"] == 2
        assert s["identified_chunks"] == 1
        assert 7 in s["confirmed_track_ids"]

    def test_multiple_track_ids_accumulated(self):
        """GK may be reidentified with different track IDs across chunks."""
        t = MatchGoalkeeperTracker("job-1")
        t.register_chunk_gk(0, 3)
        t.register_chunk_gk(1, 7)  # Re-identified with new ID after occlusion
        assert 3 in t._confirmed_gk_track_ids
        assert 7 in t._confirmed_gk_track_ids

    def test_jersey_color_stored(self):
        t = MatchGoalkeeperTracker("job-1")
        t.set_jersey_color((120.0, 0.8, 0.9))
        assert t._gk_jersey_hsv == (120.0, 0.8, 0.9)
