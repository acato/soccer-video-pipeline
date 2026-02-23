"""Unit tests for src/tracking/gk_tracker.py"""
import pytest
from src.tracking.gk_tracker import MatchDualGoalkeeperTracker, MatchGoalkeeperTracker


@pytest.mark.unit
class TestMatchDualGoalkeeperTracker:
    def test_no_keepers_initially(self):
        t = MatchDualGoalkeeperTracker("job-1")
        assert not t._keepers["keeper_a"].identified
        assert not t._keepers["keeper_b"].identified

    def test_register_keeper_a(self):
        t = MatchDualGoalkeeperTracker("job-1")
        t.register_chunk_gks(0, {"keeper_a": 5, "keeper_b": None})
        assert t._keepers["keeper_a"].identified
        assert not t._keepers["keeper_b"].identified
        assert 5 in t._keepers["keeper_a"]._confirmed_track_ids

    def test_register_both_keepers(self):
        t = MatchDualGoalkeeperTracker("job-1")
        t.register_chunk_gks(0, {"keeper_a": 5, "keeper_b": 12})
        assert t._keepers["keeper_a"].identified
        assert t._keepers["keeper_b"].identified
        assert 5 in t._keepers["keeper_a"]._confirmed_track_ids
        assert 12 in t._keepers["keeper_b"]._confirmed_track_ids

    def test_none_registration_not_counted(self):
        t = MatchDualGoalkeeperTracker("job-1")
        t.register_chunk_gks(0, {"keeper_a": None, "keeper_b": None})
        assert not t._keepers["keeper_a"].identified
        assert not t._keepers["keeper_b"].identified

    def test_identification_rate_per_keeper(self):
        t = MatchDualGoalkeeperTracker("job-1")
        t.register_chunk_gks(0, {"keeper_a": 1, "keeper_b": 10})
        t.register_chunk_gks(1, {"keeper_a": None, "keeper_b": 10})
        t.register_chunk_gks(2, {"keeper_a": 1, "keeper_b": None})
        t.register_chunk_gks(3, {"keeper_a": None, "keeper_b": None})
        assert abs(t._keepers["keeper_a"].identification_rate - 0.5) < 0.01
        assert abs(t._keepers["keeper_b"].identification_rate - 0.5) < 0.01

    def test_summary_returns_both_keepers(self):
        t = MatchDualGoalkeeperTracker("job-1")
        t.register_chunk_gks(0, {"keeper_a": 7, "keeper_b": None})
        t.register_chunk_gks(1, {"keeper_a": None, "keeper_b": 15})
        s = t.summary()
        assert s["job_id"] == "job-1"
        assert s["total_chunks"] == 2
        assert "keepers" in s
        assert 7 in s["keepers"]["keeper_a"]["confirmed_track_ids"]
        assert 15 in s["keepers"]["keeper_b"]["confirmed_track_ids"]

    def test_multiple_track_ids_accumulated(self):
        t = MatchDualGoalkeeperTracker("job-1")
        t.register_chunk_gks(0, {"keeper_a": 3, "keeper_b": None})
        t.register_chunk_gks(1, {"keeper_a": 7, "keeper_b": None})
        assert 3 in t._keepers["keeper_a"]._confirmed_track_ids
        assert 7 in t._keepers["keeper_a"]._confirmed_track_ids


@pytest.mark.unit
class TestMatchGoalkeeperTrackerBackwardCompat:
    """Test the backward-compatible wrapper."""

    def test_not_identified_initially(self):
        t = MatchGoalkeeperTracker("job-1")
        assert not t.identified
        assert t.identification_rate == 0.0

    def test_identified_after_registration(self):
        t = MatchGoalkeeperTracker("job-1")
        t.register_chunk_gk(0, gk_track_id=5)
        assert t.identified

    def test_none_registration_not_counted(self):
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

    def test_set_jersey_color_noop(self):
        t = MatchGoalkeeperTracker("job-1")
        t.set_jersey_color((120.0, 0.8, 0.9))
        # Should not raise
