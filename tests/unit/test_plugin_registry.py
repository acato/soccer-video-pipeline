"""
Unit tests for PluginRegistry.
"""
from __future__ import annotations

import pytest

from src.reel_plugins.base import ClipParams, ReelPlugin, PipelineContext
from src.reel_plugins.registry import PluginRegistry, DEFAULT_PLUGIN_NAMES


# ---------------------------------------------------------------------------
# Test plugins
# ---------------------------------------------------------------------------

class _PluginA(ReelPlugin):
    @property
    def name(self) -> str:
        return "plugin_a"

    @property
    def reel_name(self) -> str:
        return "reel_x"

    def select_events(self, events, ctx):
        return events


class _PluginB(ReelPlugin):
    @property
    def name(self) -> str:
        return "plugin_b"

    @property
    def reel_name(self) -> str:
        return "reel_x"

    def select_events(self, events, ctx):
        return []


class _PluginC(ReelPlugin):
    @property
    def name(self) -> str:
        return "plugin_c"

    @property
    def reel_name(self) -> str:
        return "reel_y"

    def select_events(self, events, ctx):
        return events


# ===========================================================================
# Registration
# ===========================================================================

@pytest.mark.unit
class TestPluginRegistration:
    def test_register_and_len(self):
        r = PluginRegistry()
        r.register(_PluginA())
        assert len(r) == 1

    def test_duplicate_name_raises(self):
        r = PluginRegistry()
        r.register(_PluginA())
        with pytest.raises(ValueError, match="Duplicate plugin name"):
            r.register(_PluginA())

    def test_plugin_names(self):
        r = PluginRegistry()
        r.register(_PluginA())
        r.register(_PluginC())
        assert r.plugin_names == ["plugin_a", "plugin_c"]


# ===========================================================================
# Lookup
# ===========================================================================

@pytest.mark.unit
class TestPluginLookup:
    def test_get_plugins_for_reel(self):
        r = PluginRegistry()
        r.register(_PluginA())
        r.register(_PluginB())
        r.register(_PluginC())
        reel_x = r.get_plugins_for_reel("reel_x")
        assert len(reel_x) == 2
        assert {p.name for p in reel_x} == {"plugin_a", "plugin_b"}

    def test_get_plugins_for_nonexistent_reel(self):
        r = PluginRegistry()
        r.register(_PluginA())
        assert r.get_plugins_for_reel("does_not_exist") == []

    def test_get_all_reel_names_preserves_order(self):
        r = PluginRegistry()
        r.register(_PluginA())  # reel_x
        r.register(_PluginC())  # reel_y
        r.register(_PluginB())  # reel_x (dup)
        assert r.get_all_reel_names() == ["reel_x", "reel_y"]

    def test_empty_registry(self):
        r = PluginRegistry()
        assert len(r) == 0
        assert r.get_all_reel_names() == []
        assert r.plugin_names == []


# ===========================================================================
# Factory methods
# ===========================================================================

@pytest.mark.unit
class TestPluginRegistryFactory:
    def test_default_has_all_builtins(self):
        r = PluginRegistry.default()
        assert len(r) == len(DEFAULT_PLUGIN_NAMES)
        assert set(r.plugin_names) == set(DEFAULT_PLUGIN_NAMES)

    def test_default_has_keeper_and_highlights_reels(self):
        r = PluginRegistry.default()
        reels = r.get_all_reel_names()
        assert "keeper" in reels
        assert "highlights" in reels

    def test_from_config_subset(self):
        r = PluginRegistry.from_config(["keeper_saves"])
        assert len(r) == 1
        assert r.plugin_names == ["keeper_saves"]

    def test_from_config_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown plugin 'bogus'"):
            PluginRegistry.from_config(["bogus"])

    def test_from_config_empty_list(self):
        r = PluginRegistry.from_config([])
        assert len(r) == 0

    def test_from_config_multiple(self):
        r = PluginRegistry.from_config(["keeper_saves", "highlights_shots"])
        assert len(r) == 2
        assert r.get_all_reel_names() == ["keeper", "highlights"]
