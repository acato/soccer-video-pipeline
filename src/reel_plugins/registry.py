"""
Plugin registry: discovers, stores, and looks up reel plugins.
"""
from __future__ import annotations

from collections import defaultdict

import structlog

from src.reel_plugins.base import ReelPlugin

log = structlog.get_logger(__name__)


# Built-in plugin classes, populated by _load_builtins() on first access.
_BUILTIN_PLUGINS: dict[str, type[ReelPlugin]] | None = None


def _load_builtins() -> dict[str, type[ReelPlugin]]:
    """Import built-in plugin classes lazily to avoid circular imports."""
    from src.reel_plugins.highlights import HighlightsShotsPlugin
    from src.reel_plugins.keeper import (
        KeeperCornerKickPlugin,
        KeeperDistributionPlugin,
        KeeperGoalKickPlugin,
        KeeperOneOnOnePlugin,
        KeeperSavesPlugin,
    )

    return {
        "keeper_saves": KeeperSavesPlugin,
        "keeper_goal_kick": KeeperGoalKickPlugin,
        "keeper_distribution": KeeperDistributionPlugin,
        "keeper_one_on_one": KeeperOneOnOnePlugin,
        "keeper_corner_kick": KeeperCornerKickPlugin,
        "highlights_shots": HighlightsShotsPlugin,
    }


def _get_builtins() -> dict[str, type[ReelPlugin]]:
    global _BUILTIN_PLUGINS
    if _BUILTIN_PLUGINS is None:
        _BUILTIN_PLUGINS = _load_builtins()
    return _BUILTIN_PLUGINS


# Default plugin set — matches pre-refactor behavior.
DEFAULT_PLUGIN_NAMES: list[str] = [
    "keeper_saves",
    "keeper_goal_kick",
    "keeper_distribution",
    "keeper_one_on_one",
    "keeper_corner_kick",
    "highlights_shots",
]


class PluginRegistry:
    """Manages a set of reel plugins for a pipeline run."""

    def __init__(self) -> None:
        self._plugins: dict[str, ReelPlugin] = {}

    def register(self, plugin: ReelPlugin) -> None:
        if plugin.name in self._plugins:
            raise ValueError(
                f"Duplicate plugin name '{plugin.name}'. "
                f"Already registered: {self._plugins[plugin.name]!r}"
            )
        self._plugins[plugin.name] = plugin
        log.debug("registry.registered", plugin=plugin.name, reel=plugin.reel_name)

    def get_plugins_for_reel(self, reel_name: str) -> list[ReelPlugin]:
        """Return all plugins that target *reel_name*, in registration order."""
        return [p for p in self._plugins.values() if p.reel_name == reel_name]

    def get_all_reel_names(self) -> list[str]:
        """Unique reel names across all registered plugins, in registration order."""
        seen: set[str] = set()
        result: list[str] = []
        for p in self._plugins.values():
            if p.reel_name not in seen:
                seen.add(p.reel_name)
                result.append(p.reel_name)
        return result

    @property
    def plugin_names(self) -> list[str]:
        return list(self._plugins.keys())

    def __len__(self) -> int:
        return len(self._plugins)

    # ----- Factory methods ------------------------------------------------

    @classmethod
    def from_config(cls, plugin_names: list[str]) -> PluginRegistry:
        """Build a registry from a list of plugin names.

        Names are looked up in the built-in plugin table.  Unknown names
        raise ``KeyError`` so configuration errors are caught early.
        """
        builtins = _get_builtins()
        registry = cls()
        for name in plugin_names:
            if name not in builtins:
                raise KeyError(
                    f"Unknown plugin '{name}'. "
                    f"Available: {sorted(builtins.keys())}"
                )
            registry.register(builtins[name]())
        return registry

    @classmethod
    def default(cls) -> PluginRegistry:
        """Registry with all built-in plugins — matches pre-refactor behavior."""
        return cls.from_config(DEFAULT_PLUGIN_NAMES)
