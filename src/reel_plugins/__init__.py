"""
Reel plugin system.

Each plugin is a self-contained unit that selects events for a named reel
and specifies how clips should be cut.  Plugins are pure event filters —
they consume detection output, they do not run detection.

Quick start:
    from src.reel_plugins.registry import PluginRegistry
    registry = PluginRegistry.default()
"""
from src.reel_plugins.base import ClipParams, PipelineContext, ReelPlugin

__all__ = ["ClipParams", "PipelineContext", "ReelPlugin"]
