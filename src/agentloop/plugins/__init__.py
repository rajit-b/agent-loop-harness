"""Plugins: packaging contract, registrar, fail-closed loader (Phase 7)."""

from agentloop.plugins.contract import (
    ENTRY_POINT_GROUP,
    FACTORY_ATTR,
    PLUGIN_API_VERSION,
    CliCommand,
    Plugin,
)
from agentloop.plugins.loader import LoadReport, PluginManager
from agentloop.plugins.registrar import Registrar

__all__ = [
    "ENTRY_POINT_GROUP",
    "FACTORY_ATTR",
    "PLUGIN_API_VERSION",
    "CliCommand",
    "LoadReport",
    "Plugin",
    "PluginManager",
    "Registrar",
]
