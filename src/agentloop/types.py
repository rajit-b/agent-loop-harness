"""Core shared types (§4 of docs/ARCHITECTURE.md).

Imported by every layer; imports nothing from the package. Grows
incrementally per phase — Phase 1 only needs the error taxonomy root
and the config error.
"""

from __future__ import annotations


class AgentLoopError(Exception):
    """Base for all framework errors."""


class ConfigError(AgentLoopError):
    """Fatal, pre-run: the manifest or its overrides are invalid."""
