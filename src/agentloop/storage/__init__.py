"""Shared SQLite plumbing (WAL, capability checks)."""

from agentloop.storage.sqlite import connect

__all__ = ["connect"]
