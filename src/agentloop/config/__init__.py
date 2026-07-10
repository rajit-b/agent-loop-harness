"""Manifest schema, loader, and config resolution (Phase 1)."""

from agentloop.config.loader import parse_manifest, read_manifest_file
from agentloop.config.manifest import Manifest
from agentloop.config.resolve import ResolvedConfig, Source, load_config

__all__ = [
    "Manifest",
    "ResolvedConfig",
    "Source",
    "load_config",
    "parse_manifest",
    "read_manifest_file",
]
