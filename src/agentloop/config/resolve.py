"""Config resolution: defaults < manifest < env < CLI (A3).

Merging happens on raw mappings *before* validation, so overrides are
validated exactly like manifest values (an env var addressing an unknown
field fails closed via extra="forbid"). Rules:

- mappings deep-merge, everything else — scalars and lists — replaces wholesale
- env vars: AGENTLOOP_ prefix, '__' as the nesting delimiter, lowercased
  (AGENTLOOP_MODEL__PROVIDER=openai → model.provider); values are parsed as
  YAML scalars, so ints/bools/lists/inline maps work. Env addressing is for
  mapping fields only — list elements cannot be targeted, and YAML 1.1
  bool spellings (on/yes) parse as booleans.
- CLI overrides: a mapping of dotted paths to values; string values are
  parsed as YAML scalars, non-strings pass through untouched.

Every explicitly-set leaf records its provenance; absent = "default".
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from agentloop.config.loader import parse_manifest, read_manifest_file
from agentloop.config.manifest import Manifest
from agentloop.types import ConfigError

Source = Literal["manifest", "env", "cli"]
Origin = Literal["manifest", "env", "cli", "default"]

ENV_PREFIX = "AGENTLOOP_"
ENV_NESTING = "__"


class ResolvedConfig(BaseModel):
    """The one object downstream code reads; nothing else touches env/argv."""

    model_config = ConfigDict(frozen=True)

    manifest: Manifest
    path: Path
    provenance: dict[str, Source]

    def origin(self, dotted_path: str) -> Origin:
        """Where a value came from; 'default' if never explicitly set."""
        return self.provenance.get(dotted_path, "default")


def load_config(
    manifest_path: Path | str,
    *,
    env: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ResolvedConfig:
    """Load, resolve and validate configuration. The only public entrypoint."""
    if env is None:
        env = os.environ
    merged: dict[str, Any] = {}
    provenance: dict[str, Source] = {}
    _overlay(merged, read_manifest_file(manifest_path), "manifest", provenance)
    _overlay(merged, env_overrides(env), "env", provenance)
    _overlay(merged, _dotted_tree(cli_overrides or {}), "cli", provenance)
    manifest = parse_manifest(merged)
    return ResolvedConfig(
        manifest=manifest, path=Path(manifest_path).resolve(), provenance=provenance
    )


def env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Extract AGENTLOOP_* variables into a nested override mapping."""
    tree: dict[str, Any] = {}
    for key in sorted(env):
        if not key.startswith(ENV_PREFIX) or key == ENV_PREFIX.rstrip("_"):
            continue
        dotted = key.removeprefix(ENV_PREFIX).lower().replace(ENV_NESTING, ".")
        _set_dotted(tree, dotted, _parse_scalar(env[key]), origin=key)
    return tree


def _dotted_tree(overrides: Mapping[str, Any]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for dotted, value in overrides.items():
        if isinstance(value, str):
            value = _parse_scalar(value)
        _set_dotted(tree, dotted, value, origin=f"CLI override {dotted!r}")
    return tree


def _parse_scalar(raw: str) -> Any:
    if raw == "":
        return ""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw  # not YAML — treat as a literal string


def _set_dotted(tree: dict[str, Any], dotted: str, value: Any, *, origin: str) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"conflicting override paths at {part!r} ({origin})")
    node[parts[-1]] = value


def _overlay(
    base: dict[str, Any],
    overlay: Mapping[str, Any],
    source: Source,
    provenance: dict[str, Source],
    prefix: str = "",
) -> None:
    for key, value in overlay.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _overlay(base[key], value, source, provenance, prefix=f"{path}.")
        else:
            base[key] = copy.deepcopy(value) if isinstance(value, (Mapping, list)) else value
            _record_leaves(value, source, provenance, path)


def _record_leaves(
    value: Any, source: Source, provenance: dict[str, Source], path: str
) -> None:
    if isinstance(value, Mapping) and value:
        for key, child in value.items():
            _record_leaves(child, source, provenance, f"{path}.{key}")
    else:
        provenance[path] = source
