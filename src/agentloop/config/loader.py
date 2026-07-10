"""YAML → validated Manifest, with errors that name the offending key."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agentloop.config.manifest import Manifest
from agentloop.types import ConfigError


def read_manifest_file(path: Path | str) -> dict[str, Any]:
    """Read and parse the YAML file; no validation beyond 'root is a mapping'."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"manifest not found: {path}") from None
    except OSError as exc:
        raise ConfigError(f"cannot read manifest {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {path}: {exc}") from exc
    if data is None:  # empty file
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"manifest root must be a mapping, got {type(data).__name__}: {path}"
        )
    return dict(data)


def parse_manifest(data: Mapping[str, Any]) -> Manifest:
    """Validate a raw mapping into a Manifest, re-raising as ConfigError."""
    try:
        return Manifest.model_validate(dict(data))
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def _format_validation_error(exc: ValidationError) -> str:
    lines = [f"invalid manifest ({exc.error_count()} error(s)):"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)
