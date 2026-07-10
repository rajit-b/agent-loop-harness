from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[2]
ANNOTATED_EXAMPLE = REPO_ROOT / "examples" / "manifests" / "annotated.manifest.yaml"

MINIMAL: dict[str, Any] = {
    "version": "1.0",
    "intent": "Answer questions about the codebase.",
    "model": {"provider": "ollama", "name": "qwen2.5:14b"},
}


@pytest.fixture
def minimal() -> dict[str, Any]:
    """A fresh copy of the smallest valid manifest mapping."""
    import copy

    return copy.deepcopy(MINIMAL)


@pytest.fixture
def write_manifest(tmp_path: Path):
    """Write a mapping (or raw string) as a manifest file, return its path."""

    def _write(data: dict[str, Any] | str, name: str = "agent.manifest.yaml") -> Path:
        path = tmp_path / name
        text = data if isinstance(data, str) else yaml.safe_dump(data)
        path.write_text(text, encoding="utf-8")
        return path

    return _write
