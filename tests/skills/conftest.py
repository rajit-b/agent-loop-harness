from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from agentloop.skills.loader import Skill, load_skill
from agentloop.tools.builtin.echo import register_echo
from agentloop.tools.executor import ToolRegistry

from ..loop.conftest import RecordingEmitter


def write_skill(
    parent: Path,
    name: str,
    description: str,
    *,
    patterns: list[str] | None = None,
    required_tools: list[str] | None = None,
    budget: dict | None = None,
    body: str = "Follow these steps.",
) -> Path:
    directory = parent / name
    directory.mkdir(parents=True)
    config: dict = {"name": name, "description": description}
    if patterns:
        config["triggers"] = {"patterns": patterns}
    if required_tools:
        config["required_tools"] = required_tools
    if budget:
        config["budget"] = budget
    (directory / "skill.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (directory / "SKILL.md").write_text(body, encoding="utf-8")
    return directory


@pytest.fixture
def make_skill(tmp_path: Path):
    def _make(name: str, description: str, **kwargs) -> Skill:
        return load_skill(write_skill(tmp_path, name, description, **kwargs))

    return _make


class VocabEmbedder:
    """Deterministic bag-of-words embedder for semantic-selection tests."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            words = re.findall(r"[a-z]+", text.lower())
            vectors.append([float(words.count(term)) for term in self.vocab])
        return vectors


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_echo(reg)
    return reg


@pytest.fixture
def emitter() -> RecordingEmitter:
    return RecordingEmitter()
