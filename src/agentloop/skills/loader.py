"""Skill loading with a lazily-read body.

skill.yaml is parsed and SKILL.md's *existence* verified at load; the
body's content is read from disk only on first access — nothing but the
description can leak into context before selection.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from agentloop.skills.model import SkillConfig
from agentloop.types import ConfigError

CONFIG_FILE = "skill.yaml"
BODY_FILE = "SKILL.md"


class Skill:
    def __init__(self, config: SkillConfig, directory: Path):
        self.config = config
        self.directory = directory
        self._body: str | None = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def description(self) -> str:
        return self.config.description

    def body(self) -> str:
        """Read SKILL.md on first access; cached afterwards."""
        if self._body is None:
            self._body = (self.directory / BODY_FILE).read_text(encoding="utf-8")
        return self._body

    def __repr__(self) -> str:
        return f"Skill({self.name!r})"


def load_skill(directory: Path | str) -> Skill:
    directory = Path(directory)
    config_path = directory / CONFIG_FILE
    if not config_path.is_file():
        raise ConfigError(f"skill directory {directory} has no {CONFIG_FILE}")
    if not (directory / BODY_FILE).is_file():
        raise ConfigError(f"skill directory {directory} has no {BODY_FILE}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a mapping")
    try:
        config = SkillConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid skill config {config_path}: {exc}") from exc
    return Skill(config, directory)
