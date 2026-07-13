"""skill.yaml schema (§10). A skill is a directory: this config plus a
SKILL.md body that stays on disk until the skill is selected."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SkillTriggers(_Base):
    patterns: tuple[str, ...] = Field(
        default=(),
        description="Case-insensitive regexes matched against the user turn.",
    )

    @field_validator("patterns")
    @classmethod
    def _compilable(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"invalid trigger pattern {pattern!r}: {exc}") from exc
        return patterns


class SkillBudget(_Base):
    """Overlays (never exceeds) the run limits while the skill is active."""

    max_tool_calls: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)


class SkillConfig(_Base):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str = Field(
        min_length=1,
        description="The ONLY part of the skill that enters context before "
        "selection (progressive disclosure).",
    )
    triggers: SkillTriggers = SkillTriggers()
    required_tools: tuple[str, ...] = Field(
        default=(),
        description="Allowlist-style globs of canonical 'server.tool' names; "
        "each must match at least one available tool.",
    )
    budget: SkillBudget = SkillBudget()
