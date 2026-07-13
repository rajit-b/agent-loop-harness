"""SkillManager: the loop's one handle on the skill layer."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

from agentloop.skills.loader import Skill, load_skill
from agentloop.skills.selector import SelectionOutcome, SkillSelector
from agentloop.types import ConfigError, EmbeddingProvider, ToolSpec


class SkillManager:
    def __init__(
        self,
        skills: Sequence[Skill],
        *,
        embedder: EmbeddingProvider | None = None,
        threshold: float = 0.55,
        top_k: int = 1,
    ):
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise ConfigError(f"duplicate skill name {skill.name!r}")
            self._skills[skill.name] = skill
        self._selector = SkillSelector(
            list(self._skills.values()),
            embedder=embedder,
            threshold=threshold,
            top_k=top_k,
        )

    @classmethod
    def from_paths(
        cls, paths: Sequence[str | Path], **kwargs
    ) -> SkillManager:
        return cls([load_skill(path) for path in paths], **kwargs)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._skills)

    def index_block(self) -> str:
        """The compact index that enters every system prompt: name and
        description only — bodies stay on disk until selection."""
        if not self._skills:
            return ""
        lines = ["## Available skills"]
        lines += [
            f"- {skill.name}: {skill.description}"
            for skill in self._skills.values()
        ]
        lines.append(
            "Call the `use_skill` tool with a skill's name to load its full "
            "instructions before relying on it."
        )
        return "\n".join(lines)

    async def select_for_turn(
        self, user_input: str, available: Sequence[ToolSpec]
    ) -> SelectionOutcome:
        return await self._selector.select(user_input, available)
