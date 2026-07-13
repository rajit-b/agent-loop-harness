"""The use_skill builtin meta-tool (§10): the model's own progressive-
disclosure lever. The body arrives as the tool result — mid-turn, the
already-sent system prompt is never rewritten."""

from __future__ import annotations

from typing import Any

from agentloop.skills.manager import SkillManager
from agentloop.tools.executor import ToolRegistry
from agentloop.types import ToolSpec

USE_SKILL_SPEC = ToolSpec(
    name="use_skill",
    description=(
        "Load the full instructions of an available skill by name. "
        "Use this before attempting a task a skill covers."
    ),
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Skill name."}},
        "required": ["name"],
    },
    source="builtin",
)


def make_use_skill_handler(manager: SkillManager):
    async def use_skill(arguments: dict[str, Any]) -> str:
        name = str(arguments.get("name", ""))
        skill = manager.get(name)
        if skill is None:
            known = ", ".join(manager.names()) or "none"
            raise ValueError(f"unknown skill {name!r}; available: {known}")
        return f"# Skill: {skill.name}\n\n{skill.body()}"

    return use_skill


def register_use_skill(registry: ToolRegistry, manager: SkillManager) -> None:
    registry.register(USE_SKILL_SPEC, make_use_skill_handler(manager))
