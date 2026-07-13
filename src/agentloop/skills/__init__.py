"""Skills: progressive disclosure and selection (Phase 6)."""

from agentloop.skills.loader import Skill, load_skill
from agentloop.skills.manager import SkillManager
from agentloop.skills.model import SkillConfig
from agentloop.skills.selector import (
    SelectedSkill,
    SelectionOutcome,
    SkillSelector,
    SkippedSkill,
)
from agentloop.skills.tooling import USE_SKILL_SPEC, register_use_skill

__all__ = [
    "USE_SKILL_SPEC",
    "SelectedSkill",
    "SelectionOutcome",
    "Skill",
    "SkillConfig",
    "SkillManager",
    "SkillSelector",
    "SkippedSkill",
    "load_skill",
    "register_use_skill",
]
