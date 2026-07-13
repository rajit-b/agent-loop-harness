"""Skill selection (§10): explicit → trigger → semantic, unioned.

1. Explicit — '@skill-name' in the user turn naming a known skill.
   Unconditional, and unsatisfiable required_tools FAIL LOUDLY with the
   missing globs named (SkillError).
2. Trigger — case-insensitive regex patterns against the user turn.
3. Semantic — cosine(turn embedding, description embedding) ≥ threshold,
   best top_k. Skipped entirely when no embedder is configured.

Auto-selected (trigger/semantic) skills whose required_tools aren't
satisfiable are skipped, not fatal — the record surfaces as a
`skill.skipped` trace event.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from collections.abc import Sequence

from agentloop.skills.loader import Skill
from agentloop.tools.permissions import canonical_name
from agentloop.types import EmbeddingProvider, SkillError, ToolSpec

_MENTION = re.compile(r"@([a-z0-9][a-z0-9-]*)")


@dataclass(frozen=True, slots=True)
class SelectedSkill:
    skill: Skill
    via: str  # "explicit" | "trigger" | "semantic"


@dataclass(frozen=True, slots=True)
class SkippedSkill:
    name: str
    via: str
    reason: str


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    selected: tuple[SelectedSkill, ...]
    skipped: tuple[SkippedSkill, ...]


def missing_tools(skill: Skill, available: Sequence[ToolSpec]) -> list[str]:
    names = [canonical_name(spec) for spec in available]
    return [
        glob
        for glob in skill.config.required_tools
        if not any(fnmatchcase(name, glob) for name in names)
    ]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class SkillSelector:
    def __init__(
        self,
        skills: Sequence[Skill],
        *,
        embedder: EmbeddingProvider | None = None,
        threshold: float = 0.55,
        top_k: int = 1,
    ):
        self._skills = {skill.name: skill for skill in skills}
        self._embedder = embedder
        self._threshold = threshold
        self._top_k = top_k
        self._description_vectors: list[list[float]] | None = None

    async def select(
        self, user_input: str, available: Sequence[ToolSpec]
    ) -> SelectionOutcome:
        selected: dict[str, SelectedSkill] = {}
        skipped: list[SkippedSkill] = []

        # 1. explicit — unconditional; unsatisfiable is fatal and loud
        for mention in _MENTION.findall(user_input):
            skill = self._skills.get(mention)
            if skill is None:
                continue  # plain @mention, not a skill invocation
            missing = missing_tools(skill, available)
            if missing:
                raise SkillError(
                    f"skill {skill.name!r} was explicitly invoked but its "
                    f"required tools are unavailable: {', '.join(missing)}"
                )
            selected.setdefault(skill.name, SelectedSkill(skill, "explicit"))

        # 2. trigger patterns
        for skill in self._skills.values():
            if skill.name in selected:
                continue
            if any(
                re.search(pattern, user_input, re.IGNORECASE)
                for pattern in skill.config.triggers.patterns
            ):
                self._auto_select(skill, "trigger", available, selected, skipped)

        # 3. semantic — description-embedding similarity
        if self._embedder is not None and len(selected) < len(self._skills):
            for skill, score in await self._semantic_candidates(user_input):
                if skill.name in selected:
                    continue
                if score < self._threshold:
                    break  # sorted desc: nothing further qualifies
                self._auto_select(skill, "semantic", available, selected, skipped)

        return SelectionOutcome(tuple(selected.values()), tuple(skipped))

    def _auto_select(
        self,
        skill: Skill,
        via: str,
        available: Sequence[ToolSpec],
        selected: dict[str, SelectedSkill],
        skipped: list[SkippedSkill],
    ) -> None:
        missing = missing_tools(skill, available)
        if missing:
            skipped.append(
                SkippedSkill(
                    name=skill.name,
                    via=via,
                    reason=f"required tools unavailable: {', '.join(missing)}",
                )
            )
        else:
            selected[skill.name] = SelectedSkill(skill, via)

    async def _semantic_candidates(self, user_input: str) -> list[tuple[Skill, float]]:
        skills = list(self._skills.values())
        if self._description_vectors is None:  # embed descriptions once
            assert self._embedder is not None
            self._description_vectors = await self._embedder.embed(
                [skill.description for skill in skills]
            )
        assert self._embedder is not None
        [query_vector] = await self._embedder.embed([user_input])
        scored = sorted(
            zip(skills, (
                _cosine(query_vector, vector)
                for vector in self._description_vectors
            ), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return scored[: self._top_k]
