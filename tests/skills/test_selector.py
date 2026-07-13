"""Selection algorithm: explicit → trigger → semantic, satisfiability."""

from __future__ import annotations

import pytest

from agentloop.skills.selector import SkillSelector
from agentloop.types import SkillError, ToolSpec

from .conftest import VocabEmbedder

FS_TOOL = ToolSpec(
    name="fs__read_file", source="mcp_server", permissions_tag="fs.read_file"
)
ECHO_TOOL = ToolSpec(name="echo")
AVAILABLE = (FS_TOOL, ECHO_TOOL)


class TestExplicit:
    async def test_at_mention_selects(self, make_skill):
        selector = SkillSelector([make_skill("code-search", "Search code.")])
        outcome = await selector.select("please @code-search for the loader", AVAILABLE)
        assert [s.skill.name for s in outcome.selected] == ["code-search"]
        assert outcome.selected[0].via == "explicit"

    async def test_unknown_mention_is_ignored(self, make_skill):
        selector = SkillSelector([make_skill("code-search", "Search code.")])
        outcome = await selector.select("email me @example.com", AVAILABLE)
        assert outcome.selected == ()

    async def test_unsatisfiable_explicit_fails_loudly_naming_tools(self, make_skill):
        skill = make_skill(
            "code-search", "Search code.",
            required_tools=["ripgrep.*", "fs.read_*"],
        )
        selector = SkillSelector([skill])
        with pytest.raises(SkillError) as exc_info:
            await selector.select("@code-search for the loader", AVAILABLE)
        message = str(exc_info.value)
        assert "code-search" in message
        assert "ripgrep.*" in message  # the missing glob, by name
        assert "fs.read_*" not in message  # the satisfied one is not blamed

    async def test_satisfiable_required_tools_pass(self, make_skill):
        skill = make_skill("code-search", "Search code.", required_tools=["fs.read_*"])
        selector = SkillSelector([skill])
        outcome = await selector.select("@code-search please", AVAILABLE)
        assert len(outcome.selected) == 1


class TestTrigger:
    async def test_regex_pattern_matches_case_insensitively(self, make_skill):
        skill = make_skill(
            "code-search", "Search code.", patterns=[r"where\s+is", "find"]
        )
        selector = SkillSelector([skill])
        outcome = await selector.select("WHERE IS the config loader?", AVAILABLE)
        assert outcome.selected[0].via == "trigger"

    async def test_no_match_no_selection(self, make_skill):
        skill = make_skill("code-search", "Search code.", patterns=["find"])
        selector = SkillSelector([skill])
        outcome = await selector.select("summarize the readme", AVAILABLE)
        assert outcome.selected == ()

    async def test_unsatisfiable_trigger_skill_is_skipped_not_fatal(self, make_skill):
        skill = make_skill(
            "code-search", "Search code.",
            patterns=["find"], required_tools=["ripgrep.*"],
        )
        selector = SkillSelector([skill])
        outcome = await selector.select("find the loader", AVAILABLE)
        assert outcome.selected == ()
        assert outcome.skipped[0].name == "code-search"
        assert "ripgrep.*" in outcome.skipped[0].reason


class TestSemantic:
    VOCAB = ["search", "codebase", "code", "function", "jira", "ticket", "issue"]

    def make_pair(self, make_skill):
        return [
            make_skill("code-search", "Search the codebase for code and functions."),
            make_skill("jira-lookup", "Look up jira tickets and issue details."),
        ]

    async def test_best_match_above_threshold_selected(self, make_skill):
        selector = SkillSelector(
            self.make_pair(make_skill),
            embedder=VocabEmbedder(self.VOCAB),
            threshold=0.3,
        )
        outcome = await selector.select(
            "search the codebase for the parse function", AVAILABLE
        )
        assert [s.skill.name for s in outcome.selected] == ["code-search"]
        assert outcome.selected[0].via == "semantic"

    async def test_below_threshold_selects_nothing(self, make_skill):
        selector = SkillSelector(
            self.make_pair(make_skill),
            embedder=VocabEmbedder(self.VOCAB),
            threshold=0.99,
        )
        outcome = await selector.select("search the codebase somewhat", AVAILABLE)
        assert outcome.selected == ()

    async def test_top_k_limits_semantic_selections(self, make_skill):
        selector = SkillSelector(
            self.make_pair(make_skill),
            embedder=VocabEmbedder(self.VOCAB),
            threshold=0.01,
            top_k=1,
        )
        outcome = await selector.select(
            "search the codebase for the jira ticket function", AVAILABLE
        )
        assert len(outcome.selected) == 1  # both would qualify; top_k=1 wins

    async def test_no_embedder_disables_semantic(self, make_skill):
        selector = SkillSelector(self.make_pair(make_skill), embedder=None)
        outcome = await selector.select("search the codebase", AVAILABLE)
        assert outcome.selected == ()

    async def test_descriptions_embedded_once_across_selects(self, make_skill):
        embedder = VocabEmbedder(self.VOCAB)
        selector = SkillSelector(
            self.make_pair(make_skill), embedder=embedder, threshold=0.3
        )
        await selector.select("search the codebase", AVAILABLE)
        await selector.select("search the code again", AVAILABLE)
        # 1 call for descriptions + 1 per query
        assert embedder.calls == 3


class TestUnion:
    async def test_explicit_and_trigger_dedupe(self, make_skill):
        skill = make_skill("code-search", "Search code.", patterns=["search"])
        selector = SkillSelector([skill])
        outcome = await selector.select("@code-search search now", AVAILABLE)
        assert len(outcome.selected) == 1
        assert outcome.selected[0].via == "explicit"  # explicit wins the label
