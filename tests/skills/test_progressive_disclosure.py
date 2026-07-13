"""Progressive disclosure through the loop — the Phase 6 gate evidence."""

from __future__ import annotations

from agentloop.config.manifest import LimitsConfig
from agentloop.loop.machine import AgentLoop
from agentloop.models.protocol import estimate_tokens
from agentloop.skills.manager import SkillManager
from agentloop.skills.tooling import register_use_skill
from agentloop.types import ToolCall

from ..loop.conftest import ScriptedProvider, echo_call, result

INTENT = "Answer questions about the codebase."
BIG_BODY = "STEP-BY-STEP INSTRUCTIONS. " + ("Lorem ipsum dolor sit amet. " * 1500)


def make_loop(provider, registry, emitter, manager, **kwargs) -> AgentLoop:
    return AgentLoop(
        provider, registry, intent=INTENT, emitter=emitter, skills=manager, **kwargs
    )


class TestDisclosure:
    async def test_only_descriptions_enter_context_pre_selection(
        self, make_skill, registry, emitter
    ):
        """The token-count proof: ~40k-char bodies stay out of context."""
        manager = SkillManager(
            [
                make_skill("code-search", "Search the codebase.", body=BIG_BODY),
                make_skill("jira-lookup", "Look up Jira tickets.", body=BIG_BODY),
            ]
        )
        provider = ScriptedProvider([result("done")])
        await make_loop(provider, registry, emitter, manager).run_turn(
            "summarize the readme"  # triggers nothing
        )
        [request] = provider.requests
        system = request.messages[0]
        # descriptions are present...
        assert "code-search: Search the codebase." in system.text
        assert "jira-lookup: Look up Jira tickets." in system.text
        # ...bodies are not, in ANY message
        assert all("STEP-BY-STEP" not in m.text for m in request.messages)
        # and the whole prompt is a fraction of ONE body's size
        body_tokens = len(BIG_BODY) // 4  # ~10.5k tokens per body
        assert estimate_tokens(list(request.messages)) < body_tokens // 10

    async def test_trigger_selected_body_enters_system_prompt(
        self, make_skill, registry, emitter
    ):
        manager = SkillManager(
            [make_skill("code-search", "Search.", patterns=["find"], body=BIG_BODY)]
        )
        provider = ScriptedProvider([result("done")])
        await make_loop(provider, registry, emitter, manager).run_turn(
            "find the loader"
        )
        system = provider.requests[0].messages[0]
        assert "Active skill: code-search" in system.text
        assert "STEP-BY-STEP" in system.text
        events = [p for k, p in emitter.events if k == "skill.selected"]
        assert len(events) == 1
        assert events[0]["name"] == "code-search"
        assert events[0]["via"] == "trigger"

    async def test_explicit_mention_selects(self, make_skill, registry, emitter):
        manager = SkillManager(
            [make_skill("code-search", "Search.", body="THE BODY")]
        )
        provider = ScriptedProvider([result("done")])
        await make_loop(provider, registry, emitter, manager).run_turn(
            "@code-search where is the loader?"
        )
        assert "THE BODY" in provider.requests[0].messages[0].text

    async def test_use_skill_tool_discloses_body_mid_turn(
        self, make_skill, registry, emitter
    ):
        manager = SkillManager(
            [make_skill("code-search", "Search.", body=BIG_BODY)]
        )
        register_use_skill(registry, manager)
        provider = ScriptedProvider(
            [
                result(
                    tool_calls=(
                        ToolCall(id="c1", name="use_skill",
                                 arguments={"name": "code-search"}),
                    )
                ),
                result("done"),
            ]
        )
        await make_loop(provider, registry, emitter, manager).run_turn(
            "summarize the readme"
        )
        # not in context before the model asked
        assert all(
            "STEP-BY-STEP" not in m.text for m in provider.requests[0].messages
        )
        # delivered as the tool result once requested
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert "STEP-BY-STEP" in tool_message.text
        assert "# Skill: code-search" in tool_message.text

    async def test_use_skill_with_unknown_name_is_error_result(
        self, make_skill, registry, emitter
    ):
        manager = SkillManager([make_skill("code-search", "Search.")])
        register_use_skill(registry, manager)
        provider = ScriptedProvider(
            [
                result(
                    tool_calls=(
                        ToolCall(id="c1", name="use_skill",
                                 arguments={"name": "nope"}),
                    )
                ),
                result("done"),
            ]
        )
        turn = await make_loop(provider, registry, emitter, manager).run_turn("q")
        assert turn.status == "completed"
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert tool_message.is_error is True
        assert "code-search" in tool_message.text  # names the available skills


class TestFailsLoudly:
    async def test_explicit_selection_with_missing_tools_errors_the_turn(
        self, make_skill, registry, emitter
    ):
        manager = SkillManager(
            [make_skill("code-search", "Search.", required_tools=["ripgrep.*"])]
        )
        provider = ScriptedProvider([result("never")])
        turn = await make_loop(provider, registry, emitter, manager).run_turn(
            "@code-search find it"
        )
        assert turn.status == "error"
        assert turn.error is not None
        assert "ripgrep.*" in turn.error  # the missing glob, by name
        assert provider.requests == []  # failed before any model call

    async def test_unsatisfiable_trigger_skill_skipped_with_trace(
        self, make_skill, registry, emitter
    ):
        manager = SkillManager(
            [
                make_skill(
                    "code-search", "Search.",
                    patterns=["find"], required_tools=["ripgrep.*"],
                )
            ]
        )
        provider = ScriptedProvider([result("done")])
        turn = await make_loop(provider, registry, emitter, manager).run_turn(
            "find the loader"
        )
        assert turn.status == "completed"  # auto-selection failure is not fatal
        skipped = [p for k, p in emitter.events if k == "skill.skipped"]
        assert skipped[0]["name"] == "code-search"
        assert "ripgrep.*" in skipped[0]["reason"]


class TestBudgetOverlay:
    async def test_skill_max_tokens_tightens_run_limit(
        self, make_skill, registry, emitter
    ):
        from agentloop.types import Usage

        manager = SkillManager(
            [
                make_skill(
                    "code-search", "Search.",
                    patterns=["find"], budget={"max_tokens": 500},
                )
            ]
        )
        provider = ScriptedProvider(
            [
                result(tool_calls=(echo_call("x"),), usage=Usage(input_tokens=600)),
                result("wrap-up"),
            ]
        )
        loop = make_loop(
            provider, registry, emitter, manager,
            limits=LimitsConfig(max_tokens=200_000),  # run limit is generous
        )
        turn = await loop.run_turn("find it")
        assert turn.status == "budget_exceeded"  # the skill's 500 bound, not 200k

    async def test_skill_max_tool_calls_forces_wrap_up(
        self, make_skill, registry, emitter
    ):
        manager = SkillManager(
            [
                make_skill(
                    "code-search", "Search.",
                    patterns=["find"], budget={"max_tool_calls": 1},
                )
            ]
        )
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("1"),)), result("wrap-up answer")]
        )
        turn = await make_loop(provider, registry, emitter, manager).run_turn(
            "find it"
        )
        assert turn.status == "budget_exceeded"
        assert turn.text == "wrap-up answer"
        reasons = [
            p["reason"] for k, p in emitter.events if k == "loop.transition"
        ]
        assert "budget:max_tool_calls" in reasons
