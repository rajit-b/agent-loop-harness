"""§11 injection points verified in assembled prompts — the gate evidence."""

from __future__ import annotations

import pytest

from agentloop.loop.machine import AgentLoop
from agentloop.memory.manager import MemoryManager
from agentloop.skills.manager import SkillManager
from agentloop.tools.builtin.echo import register_echo
from agentloop.tools.executor import ToolRegistry

from ..loop.conftest import ScriptedProvider, result
from ..skills.conftest import write_skill
from .conftest import candidate, extraction, make_manager

INTENT = "Answer questions about the codebase."
PERSONA = "Be precise."


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_echo(reg)
    return reg


async def seed_fact(mem_store, embedder, text: str) -> int:
    [embedding] = await embedder.embed([text])
    fact_id = mem_store.insert_fact(
        text=text, type="preference", confidence=0.9, quote="",
        session_id="s0", turn_id="t0", embedding=embedding,
    )
    mem_store.set_status(fact_id, "active")
    return fact_id


def make_loop(provider, registry, emitter, memory: MemoryManager, **kwargs):
    return AgentLoop(
        provider, registry, intent=INTENT, persona=PERSONA,
        emitter=emitter, memory=memory, **kwargs,
    )


class TestSystemPromptInjection:
    async def test_recalled_facts_enter_the_system_prompt_tagged(
        self, mem_store, embedder, registry, emitter
    ):
        fact_id = await seed_fact(mem_store, embedder, "User prefers config in YAML")
        manager, _ = make_manager(mem_store, embedder, [extraction()])
        provider = ScriptedProvider([result("answered")])
        loop = make_loop(provider, registry, emitter, manager)
        await loop.run_turn("how should the config loader work?")
        system = provider.requests[0].messages[0]
        assert system.role == "system"
        assert f"[fact:{fact_id}] User prefers config in YAML" in system.text

    async def test_full_system_prompt_ordering(
        self, mem_store, embedder, registry, emitter, tmp_path
    ):
        """§11 order: intent → persona → memory → skill index → bodies."""
        await seed_fact(mem_store, embedder, "User prefers config in YAML")
        manager, _ = make_manager(mem_store, embedder, [extraction()])
        skills = SkillManager(
            [__import__("agentloop.skills.loader", fromlist=["load_skill"]).load_skill(
                write_skill(tmp_path, "code-search", "Search code.",
                            patterns=["loader"], body="SKILL BODY HERE")
            )]
        )
        provider = ScriptedProvider([result("done")])
        loop = make_loop(provider, registry, emitter, manager, skills=skills)
        await loop.run_turn("where is the config loader?")
        text = provider.requests[0].messages[0].text
        positions = [
            text.index(INTENT),
            text.index(PERSONA),
            text.index("Known facts"),
            text.index("Available skills"),
            text.index("SKILL BODY HERE"),
        ]
        assert positions == sorted(positions), f"§11 order violated: {positions}"

    async def test_no_facts_no_memory_block(
        self, mem_store, embedder, registry, emitter
    ):
        manager, _ = make_manager(mem_store, embedder, [extraction()])
        provider = ScriptedProvider([result("done")])
        loop = make_loop(provider, registry, emitter, manager)
        await loop.run_turn("hello?")
        assert "Known facts" not in provider.requests[0].messages[0].text


class TestHistoryInjection:
    async def test_summary_then_tail_between_system_and_user(
        self, mem_store, embedder, registry, emitter
    ):
        mem_store.set_summary("session-1", "Earlier we fixed the loader.")
        mem_store.append_episodic("session-1", "t0", "user", "old question")
        mem_store.append_episodic("session-1", "t0", "assistant", "old answer")
        manager, _ = make_manager(mem_store, embedder, [extraction()])
        provider = ScriptedProvider([result("done")])
        loop = make_loop(provider, registry, emitter, manager)
        await loop.run_turn("new question")
        messages = provider.requests[0].messages
        assert messages[0].role == "system"
        assert "Earlier we fixed the loader." in messages[1].text  # summary first
        assert messages[2].text == "old question"
        assert messages[3].text == "old answer" and messages[3].role == "assistant"
        assert messages[4].text == "new question"  # current turn last

    async def test_turn_is_written_back_at_turn_end(
        self, mem_store, embedder, registry, emitter
    ):
        manager, _ = make_manager(mem_store, embedder, [extraction()])
        provider = ScriptedProvider([result("the loader is in config/")])
        loop = make_loop(provider, registry, emitter, manager)
        await loop.run_turn("where is the loader?")
        rows = mem_store.episodic_rows("session-1", include_folded=True)
        assert [(r.role, r.content) for r in rows] == [
            ("user", "where is the loader?"),
            ("assistant", "the loader is in config/"),
        ]

    async def test_second_turn_sees_first_turn_history(
        self, mem_store, embedder, registry, emitter
    ):
        manager, _ = make_manager(
            mem_store, embedder, [extraction(), extraction()]
        )
        provider = ScriptedProvider([result("first answer"), result("second answer")])
        loop = make_loop(provider, registry, emitter, manager)
        await loop.run_turn("first question")
        await loop.run_turn("second question")
        second_request = provider.requests[1]
        texts = [m.text for m in second_request.messages]
        assert "first question" in texts
        assert "first answer" in texts


class TestResilience:
    async def test_memory_failure_does_not_change_turn_result(
        self, mem_store, embedder, registry, emitter, monkeypatch
    ):
        manager, _ = make_manager(mem_store, embedder, [])

        async def broken(**kwargs):
            raise RuntimeError("db locked")

        monkeypatch.setattr(manager, "on_turn_end", broken)
        provider = ScriptedProvider([result("fine")])
        loop = make_loop(provider, registry, emitter, manager)
        turn = await loop.run_turn("q")
        assert turn.status == "completed"
        assert turn.text == "fine"
        assert any(k == "memory.error" for k, _ in emitter.events)


class TestEndToEndConsolidationThroughLoop:
    async def test_preference_stated_in_one_session_recalled_in_the_next(
        self, mem_store, embedder, registry, emitter
    ):
        """The §14 Phase-11 shape, provable already: an explicit preference
        from session A enters the system prompt of session B."""
        # session A: user states a directive; extractor sees it
        manager_a, _ = make_manager(
            mem_store, embedder,
            [extraction(candidate(
                "User prefers config in YAML", explicit=True,
                quote="always put config in YAML",
            ))],
            session_id="session-A",
        )
        provider_a = ScriptedProvider([result("noted!")])
        loop_a = make_loop(provider_a, registry, emitter, manager_a)
        await loop_a.run_turn("always put config in YAML please")

        # session B: fresh session, same store
        manager_b, _ = make_manager(
            mem_store, embedder, [extraction()], session_id="session-B"
        )
        provider_b = ScriptedProvider([result("using YAML.")])
        loop_b = make_loop(provider_b, registry, emitter, manager_b)
        await loop_b.run_turn("how should I add a config option?")
        system = provider_b.requests[0].messages[0].text
        assert "User prefers config in YAML" in system
        # and session A's chat log did NOT leak into session B's history
        assert all(
            "always put config in YAML please" != m.text
            for m in provider_b.requests[0].messages[1:]
        )
