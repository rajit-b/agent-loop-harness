"""RETRIEVE wired into the loop: context injection + the §9 retrieval hooks."""

from __future__ import annotations

import pytest

from agentloop.hooks.bus import HookBus
from agentloop.hooks.contract import Continue, Veto
from agentloop.loop.machine import AgentLoop
from agentloop.rag.ingest import Indexer
from agentloop.rag.retrieve import Retriever
from agentloop.rag.sources import FileSource
from agentloop.tools.builtin.echo import register_echo
from agentloop.tools.executor import ToolRegistry

from ..loop.conftest import ScriptedProvider, result

INTENT = "Answer questions about the framework."


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_echo(reg)
    return reg


@pytest.fixture
async def retriever(store, embedder, docs_dir) -> Retriever:
    await Indexer(store, embedder, chunk_size=128, chunk_overlap=16).index(
        FileSource([docs_dir])
    )
    return Retriever(store, embedder, top_k=2)


def make_loop(provider, registry, emitter, retriever, bus=None) -> AgentLoop:
    return AgentLoop(
        provider, registry, intent=INTENT, emitter=emitter,
        retriever=retriever, hooks=bus,
    )


class TestInjection:
    async def test_cited_context_attaches_to_the_user_message(
        self, registry, emitter, retriever
    ):
        provider = ScriptedProvider([result("answered")])
        loop = make_loop(provider, registry, emitter, retriever)
        await loop.run_turn("how does the config loader resolve precedence?")
        [request] = provider.requests
        user_message = request.messages[1]
        assert user_message.role == "user"
        assert "Retrieved context" in user_message.text
        assert "config.md" in user_message.text  # citation location
        assert "[1]" in user_message.text  # citation marker
        assert "precedence" in user_message.text  # actual chunk content
        events = [k for k, _ in emitter.events]
        assert "retrieval.complete" in events

    async def test_no_retriever_means_no_injection(self, registry, emitter):
        provider = ScriptedProvider([result("done")])
        loop = make_loop(provider, registry, emitter, retriever=None)
        await loop.run_turn("anything")
        assert "Retrieved context" not in provider.requests[0].messages[1].text


class TestRetrievalHooks:
    async def test_pre_retrieval_veto_skips_retrieval(
        self, registry, emitter, retriever, embedder
    ):
        bus = HookBus(emitter=emitter)
        bus.register("pre_retrieval", lambda p, c: Veto("query contains PII"))
        provider = ScriptedProvider([result("done")])
        calls_before = embedder.calls
        loop = make_loop(provider, registry, emitter, retriever, bus)
        turn = await loop.run_turn("config loader?")
        assert turn.status == "completed"
        assert embedder.calls == calls_before  # retrieval never ran
        assert "Retrieved context" not in provider.requests[0].messages[1].text
        assert any(k == "retrieval.skipped" for k, _ in emitter.events)

    async def test_pre_retrieval_mutation_rewrites_the_query(
        self, registry, emitter, retriever
    ):
        bus = HookBus(emitter=emitter)

        def rewrite(payload, ctx):
            return Continue(
                payload=payload.model_copy(update={"query": "ZQXW-7741 budgets"})
            )

        bus.register("pre_retrieval", rewrite)
        provider = ScriptedProvider([result("done")])
        loop = make_loop(provider, registry, emitter, retriever, bus)
        await loop.run_turn("something entirely different")
        user_text = provider.requests[0].messages[1].text
        # the mutated query drove retrieval: budgets.md content was injected
        assert "budgets.md" in user_text

    async def test_post_retrieval_veto_drops_all_context(
        self, registry, emitter, retriever
    ):
        bus = HookBus(emitter=emitter)
        bus.register("post_retrieval", lambda p, c: Veto("context untrusted"))
        provider = ScriptedProvider([result("done")])
        loop = make_loop(provider, registry, emitter, retriever, bus)
        await loop.run_turn("config loader precedence?")
        assert "Retrieved context" not in provider.requests[0].messages[1].text
        complete = [p for k, p in emitter.events if k == "retrieval.complete"]
        assert complete[0]["dropped"] is True

    async def test_post_retrieval_mutation_filters_chunks(
        self, registry, emitter, retriever
    ):
        bus = HookBus(emitter=emitter)

        def keep_first(payload, ctx):
            return Continue(
                payload=payload.model_copy(update={"chunks": payload.chunks[:1]})
            )

        bus.register("post_retrieval", keep_first)
        provider = ScriptedProvider([result("done")])
        loop = make_loop(provider, registry, emitter, retriever, bus)
        await loop.run_turn("config loader precedence?")
        user_text = provider.requests[0].messages[1].text
        assert "[1]" in user_text
        assert "[2]" not in user_text  # second chunk was filtered out
