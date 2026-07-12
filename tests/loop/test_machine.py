"""State machine: transition sequences, budgets, errors, cancellation."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from agentloop.config.manifest import LimitsConfig
from agentloop.loop.machine import AgentLoop, State
from agentloop.tools.executor import ToolRegistry
from agentloop.types import (
    Cost,
    ProviderExhaustedError,
    ToolCall,
    ToolSpec,
    Usage,
)

from .conftest import (
    FakeClock,
    ScriptedProvider,
    echo_call,
    result,
)

INTENT = "Answer questions about the codebase."

HAPPY_PATH = [
    ("start", State.PERCEIVE),
    (State.PERCEIVE, State.RETRIEVE),
    (State.RETRIEVE, State.PLAN),
    (State.PLAN, State.ACT),
    (State.ACT, State.OBSERVE),
    (State.OBSERVE, State.PLAN),
    (State.PLAN, State.REFLECT),
    (State.REFLECT, State.TERMINATE),
]


def make_loop(provider, registry, emitter, **kwargs) -> AgentLoop:
    return AgentLoop(provider, registry, intent=INTENT, emitter=emitter, **kwargs)


class TestHappyPath:
    async def test_transition_sequence(self, registry, emitter):
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("hi"),)), result("The answer.")]
        )
        turn = await make_loop(provider, registry, emitter).run_turn("question?")
        assert turn.status == "completed"
        assert turn.text == "The answer."
        assert emitter.transitions() == HAPPY_PATH

    async def test_tool_result_reaches_the_model(self, registry, emitter):
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("hello world"),)), result("done")]
        )
        await make_loop(provider, registry, emitter).run_turn("q")
        second_request = provider.requests[1]
        tool_messages = [m for m in second_request.messages if m.role == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].text == "hello world"
        assert tool_messages[0].tool_call_id == "call_0"

    async def test_system_prompt_carries_intent_and_persona(self, registry, emitter):
        provider = ScriptedProvider([result("hi")])
        loop = AgentLoop(
            provider, registry, intent=INTENT, persona="Be terse.", emitter=emitter
        )
        await loop.run_turn("q")
        system = provider.requests[0].messages[0]
        assert system.role == "system"
        assert INTENT in system.text and "Be terse." in system.text

    async def test_usage_and_cost_accumulate_across_calls(self, registry, emitter):
        provider = ScriptedProvider(
            [
                result(
                    tool_calls=(echo_call("x"),),
                    usage=Usage(input_tokens=100, output_tokens=10),
                    cost=Cost(usd=Decimal("0.01")),
                ),
                result(
                    "done",
                    usage=Usage(input_tokens=200, output_tokens=20),
                    cost=Cost(usd=Decimal("0.02")),
                ),
            ]
        )
        turn = await make_loop(provider, registry, emitter).run_turn("q")
        assert turn.usage.input_tokens == 300
        assert turn.usage.output_tokens == 30
        assert turn.cost.usd == Decimal("0.03")
        assert turn.steps == 2

    async def test_parallel_tool_calls_keep_call_order(self, emitter):
        registry = ToolRegistry()
        finish_first = asyncio.Event()

        async def slow(arguments):
            await finish_first.wait()
            return "slow-result"

        async def fast(arguments):
            finish_first.set()  # proves both run concurrently
            return "fast-result"

        registry.register(ToolSpec(name="slow"), slow)
        registry.register(ToolSpec(name="fast"), fast)
        provider = ScriptedProvider(
            [
                result(
                    tool_calls=(
                        ToolCall(id="c0", name="slow"),
                        ToolCall(id="c1", name="fast"),
                    )
                ),
                result("done"),
            ]
        )
        await make_loop(provider, registry, emitter).run_turn("q")
        tool_messages = [m for m in provider.requests[1].messages if m.role == "tool"]
        # results appended in call order even though fast finished first
        assert [m.text for m in tool_messages] == ["slow-result", "fast-result"]


class TestBudgets:
    async def test_step_cap_forces_wrap_up(self, registry, emitter):
        provider = ScriptedProvider(
            [
                result(tool_calls=(echo_call("1"),)),
                result(tool_calls=(echo_call("2"),)),
                result("best effort answer"),  # the wrap-up call
            ]
        )
        loop = make_loop(
            provider, registry, emitter, limits=LimitsConfig(max_steps=2)
        )
        turn = await loop.run_turn("q")
        assert turn.status == "budget_exceeded"
        assert turn.steps == 2
        assert turn.text == "best effort answer"
        # the wrap-up call had tools disabled
        assert provider.requests[-1].tools == ()
        # and the nudge names the budget
        assert "max_steps" in provider.requests[-1].messages[-1].text
        assert (State.OBSERVE, State.REFLECT) in emitter.transitions()
        assert "budget:max_steps" in emitter.transition_reasons()

    async def test_token_budget_forces_wrap_up(self, registry, emitter):
        provider = ScriptedProvider(
            [
                result(
                    tool_calls=(echo_call("x"),),
                    usage=Usage(input_tokens=90_000, output_tokens=20_000),
                ),
                result("wrap-up"),
            ]
        )
        loop = make_loop(
            provider, registry, emitter, limits=LimitsConfig(max_tokens=100_000)
        )
        turn = await loop.run_turn("q")
        assert turn.status == "budget_exceeded"
        assert "budget:max_tokens" in emitter.transition_reasons()

    async def test_wall_clock_budget(self, registry, emitter):
        clock = FakeClock()
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("x"),)), result("wrap-up")]
        )

        original = provider.complete

        async def slow_complete(request):
            clock.now += 400.0  # each model call "takes" 400s
            return await original(request)

        provider.complete = slow_complete
        loop = make_loop(
            provider, registry, emitter,
            limits=LimitsConfig(max_wall_clock_s=300), clock=clock,
        )
        turn = await loop.run_turn("q")
        assert turn.status == "budget_exceeded"
        assert "budget:max_wall_clock_s" in emitter.transition_reasons()

    async def test_cost_budget(self, registry, emitter):
        provider = ScriptedProvider(
            [
                result(tool_calls=(echo_call("x"),), cost=Cost(usd=Decimal("2.50"))),
                result("wrap-up"),
            ]
        )
        loop = make_loop(
            provider, registry, emitter,
            limits=LimitsConfig(max_cost_usd=Decimal("1.00")),
        )
        turn = await loop.run_turn("q")
        assert turn.status == "budget_exceeded"
        assert "budget:max_cost_usd" in emitter.transition_reasons()

    async def test_pending_tool_calls_complete_before_wrap_up(self, registry, emitter):
        """Budgets only redirect entries into PLAN: the ACT for already-issued
        calls still runs, so no dangling tool_use ever reaches a provider."""
        provider = ScriptedProvider(
            [
                result(
                    tool_calls=(echo_call("x"),),
                    usage=Usage(input_tokens=999_999),  # blows the budget instantly
                ),
                result("wrap-up"),
            ]
        )
        loop = make_loop(
            provider, registry, emitter, limits=LimitsConfig(max_tokens=100)
        )
        await loop.run_turn("q")
        transitions = emitter.transitions()
        assert (State.PLAN, State.ACT) in transitions  # ACT still happened
        wrap_up_messages = provider.requests[-1].messages
        assert any(m.role == "tool" for m in wrap_up_messages)


class TestErrors:
    async def test_tool_failure_becomes_error_result(self, emitter):
        registry = ToolRegistry()

        async def broken(arguments):
            raise ValueError("disk on fire")

        registry.register(ToolSpec(name="broken"), broken)
        provider = ScriptedProvider(
            [
                result(tool_calls=(ToolCall(id="c0", name="broken"),)),
                result("recovered"),
            ]
        )
        turn = await make_loop(provider, registry, emitter).run_turn("q")
        assert turn.status == "completed"  # the loop survived
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert tool_message.is_error is True
        assert "ValueError" in tool_message.text

    async def test_unknown_tool_becomes_error_result(self, registry, emitter):
        provider = ScriptedProvider(
            [
                result(tool_calls=(ToolCall(id="c0", name="nonexistent"),)),
                result("ok"),
            ]
        )
        await make_loop(provider, registry, emitter).run_turn("q")
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert tool_message.is_error is True
        assert "unknown tool" in tool_message.text

    async def test_provider_exhaustion_terminates_with_error(self, registry, emitter):
        provider = ScriptedProvider([ProviderExhaustedError(["a/m: 500", "b/m: 401"])])
        turn = await make_loop(provider, registry, emitter).run_turn("q")
        assert turn.status == "error"
        assert turn.error is not None and "exhausted" in turn.error
        assert emitter.transitions()[-1] == (State.PLAN, State.TERMINATE)


class TestCancellation:
    async def test_cancel_mid_act_terminates_cleanly(self, emitter):
        registry = ToolRegistry()
        tool_started = asyncio.Event()

        async def hang(arguments):
            tool_started.set()
            await asyncio.Event().wait()  # blocks forever
            return "never"

        registry.register(ToolSpec(name="hang"), hang)
        provider = ScriptedProvider(
            [result(tool_calls=(ToolCall(id="c0", name="hang"),))]
        )
        loop = make_loop(provider, registry, emitter)
        task = asyncio.create_task(loop.run_turn("q"))
        await tool_started.wait()  # we are mid-ACT now
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        last_kind, last_payload = emitter.events[-1]
        assert last_kind == "loop.transition"
        assert last_payload["to"] == State.TERMINATE
        assert last_payload["reason"] == "cancelled"


class TestTraceEvents:
    async def test_model_and_tool_events_emitted_with_run_id_and_seq(
        self, registry, emitter
    ):
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("x"),)), result("done")]
        )
        await make_loop(provider, registry, emitter).run_turn("q")
        kinds = [k for k, _ in emitter.events]
        assert kinds.count("model.complete") == 2
        assert kinds.count("tool.call") == 1
        assert kinds.count("tool.result") == 1
        run_ids = {p["run_id"] for _, p in emitter.events}
        assert len(run_ids) == 1  # one run, one id
        seqs = [p["seq"] for _, p in emitter.events]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)  # monotonic
