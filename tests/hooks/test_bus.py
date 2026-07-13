"""HookBus semantics: ordering, mutation, veto, failure isolation."""

from __future__ import annotations

import threading

from agentloop.hooks.bus import HookBus
from agentloop.hooks.contract import (
    Continue,
    ErrorPayload,
    PreToolPayload,
    TurnEndPayload,
    Veto,
)
from agentloop.types import Cost, ToolCall, Usage


class RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind, payload) -> None:
        self.events.append((kind, dict(payload)))


def payload_with_text(text: str) -> PreToolPayload:
    return PreToolPayload(call=ToolCall(id="c1", name="echo", arguments={"text": text}))


def append_marker(marker: str):
    """A hook that appends `marker` to the text argument."""

    def hook(payload: PreToolPayload, ctx):
        call = payload.call
        new_text = call.arguments["text"] + marker
        return Continue(
            payload=payload.model_copy(
                update={"call": call.model_copy(update={"arguments": {"text": new_text}})}
            )
        )

    return hook


class TestOrderingAndMutation:
    async def test_mutations_compose_in_priority_order(self):
        bus = HookBus()
        bus.register("pre_tool", append_marker("B"), priority=20)
        bus.register("pre_tool", append_marker("A"), priority=10)
        outcome = await bus.dispatch("pre_tool", payload_with_text("x"))
        assert outcome.payload.call.arguments["text"] == "xAB"

    async def test_ties_break_by_registration_order(self):
        bus = HookBus()
        bus.register("pre_tool", append_marker("1"), priority=50)
        bus.register("pre_tool", append_marker("2"), priority=50)
        outcome = await bus.dispatch("pre_tool", payload_with_text(""))
        assert outcome.payload.call.arguments["text"] == "12"

    async def test_none_and_bare_continue_pass_through(self):
        bus = HookBus()
        bus.register("pre_tool", lambda p, c: None)
        bus.register("pre_tool", lambda p, c: Continue())
        original = payload_with_text("unchanged")
        outcome = await bus.dispatch("pre_tool", original)
        assert outcome.payload is original

    async def test_async_and_sync_handlers_mix(self):
        bus = HookBus()

        async def async_hook(payload, ctx):
            return append_marker("a")(payload, ctx)

        bus.register("pre_tool", async_hook, priority=1)
        bus.register("pre_tool", append_marker("s"), priority=2)
        outcome = await bus.dispatch("pre_tool", payload_with_text(""))
        assert outcome.payload.call.arguments["text"] == "as"

    async def test_blocking_sync_handler_runs_off_loop(self):
        bus = HookBus()
        thread_names = []

        def blocking_hook(payload, ctx):
            thread_names.append(threading.current_thread().name)
            return None

        bus.register("pre_tool", blocking_hook, blocking=True)
        await bus.dispatch("pre_tool", payload_with_text("x"))
        assert "MainThread" not in thread_names[0]


class TestVeto:
    async def test_veto_short_circuits_remaining_hooks(self):
        bus = HookBus()
        ran = []
        bus.register("pre_tool", lambda p, c: Veto("nope"), priority=1)
        bus.register("pre_tool", lambda p, c: ran.append(1), priority=2)
        outcome = await bus.dispatch("pre_tool", payload_with_text("x"))
        assert outcome.vetoed and outcome.veto.reason == "nope"
        assert ran == []

    async def test_veto_reports_the_hook_name(self):
        bus = HookBus()

        def policy_gate(p, c):
            return Veto("blocked")

        bus.register("pre_tool", policy_gate)
        outcome = await bus.dispatch("pre_tool", payload_with_text("x"))
        assert outcome.vetoed_by and "policy_gate" in outcome.vetoed_by

    async def test_veto_ignored_on_turn_end(self):
        emitter = RecordingEmitter()
        bus = HookBus(emitter=emitter)
        ran = []
        bus.register("on_turn_end", lambda p, c: Veto("try me"), priority=1)
        bus.register("on_turn_end", lambda p, c: ran.append(1) and None, priority=2)
        payload = TurnEndPayload(status="completed", steps=1, usage=Usage(), cost=Cost())
        outcome = await bus.dispatch("on_turn_end", payload)
        assert not outcome.vetoed
        assert ran == [1]  # chain continued past the ignored veto
        decisions = [p["decision"] for k, p in emitter.events if k == "hook.executed"]
        assert "veto_ignored" in decisions

    async def test_veto_ignored_on_error_event(self):
        bus = HookBus()
        bus.register("on_error", lambda p, c: Veto("cannot"))
        payload = ErrorPayload(error="x", kind="Boom", source="loop")
        outcome = await bus.dispatch("on_error", payload)
        assert not outcome.vetoed


class TestFailureIsolation:
    async def test_raising_hook_is_skipped_and_chain_continues(self):
        emitter = RecordingEmitter()
        bus = HookBus(emitter=emitter)

        def broken(p, c):
            raise RuntimeError("hook bug")

        bus.register("pre_tool", broken, priority=1)
        bus.register("pre_tool", append_marker("!"), priority=2)
        outcome = await bus.dispatch("pre_tool", payload_with_text("x"))
        assert outcome.payload.call.arguments["text"] == "x!"  # chain survived
        assert any(k == "hook.error" for k, _ in emitter.events)

    async def test_hook_failure_dispatches_on_error(self):
        bus = HookBus()
        seen: list[ErrorPayload] = []
        bus.register("on_error", lambda p, c: seen.append(p) and None)
        bus.register("pre_tool", lambda p, c: 1 / 0)
        await bus.dispatch("pre_tool", payload_with_text("x"))
        assert len(seen) == 1
        assert seen[0].kind == "ZeroDivisionError"
        assert seen[0].source.startswith("hook:pre_tool:")

    async def test_raising_on_error_hook_does_not_recurse(self):
        emitter = RecordingEmitter()
        bus = HookBus(emitter=emitter)
        bus.register("on_error", lambda p, c: 1 / 0)
        # dispatching on_error with a broken on_error hook must terminate
        await bus.dispatch(
            "on_error", ErrorPayload(error="x", kind="Boom", source="loop")
        )
        errors = [k for k, _ in emitter.events if k == "hook.error"]
        assert errors == ["hook.error"]  # exactly one, no recursion


class TestTracing:
    async def test_executions_trace_decision_and_changed_fields(self):
        emitter = RecordingEmitter()
        bus = HookBus(emitter=emitter)
        bus.register("pre_tool", append_marker("y"), name="mutator")
        bus.register("pre_tool", lambda p, c: None, name="observer")
        await bus.dispatch("pre_tool", payload_with_text("x"))
        executed = [p for k, p in emitter.events if k == "hook.executed"]
        assert executed[0]["decision"] == "mutated"
        assert executed[0]["changed_fields"] == ["call"]
        assert executed[1]["decision"] == "continue"
