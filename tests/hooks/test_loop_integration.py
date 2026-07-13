"""Hooks wired into the loop: every §9 veto row that has a Phase-5 surface."""

from __future__ import annotations

from agentloop.hooks.bus import HookBus
from agentloop.hooks.contract import Continue, TurnEndPayload, Veto
from agentloop.loop.machine import AgentLoop, State

from ..loop.conftest import ScriptedProvider, echo_call, result

INTENT = "Answer questions."


def make_loop(provider, registry, emitter, bus, **kwargs) -> AgentLoop:
    return AgentLoop(
        provider, registry, intent=INTENT, emitter=emitter, hooks=bus, **kwargs
    )


def mutate_echo_args(transform):
    def hook(payload, ctx):
        call = payload.call
        if call.name != "echo":
            return None
        new_arguments = {"text": transform(call.arguments.get("text", ""))}
        return Continue(
            payload=payload.model_copy(
                update={"call": call.model_copy(update={"arguments": new_arguments})}
            )
        )

    return hook


class TestPreTool:
    async def test_mutating_hook_observably_alters_arguments(self, registry, emitter):
        """The gate's headline case: echo returns the MUTATED text."""
        bus = HookBus(emitter=emitter)
        bus.register("pre_tool", mutate_echo_args(str.upper))
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("quiet please"),)), result("done")]
        )
        await make_loop(provider, registry, emitter, bus).run_turn("q")
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert tool_message.text == "QUIET PLEASE"

    async def test_redaction_style_mutation(self, registry, emitter):
        bus = HookBus(emitter=emitter)
        bus.register(
            "pre_tool", mutate_echo_args(lambda t: t.replace("sk-live-42", "[REDACTED]"))
        )
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("key is sk-live-42"),)), result("done")]
        )
        await make_loop(provider, registry, emitter, bus).run_turn("q")
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert "sk-live-42" not in tool_message.text
        assert "[REDACTED]" in tool_message.text

    async def test_veto_skips_execution_and_feeds_error_result(self, emitter):
        from agentloop.tools.executor import ToolRegistry
        from agentloop.types import ToolSpec

        registry = ToolRegistry()
        executed = []

        async def spy(arguments):
            executed.append(arguments)
            return "ran"

        registry.register(ToolSpec(name="echo"), spy)
        bus = HookBus(emitter=emitter)
        bus.register("pre_tool", lambda p, c: Veto("secrets in arguments"))
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("x"),)), result("recovered")]
        )
        turn = await make_loop(provider, registry, emitter, bus).run_turn("q")
        assert turn.status == "completed"  # per §9: the LOOP continues
        assert executed == []  # the tool never ran
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert tool_message.is_error is True
        assert "vetoed" in tool_message.text and "secrets" in tool_message.text


class TestPostTool:
    async def test_mutation_rewrites_the_result(self, registry, emitter):
        bus = HookBus(emitter=emitter)

        def scrub(payload, ctx):
            return Continue(
                payload=payload.model_copy(
                    update={
                        "result": payload.result.model_copy(
                            update={"content": payload.result.content.replace("hi", "**")}
                        )
                    }
                )
            )

        bus.register("post_tool", scrub)
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("hi there"),)), result("done")]
        )
        await make_loop(provider, registry, emitter, bus).run_turn("q")
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert tool_message.text == "** there"

    async def test_veto_replaces_result_with_error(self, registry, emitter):
        bus = HookBus(emitter=emitter)
        bus.register("post_tool", lambda p, c: Veto("result contained PII"))
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("ssn 123"),)), result("done")]
        )
        turn = await make_loop(provider, registry, emitter, bus).run_turn("q")
        assert turn.status == "completed"
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert tool_message.is_error is True
        assert "PII" in tool_message.text
        assert "ssn 123" not in tool_message.text  # original content is gone


class TestPreModel:
    async def test_mutation_alters_the_request(self, registry, emitter):
        bus = HookBus(emitter=emitter)

        def drop_tools(payload, ctx):
            return Continue(
                payload=payload.model_copy(
                    update={"request": payload.request.model_copy(update={"tools": ()})}
                )
            )

        bus.register("pre_model", drop_tools)
        provider = ScriptedProvider([result("done")])
        await make_loop(provider, registry, emitter, bus).run_turn("q")
        assert provider.requests[0].tools == ()  # provider saw the mutated request

    async def test_veto_aborts_the_turn(self, registry, emitter):
        bus = HookBus(emitter=emitter)
        bus.register("pre_model", lambda p, c: Veto("prompt injection detected"))
        provider = ScriptedProvider([result("never")])
        turn = await make_loop(provider, registry, emitter, bus).run_turn("q")
        assert turn.status == "vetoed"
        assert turn.error is not None and "prompt injection" in turn.error
        assert provider.requests == []  # the model was never called
        last = emitter.transitions()[-1]
        assert last == (State.PLAN, State.TERMINATE)


class TestPostModel:
    async def test_veto_discards_completion_and_replans(self, registry, emitter):
        bus = HookBus(emitter=emitter)
        vetoed_once = []

        def veto_first(payload, ctx):
            if not vetoed_once:
                vetoed_once.append(True)
                return Veto("try again")
            return None

        bus.register("post_model", veto_first)
        provider = ScriptedProvider([result("draft answer"), result("final answer")])
        turn = await make_loop(provider, registry, emitter, bus).run_turn("q")
        assert turn.status == "completed"
        assert turn.text == "final answer"
        assert turn.steps == 2  # the discarded call counted against the step cap
        # the discarded draft never entered the second request's history
        assert all("draft answer" not in m.text for m in provider.requests[1].messages)
        assert (State.PLAN, State.PLAN) in emitter.transitions()
        kinds = [k for k, _ in emitter.events]
        assert "model.discarded" in kinds


class TestLifecycleEvents:
    async def test_on_turn_end_receives_final_accounting(self, registry, emitter):
        bus = HookBus(emitter=emitter)
        seen: list[TurnEndPayload] = []
        bus.register("on_turn_end", lambda p, c: seen.append(p) and None)
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("x"),)), result("done")]
        )
        turn = await make_loop(provider, registry, emitter, bus).run_turn("q")
        assert len(seen) == 1
        assert seen[0].status == "completed"
        assert seen[0].steps == turn.steps == 2
        assert seen[0].usage == turn.usage

    async def test_on_error_fires_on_provider_exhaustion(self, registry, emitter):
        from agentloop.types import ProviderExhaustedError

        bus = HookBus(emitter=emitter)
        seen = []
        bus.register("on_error", lambda p, c: seen.append(p) and None)
        provider = ScriptedProvider([ProviderExhaustedError(["a/m: down"])])
        turn = await make_loop(provider, registry, emitter, bus).run_turn("q")
        assert turn.status == "error"
        assert len(seen) == 1
        assert seen[0].kind == "ProviderExhaustedError"

    async def test_throwing_hook_does_not_kill_the_run(self, registry, emitter):
        bus = HookBus(emitter=emitter)

        def broken(payload, ctx):
            raise RuntimeError("bad hook")

        bus.register("pre_tool", broken)
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("hi"),)), result("survived")]
        )
        turn = await make_loop(provider, registry, emitter, bus).run_turn("q")
        assert turn.status == "completed"
        assert turn.text == "survived"
        # and the tool still ran with the ORIGINAL arguments
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert tool_message.text == "hi"


class TestPriorityThroughTheLoop:
    async def test_mutations_compose_by_priority(self, registry, emitter):
        bus = HookBus(emitter=emitter)
        bus.register("pre_tool", mutate_echo_args(lambda t: t + "-second"), priority=20)
        bus.register("pre_tool", mutate_echo_args(lambda t: t + "-first"), priority=10)
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("base"),)), result("done")]
        )
        await make_loop(provider, registry, emitter, bus).run_turn("q")
        tool_message = [m for m in provider.requests[1].messages if m.role == "tool"][0]
        assert tool_message.text == "base-first-second"


class TestWrapUpInteraction:
    async def test_post_model_veto_ignored_on_wrap_up_call(self, registry, emitter):
        """A wrap-up completion cannot be re-planned; veto is traced, not honored."""
        from agentloop.config.manifest import LimitsConfig

        bus = HookBus(emitter=emitter)
        bus.register("post_model", lambda p, c: Veto("always veto"))
        provider = ScriptedProvider(
            [result(tool_calls=(echo_call("x"),)), result("salvaged answer")]
        )
        loop = make_loop(
            provider, registry, emitter, bus, limits=LimitsConfig(max_steps=1)
        )
        turn = await loop.run_turn("q")
        assert turn.status == "budget_exceeded"
        assert turn.text == "salvaged answer"  # wrap-up result kept despite veto
