"""The agent loop state machine (§7).

All seven states exist from this phase so the machine's shape never
migrates: RETRIEVE is a structural no-op until Phases 8-9 wire memory and
RAG into it; REFLECT is pass-through except for the forced budget wrap-up
(the optional self-critique call arrives when there are criteria to check
against). Invariants:

- every transition emits a `loop.transition` trace event before the
  target state runs;
- budgets are checked at transition boundaries, never inside states, and
  only redirect transitions INTO PLAN — in-flight tool calls always
  complete, so no provider ever sees a dangling tool_use;
- a forced wrap-up gives the model one final call with tools disabled;
- tool failures become error ToolResults, never loop exceptions;
- cancellation flushes a TERMINATE(cancelled) event, then re-raises.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agentloop.config.manifest import LimitsConfig
from agentloop.loop.budgets import BudgetTracker
from agentloop.loop.context import TurnContext, TurnStatus, build_system_prompt
from agentloop.models.protocol import CompletionRequest, CompletionResult, ModelProvider
from agentloop.tools.executor import ToolExecutor
from agentloop.types import (
    AgentLoopError,
    Cost,
    Message,
    NullEmitter,
    ToolResult,
    TraceEmitter,
    Usage,
)


class State(StrEnum):
    PERCEIVE = "perceive"
    RETRIEVE = "retrieve"
    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    REFLECT = "reflect"
    TERMINATE = "terminate"


WRAP_UP_NUDGE = (
    "Budget exhausted ({reason}). Stop using tools and give your best "
    "final answer now from what you already have."
)


@dataclass(slots=True)
class TurnResult:
    status: TurnStatus
    text: str
    usage: Usage
    cost: Cost
    steps: int
    error: str | None = None


class AgentLoop:
    def __init__(
        self,
        provider: ModelProvider,
        executor: ToolExecutor,
        *,
        intent: str,
        persona: str = "",
        limits: LimitsConfig | None = None,
        emitter: TraceEmitter | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._provider = provider
        self._executor = executor
        self._intent = intent
        self._persona = persona
        self._limits = limits or LimitsConfig()
        self._emitter = emitter or NullEmitter()
        self._clock = clock

    async def run_turn(self, user_input: str) -> TurnResult:
        ctx = TurnContext(
            user_input=user_input,
            budgets=BudgetTracker(self._limits, clock=self._clock),
        )
        handlers: Mapping[State, Any] = {
            State.PERCEIVE: self._perceive,
            State.RETRIEVE: self._retrieve,
            State.PLAN: self._plan,
            State.ACT: self._act,
            State.OBSERVE: self._observe,
            State.REFLECT: self._reflect,
        }
        state = State.PERCEIVE
        self._emit(ctx, "loop.transition", {"from": "start", "to": state, "reason": None})
        try:
            while state is not State.TERMINATE:
                next_state, reason = await handlers[state](ctx)
                override = ctx.budgets.exceeded(
                    entering_plan=(next_state is State.PLAN)
                )
                if override and next_state is State.PLAN:
                    ctx.budget_reason = override
                    next_state, reason = State.REFLECT, f"budget:{override}"
                self._emit(
                    ctx,
                    "loop.transition",
                    {"from": state, "to": next_state, "reason": reason},
                )
                state = next_state
        except asyncio.CancelledError:
            ctx.status = "cancelled"
            self._emit(
                ctx,
                "loop.transition",
                {"from": state, "to": State.TERMINATE, "reason": "cancelled"},
            )
            raise
        except AgentLoopError as exc:
            ctx.status = "error"
            ctx.error = str(exc)
            self._emit(
                ctx,
                "loop.transition",
                {"from": state, "to": State.TERMINATE, "reason": f"error:{exc}"},
            )
        return TurnResult(
            status=ctx.status,
            text=ctx.final_text,
            usage=ctx.budgets.usage,
            cost=ctx.budgets.cost,
            steps=ctx.budgets.steps,
            error=ctx.error,
        )

    # ------------------------------------------------------------------
    # State handlers: each returns (next_state, transition_reason)
    # ------------------------------------------------------------------

    async def _perceive(self, ctx: TurnContext) -> tuple[State, str | None]:
        ctx.messages.append(
            Message.system(build_system_prompt(self._intent, self._persona))
        )
        ctx.messages.append(Message.user(ctx.user_input))
        return State.RETRIEVE, None

    async def _retrieve(self, ctx: TurnContext) -> tuple[State, str | None]:
        return State.PLAN, None  # memory recall + RAG land here (Phases 8-9)

    async def _plan(self, ctx: TurnContext) -> tuple[State, str | None]:
        ctx.budgets.record_plan_entry()
        result = await self._model_call(ctx, tools=self._executor.specs())
        if result.message.tool_calls:
            return State.ACT, "tool_calls"
        return State.REFLECT, "final_answer"

    async def _act(self, ctx: TurnContext) -> tuple[State, str | None]:
        calls = ctx.messages[-1].tool_calls
        results: list[ToolResult | None] = [None] * len(calls)

        async def run_one(index: int, call) -> None:
            self._emit(
                ctx, "tool.call",
                {"id": call.id, "name": call.name, "arguments": call.arguments},
            )
            try:
                result = await self._executor.execute(call)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — never let a tool kill the loop
                result = ToolResult(
                    tool_call_id=call.id,
                    content=f"tool {call.name!r} failed: {type(exc).__name__}: {exc}",
                    is_error=True,
                )
            self._emit(
                ctx, "tool.result",
                {"id": call.id, "name": call.name, "is_error": result.is_error},
            )
            results[index] = result

        async with asyncio.TaskGroup() as tg:
            for i, call in enumerate(calls):
                tg.create_task(run_one(i, call))

        for call, result in zip(calls, results, strict=True):
            assert result is not None
            ctx.messages.append(
                Message.tool_result(
                    result.tool_call_id,
                    result.content,
                    name=call.name,
                    is_error=result.is_error,
                )
            )
        return State.OBSERVE, None

    async def _observe(self, ctx: TurnContext) -> tuple[State, str | None]:
        return State.PLAN, None  # the boundary check may redirect to REFLECT

    async def _reflect(self, ctx: TurnContext) -> tuple[State, str | None]:
        if ctx.budget_reason is None:
            return State.TERMINATE, "done"
        # Forced wrap-up: if the model hasn't produced a final answer yet,
        # give it one last call with tools disabled.
        if ctx.messages[-1].role != "assistant" or not ctx.messages[-1].text:
            ctx.messages.append(
                Message.user(WRAP_UP_NUDGE.format(reason=ctx.budget_reason))
            )
            await self._model_call(ctx, tools=(), wrap_up=True)
        ctx.status = "budget_exceeded"
        return State.TERMINATE, f"budget:{ctx.budget_reason}"

    # ------------------------------------------------------------------

    async def _model_call(
        self, ctx: TurnContext, *, tools, wrap_up: bool = False
    ) -> CompletionResult:
        request = CompletionRequest(messages=tuple(ctx.messages), tools=tuple(tools))
        result = await self._provider.complete(request)
        ctx.budgets.add(result.usage, result.cost)
        ctx.messages.append(result.message)
        self._emit(
            ctx,
            "model.complete",
            {
                "provider": result.provider,
                "model": result.model,
                "stop_reason": result.stop_reason,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cost_usd": str(result.cost.usd),
                "wrap_up": wrap_up,
            },
        )
        return result

    def _emit(self, ctx: TurnContext, kind: str, payload: dict[str, Any]) -> None:
        self._emitter.emit(
            kind, {"run_id": ctx.run_id, "seq": ctx.next_seq(), **payload}
        )
