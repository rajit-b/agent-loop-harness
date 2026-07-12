from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from decimal import Decimal
from typing import Any

import pytest

from agentloop.models.protocol import (
    CompletionRequest,
    CompletionResult,
    StreamEvent,
)
from agentloop.tools.builtin.echo import register_echo
from agentloop.tools.executor import ToolRegistry
from agentloop.types import Cost, Message, TextPart, ToolCall, ToolSpec, Usage


def result(
    text: str = "",
    tool_calls: tuple[ToolCall, ...] = (),
    *,
    usage: Usage | None = None,
    cost: Cost | None = None,
) -> CompletionResult:
    return CompletionResult(
        message=Message(
            role="assistant",
            content=(TextPart(text=text),) if text else (),
            tool_calls=tool_calls,
        ),
        stop_reason="tool_calls" if tool_calls else "stop",
        usage=usage or Usage(input_tokens=10, output_tokens=5),
        cost=cost or Cost(usd=Decimal("0.001")),
        provider="fake",
        model="fake-model",
    )


def echo_call(text: str, call_id: str = "call_0") -> ToolCall:
    return ToolCall(id=call_id, name="echo", arguments={"text": text})


class ScriptedProvider:
    """Returns scripted results in order; records every request it saw."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, script: Sequence[CompletionResult | Exception]):
        self.script = list(script)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError
        yield  # pragma: no cover

    def tool_call_schema(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        return []

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return 0


class RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind: str, payload) -> None:
        self.events.append((kind, dict(payload)))

    def transitions(self) -> list[tuple[str, str]]:
        return [
            (p["from"], p["to"])
            for kind, p in self.events
            if kind == "loop.transition"
        ]

    def transition_reasons(self) -> list[str | None]:
        return [p["reason"] for kind, p in self.events if kind == "loop.transition"]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_echo(reg)
    return reg


@pytest.fixture
def emitter() -> RecordingEmitter:
    return RecordingEmitter()
