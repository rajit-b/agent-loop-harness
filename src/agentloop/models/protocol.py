"""ModelProvider Protocol and the request/result types it speaks (§6).

The loop only ever sees these types; each adapter owns the round-trip
translation to its provider's wire format.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agentloop.types import Cost, Message, ToolSpec, Usage


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


StopReason = Literal["stop", "tool_calls", "max_tokens", "refusal", "other"]


class CompletionRequest(_Frozen):
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    params: dict[str, Any] = Field(
        default_factory=dict, description="Per-request overrides of the adapter's params."
    )
    max_tokens: int | None = None


class CompletionResult(_Frozen):
    message: Message  # assistant role; tool_calls normalized to internal ToolCall
    stop_reason: StopReason
    usage: Usage
    cost: Cost
    provider: str  # the provider that actually served this (matters under fallback)
    model: str


class TextDelta(_Frozen):
    type: Literal["text_delta"] = "text_delta"
    text: str


class StreamCompleted(_Frozen):
    type: Literal["completed"] = "completed"
    result: CompletionResult


# Streaming contract: incremental text, then exactly one StreamCompleted
# carrying the full result (tool calls are not streamed incrementally in v1;
# they arrive fully-formed on the final result).
StreamEvent = TextDelta | StreamCompleted


@runtime_checkable
class ModelProvider(Protocol):
    """The single seam between the loop and any model backend."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def complete(self, request: CompletionRequest) -> CompletionResult: ...

    def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]: ...

    def tool_call_schema(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        """Translate internal ToolSpecs to this provider's tool payload."""
        ...

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate; used for budgeting, not billing (billing uses Usage)."""
        ...


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Shared chars/4 heuristic + small per-message overhead.

    Deliberately provider-agnostic: budget checks need a stable, cheap
    estimate, not exactness — real counts come back in Usage.
    """
    total = 0
    for message in messages:
        total += 4 + len(message.text) // 4
        for call in message.tool_calls:
            total += 4 + (len(call.name) + len(str(call.arguments))) // 4
    return total
