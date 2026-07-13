"""The hook contract (§9), stated precisely.

A hook is `(payload, ctx) -> HookDecision | None`:

- `Continue()` / `None`          — pass the payload through unchanged;
- `Continue(payload=modified)`   — replace it (payloads are frozen pydantic
  models: mutation means returning a modified copy, so the trace can
  record exact before/after diffs);
- `Veto(reason)`                 — short-circuit: remaining hooks for the
  event do not run. What a veto *means* differs per event; that table is
  §9 and is enforced by the call sites (the loop), not the bus. The bus
  only refuses vetoes on the two events where nothing can be undone:
  on_error and on_turn_end.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from agentloop.config.manifest import HookEvent
from agentloop.models.protocol import CompletionRequest, CompletionResult
from agentloop.types import Cost, ToolCall, ToolResult, ToolSpec, Usage

P = TypeVar("P")

#: Events where a veto is meaningful. on_error may only shape handling
#: (mutation); on_turn_end is notify-only.
VETOABLE_EVENTS: frozenset[HookEvent] = frozenset(
    {"pre_model", "post_model", "pre_tool", "post_tool",
     "pre_retrieval", "post_retrieval"}
)


@dataclass(frozen=True, slots=True)
class Continue(Generic[P]):
    payload: P | None = None  # None = unchanged


@dataclass(frozen=True, slots=True)
class Veto:
    reason: str


HookDecision = Continue[P] | Veto


@dataclass(frozen=True, slots=True)
class HookContext:
    event: HookEvent
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)  # HookEntry.config


HookHandler = Callable[
    [Any, HookContext], "HookDecision[Any] | None | Awaitable[HookDecision[Any] | None]"
]


# ---------------------------------------------------------------------------
# Event payloads — frozen, event-specific
# ---------------------------------------------------------------------------


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PreModelPayload(_Frozen):
    request: CompletionRequest
    wrap_up: bool = False  # True on the forced budget wrap-up call


class PostModelPayload(_Frozen):
    result: CompletionResult
    wrap_up: bool = False


class PreToolPayload(_Frozen):
    call: ToolCall
    spec: ToolSpec | None = None


class PostToolPayload(_Frozen):
    call: ToolCall
    result: ToolResult


class PreRetrievalPayload(_Frozen):
    query: str
    top_k: int = 8


class PostRetrievalPayload(_Frozen):
    query: str
    chunks: tuple[Any, ...] = ()  # typed RetrievedChunk when RAG lands (Phase 8)


class ErrorPayload(_Frozen):
    error: str
    kind: str  # exception class name
    source: str  # "loop", "provider", "hook:<event>:<name>", ...


class TurnEndPayload(_Frozen):
    status: str
    steps: int
    usage: Usage
    cost: Cost
