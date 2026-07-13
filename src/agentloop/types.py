"""Core shared types (§4 of docs/ARCHITECTURE.md).

Imported by every layer; imports nothing from the package. This module is
the lingua franca: provider wire formats are translated at adapter
boundaries and never leak past them.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AgentLoopError(Exception):
    """Base for all framework errors."""


class ConfigError(AgentLoopError):
    """Fatal, pre-run: the manifest or its overrides are invalid."""


class ProviderError(AgentLoopError):
    """A model provider failed in a way that retrying will not fix
    (bad request, auth). The fallback chain advances immediately."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class TransientProviderError(ProviderError):
    """Retryable provider failure: network error, timeout, 429, 5xx."""


class HookVeto(AgentLoopError):
    """A hook vetoed an operation whose veto semantics abort the turn."""

    def __init__(self, event: str, reason: str):
        self.event = event
        self.reason = reason
        super().__init__(f"vetoed at {event}: {reason}")


class ProviderExhaustedError(AgentLoopError):
    """Every provider in the fallback chain failed; fatal for the turn."""

    def __init__(self, attempts: list[str]):
        self.attempts = attempts
        super().__init__("all providers exhausted: " + "; ".join(attempts))


# ---------------------------------------------------------------------------
# Observability seam (§3 rule 3: injected, never imported from observability)
# ---------------------------------------------------------------------------


@runtime_checkable
class TraceEmitter(Protocol):
    """Structural interface every component emits trace events through.

    Implementations (JSONL sink, sqlite index, replay recorder) live in
    agentloop.observability (Phase 10); the Protocol lives here so no layer
    ever imports that package.
    """

    def emit(self, kind: str, payload: Mapping[str, Any]) -> None: ...


class NullEmitter:
    """Default no-op emitter."""

    def emit(self, kind: str, payload: Mapping[str, Any]) -> None:  # noqa: ARG002
        return None


# ---------------------------------------------------------------------------
# Messages and tool calling — the single internal representation
# ---------------------------------------------------------------------------


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Role = Literal["system", "user", "assistant", "tool"]


class TextPart(_Frozen):
    type: Literal["text"] = "text"
    text: str


# Union grows as content kinds land (ImagePart is reserved, not v1).
ContentPart = TextPart


class ToolCall(_Frozen):
    """A model's request to invoke a tool, normalized across providers."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Citation(_Frozen):
    source_uri: str
    start: int = 0
    end: int = 0
    content_hash: str = ""
    score: float = 0.0


class ToolResult(_Frozen):
    tool_call_id: str
    content: str
    is_error: bool = False
    citations: tuple[Citation, ...] = ()


class ToolSpec(_Frozen):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema of the tool's arguments.",
    )
    source: Literal["mcp_server", "plugin", "builtin"] = "builtin"
    permissions_tag: str | None = None  # canonical 'server.tool' id for gating
    path_hints: tuple[str, ...] = ()  # argument names holding filesystem paths


class Message(_Frozen):
    role: Role
    content: tuple[ContentPart, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None  # tool role: which call this answers
    name: str | None = None  # tool role: the tool's name, when known
    is_error: bool = False  # tool role: result is an error

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.content if isinstance(part, TextPart))

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role="system", content=(TextPart(text=text),))

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=(TextPart(text=text),))

    @classmethod
    def assistant(cls, text: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        content = (TextPart(text=text),) if text else ()
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        content: str,
        *,
        name: str | None = None,
        is_error: bool = False,
    ) -> Message:
        return cls(
            role="tool",
            content=(TextPart(text=content),),
            tool_call_id=tool_call_id,
            name=name,
            is_error=is_error,
        )


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


class Usage(_Frozen):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )


class Cost(_Frozen):
    usd: Decimal = Decimal("0")
    known: bool = True  # False when no pricing entry existed for the model

    def __add__(self, other: Cost) -> Cost:
        return Cost(usd=self.usd + other.usd, known=self.known and other.known)
