"""Deterministic replay (§12).

Recording wrappers sit at every non-deterministic boundary — model,
tools, embeddings, clock — implementing the same Protocols and emitting
`record.*` events carrying the FULL request and response. Replay stubs
answer from those records (per-boundary-kind FIFO — same determinism as
strict seq order, without false divergence when independent boundary
types interleave) and ASSERT the live request matches the recorded one:
a mismatch raises ReplayDivergence naming the boundary, both hashes, and
the first differing field. Divergence means the code changed behavior —
which is the point of replay.

Hooks are not recorded: they are deterministic functions of their
payloads and simply re-run. Streaming is not supported under recording
in v1 (the loop uses complete()).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from typing import Any

from agentloop.models.protocol import (
    CompletionRequest,
    CompletionResult,
    ModelProvider,
)
from agentloop.observability.events import TraceEvent
from agentloop.tools.executor import ToolExecutor
from agentloop.types import (
    AgentLoopError,
    EmbeddingProvider,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
    TraceEmitter,
)


class ReplayDivergence(AgentLoopError):
    """The live run stopped matching the recording."""


def stable_hash(data: Any) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def first_difference(recorded: Any, live: Any, path: str = "") -> str | None:
    """Human-pointable first divergence between two JSON-ish structures."""
    label = path or "<root>"
    if type(recorded) is not type(live):
        return f"{label}: recorded {recorded!r} vs live {live!r}"
    if isinstance(recorded, dict):
        for key in sorted(set(recorded) | set(live)):
            if key not in recorded:
                return f"{label}.{key}: absent in recording, live has {live[key]!r}"
            if key not in live:
                return f"{label}.{key}: recorded {recorded[key]!r}, absent live"
            found = first_difference(recorded[key], live[key], f"{label}.{key}")
            if found:
                return found
        return None
    if isinstance(recorded, list):
        if len(recorded) != len(live):
            return f"{label}: recorded {len(recorded)} items, live {len(live)}"
        for i, (a, b) in enumerate(zip(recorded, live, strict=True)):
            found = first_difference(a, b, f"{label}[{i}]")
            if found:
                return found
        return None
    if recorded != live:
        return f"{label}: recorded {recorded!r} vs live {live!r}"
    return None


# ---------------------------------------------------------------------------
# recording wrappers
# ---------------------------------------------------------------------------


class RecordingProvider:
    def __init__(self, inner: ModelProvider, emitter: TraceEmitter):
        self._inner = inner
        self._emitter = emitter

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        result = await self._inner.complete(request)
        dump = request.model_dump(mode="json")
        self._emitter.emit(
            "record.model",
            {
                "request_hash": stable_hash(dump),
                "request": dump,
                "result": result.model_dump(mode="json"),
            },
        )
        return result

    def stream(self, request: CompletionRequest):
        raise NotImplementedError("streaming is not recordable in v1")

    def tool_call_schema(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        return self._inner.tool_call_schema(tools)

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return self._inner.count_tokens(messages)


class RecordingExecutor:
    def __init__(self, inner: ToolExecutor, emitter: TraceEmitter):
        self._inner = inner
        self._emitter = emitter
        self._specs_recorded = False

    def specs(self) -> tuple[ToolSpec, ...]:
        specs = self._inner.specs()
        if not self._specs_recorded:
            self._specs_recorded = True
            self._emitter.emit(
                "record.specs",
                {"specs": [s.model_dump(mode="json") for s in specs]},
            )
        return specs

    async def execute(self, call: ToolCall) -> ToolResult:
        result = await self._inner.execute(call)
        dump = call.model_dump(mode="json")
        self._emitter.emit(
            "record.tool",
            {
                "call_hash": stable_hash(dump),
                "call": dump,
                "result": result.model_dump(mode="json"),
            },
        )
        return result


class RecordingEmbedder:
    def __init__(self, inner: EmbeddingProvider, emitter: TraceEmitter):
        self._inner = inner
        self._emitter = emitter

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = await self._inner.embed(texts)
        self._emitter.emit(
            "record.embed",
            {"texts_hash": stable_hash(list(texts)), "texts": list(texts),
             "vectors": vectors},
        )
        return vectors


class RecordingClock:
    def __init__(self, inner: Callable[[], float], emitter: TraceEmitter):
        self._inner = inner
        self._emitter = emitter

    def __call__(self) -> float:
        value = self._inner()
        self._emitter.emit("record.clock", {"value": value})
        return value


# ---------------------------------------------------------------------------
# replay stubs
# ---------------------------------------------------------------------------


class ReplayLog:
    """Per-kind FIFO cursor over a recording's record.* events."""

    def __init__(self, events: Sequence[TraceEvent]):
        self._queues: dict[str, deque[TraceEvent]] = defaultdict(deque)
        self._served: dict[str, int] = defaultdict(int)
        for event in events:
            if event.kind.startswith("record."):
                self._queues[event.kind].append(event)

    def take(self, kind: str, description: str) -> TraceEvent:
        self._served[kind] += 1
        queue = self._queues.get(kind)
        if not queue:
            raise ReplayDivergence(
                f"{description} #{self._served[kind]} has no counterpart in the "
                f"recording — the live code makes more {kind} calls than the "
                f"recorded run did"
            )
        return queue.popleft()

    def serial(self, kind: str) -> int:
        return self._served[kind]

    def remaining(self) -> dict[str, int]:
        return {kind: len(q) for kind, q in self._queues.items() if q}


class ReplayProvider:
    provider = "replay"
    model = "replay"

    def __init__(self, log: ReplayLog, emitter: TraceEmitter):
        self._log = log
        self._emitter = emitter

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        record = self._log.take("record.model", "model call")
        dump = request.model_dump(mode="json")
        if stable_hash(dump) != record.payload["request_hash"]:
            difference = first_difference(record.payload["request"], dump)
            raise ReplayDivergence(
                f"model call #{self._log.serial('record.model')} diverged from "
                f"the recording.\n"
                f"recorded hash {record.payload['request_hash']}, live hash "
                f"{stable_hash(dump)}\nfirst difference at {difference}"
            )
        self._emitter.emit("record.model", record.payload)  # streams stay aligned
        return CompletionResult.model_validate(record.payload["result"])

    def stream(self, request: CompletionRequest):
        raise NotImplementedError("streaming is not replayable in v1")

    def tool_call_schema(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        return []

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return 0


class ReplayExecutor:
    def __init__(self, log: ReplayLog, emitter: TraceEmitter):
        self._log = log
        self._emitter = emitter
        self._specs: tuple[ToolSpec, ...] | None = None
        self._specs_emitted = False

    def specs(self) -> tuple[ToolSpec, ...]:
        if self._specs is None:
            record = self._log.take("record.specs", "tool spec listing")
            self._specs = tuple(
                ToolSpec.model_validate(s) for s in record.payload["specs"]
            )
            self._payload = record.payload
        if not self._specs_emitted:
            self._specs_emitted = True
            self._emitter.emit("record.specs", self._payload)
        return self._specs

    async def execute(self, call: ToolCall) -> ToolResult:
        record = self._log.take("record.tool", "tool call")
        dump = call.model_dump(mode="json")
        if stable_hash(dump) != record.payload["call_hash"]:
            difference = first_difference(record.payload["call"], dump)
            raise ReplayDivergence(
                f"tool call #{self._log.serial('record.tool')} diverged from "
                f"the recording.\nfirst difference at {difference}"
            )
        self._emitter.emit("record.tool", record.payload)
        return ToolResult.model_validate(record.payload["result"])


class ReplayEmbedder:
    def __init__(self, log: ReplayLog, emitter: TraceEmitter):
        self._log = log
        self._emitter = emitter

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        record = self._log.take("record.embed", "embedding call")
        if stable_hash(list(texts)) != record.payload["texts_hash"]:
            difference = first_difference(record.payload["texts"], list(texts))
            raise ReplayDivergence(
                f"embedding call #{self._log.serial('record.embed')} diverged "
                f"from the recording.\nfirst difference at {difference}"
            )
        self._emitter.emit("record.embed", record.payload)
        return record.payload["vectors"]


class ReplayClock:
    def __init__(self, log: ReplayLog, emitter: TraceEmitter):
        self._log = log
        self._emitter = emitter

    def __call__(self) -> float:
        record = self._log.take("record.clock", "clock sample")
        self._emitter.emit("record.clock", record.payload)
        return float(record.payload["value"])
