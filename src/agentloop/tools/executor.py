"""ToolExecutor interface, registry, and the gated dispatch path (§8).

The loop has exactly one dispatch path: ToolGateway fronts every backend
(builtin ToolRegistry, MCPClient) and applies, in order: allowlist gating
(fail closed), path-jail validation, per-call timeout, result size cap.
Contract everywhere: tool-level failures come back as error ToolResults
the model can react to — never exceptions into the loop. Only
cancellation propagates.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from agentloop.tools.permissions import canonical_name, is_allowed
from agentloop.tools.sandbox import Sandbox
from agentloop.types import (
    NullEmitter,
    ToolCall,
    ToolResult,
    ToolSpec,
    TraceEmitter,
)


@runtime_checkable
class ToolExecutor(Protocol):
    def specs(self) -> tuple[ToolSpec, ...]: ...

    async def execute(self, call: ToolCall) -> ToolResult: ...


ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


class ToolRegistry:
    """In-process ToolExecutor for builtin (and later plugin-registered) tools."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = (spec, handler)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(spec for spec, _ in self._tools.values())

    async def execute(self, call: ToolCall) -> ToolResult:
        entry = self._tools.get(call.name)
        if entry is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"unknown tool: {call.name!r}",
                is_error=True,
            )
        _, handler = entry
        try:
            content = await handler(call.arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — tool failures feed the model
            return ToolResult(
                tool_call_id=call.id,
                content=f"tool {call.name!r} failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResult(tool_call_id=call.id, content=content)


class ToolGateway:
    """The single, gated dispatch path the loop sees.

    specs() exposes only allowlisted tools (the model never learns about
    tools it cannot call); execute() re-checks anyway — a hallucinated or
    denied name yields a synthetic error result plus a `tool.denied`
    trace event, never an execution.
    """

    def __init__(
        self,
        executors: Sequence[ToolExecutor],
        *,
        allowlist: Sequence[str],
        sandbox: Sandbox | None = None,
        tool_timeout_s: float = 30.0,
        emitter: TraceEmitter | None = None,
    ):
        self._allowlist = list(allowlist)
        self._sandbox = sandbox or Sandbox()
        self._timeout = tool_timeout_s
        self._emitter = emitter or NullEmitter()
        self._routes: dict[str, tuple[ToolSpec, ToolExecutor]] = {}
        for executor in executors:
            for spec in executor.specs():
                if spec.name in self._routes:
                    raise ValueError(f"tool name collision: {spec.name!r}")
                self._routes[spec.name] = (spec, executor)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(
            spec
            for spec, _ in self._routes.values()
            if is_allowed(spec, self._allowlist)
        )

    async def execute(self, call: ToolCall) -> ToolResult:
        route = self._routes.get(call.name)
        if route is None or not is_allowed(route[0], self._allowlist):
            reason = (
                "unknown tool"
                if route is None
                else f"denied by policy: {canonical_name(route[0])!r} is not allowlisted"
            )
            self._emitter.emit(
                "tool.denied", {"id": call.id, "name": call.name, "reason": reason}
            )
            return ToolResult(tool_call_id=call.id, content=reason, is_error=True)

        spec, executor = route
        violation = self._sandbox.check_paths(spec, call)
        if violation is not None:
            self._emitter.emit(
                "tool.denied", {"id": call.id, "name": call.name, "reason": violation}
            )
            return ToolResult(tool_call_id=call.id, content=violation, is_error=True)

        try:
            result = await asyncio.wait_for(executor.execute(call), self._timeout)
        except TimeoutError:
            return ToolResult(
                tool_call_id=call.id,
                content=f"tool {call.name!r} timed out after {self._timeout}s",
                is_error=True,
            )
        return ToolResult(
            tool_call_id=result.tool_call_id,
            content=self._sandbox.cap_result(result.content),
            is_error=result.is_error,
            citations=result.citations,
        )
