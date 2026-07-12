"""ToolExecutor interface and the in-process registry (Phase 3 slice).

The loop has exactly one dispatch path: builtins here, MCP-backed tools in
Phase 4, both behind ToolExecutor. Contract: execute() never raises for
tool-level failures — unknown names and handler exceptions come back as
error ToolResults the model can react to. Only cancellation propagates.

Permission gating and sandboxing land in Phase 4.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from agentloop.types import ToolCall, ToolResult, ToolSpec


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
