"""The Phase 3 stub tool: proves the dispatch path end to end."""

from __future__ import annotations

from typing import Any

from agentloop.tools.executor import ToolRegistry
from agentloop.types import ToolSpec

ECHO_SPEC = ToolSpec(
    name="echo",
    description="Echo the provided text back verbatim.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    source="builtin",
)


async def echo(arguments: dict[str, Any]) -> str:
    return str(arguments.get("text", ""))


def register_echo(registry: ToolRegistry) -> None:
    registry.register(ECHO_SPEC, echo)
