from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentloop.models.protocol import CompletionRequest
from agentloop.types import Message, ToolCall, ToolSpec

GOLDEN_DIR = Path(__file__).parent / "golden"


def load_golden(name: str) -> Any:
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def convo() -> tuple[Message, ...]:
    """One exchange exercising every role, incl. parallel tool calls."""
    return (
        Message.system("You answer questions about code."),
        Message.user("Where is the config loader?"),
        Message.assistant(
            "I'll search.",
            tool_calls=(
                ToolCall(id="call_0", name="grep", arguments={"pattern": "load_config"}),
                ToolCall(id="call_1", name="ls", arguments={"path": "src"}),
            ),
        ),
        Message.tool_result("call_0", "src/agentloop/config/loader.py"),
        Message.tool_result("call_1", "config/  models/", name="ls"),
    )


@pytest.fixture
def tools() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="grep",
            description="Search files for a pattern",
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        ),
    )


@pytest.fixture
def request_fx(convo, tools) -> CompletionRequest:
    return CompletionRequest(messages=convo, tools=tools)
