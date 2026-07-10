"""Message helpers and token estimation."""

from __future__ import annotations

import pydantic
import pytest

from agentloop.models.protocol import estimate_tokens
from agentloop.types import Message, ToolCall


class TestMessage:
    def test_helpers_and_text_property(self):
        assert Message.system("s").role == "system"
        assert Message.user("hello").text == "hello"
        message = Message.assistant(
            "thinking", tool_calls=(ToolCall(id="c", name="t"),)
        )
        assert message.text == "thinking"
        assert message.tool_calls[0].name == "t"

    def test_tool_result_carries_pairing_and_error(self):
        message = Message.tool_result("call_3", "oops", is_error=True)
        assert message.role == "tool"
        assert message.tool_call_id == "call_3"
        assert message.is_error is True

    def test_messages_are_frozen(self):
        with pytest.raises(pydantic.ValidationError):
            Message.user("x").role = "assistant"  # type: ignore[misc]


def test_estimate_tokens_scales_with_content():
    short = [Message.user("hi")]
    long = [Message.user("hi " * 500)]
    assert estimate_tokens(long) > estimate_tokens(short) > 0
