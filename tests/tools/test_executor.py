"""ToolRegistry: dispatch contract — failures return, never raise."""

from __future__ import annotations

import pytest

from agentloop.tools.builtin.echo import ECHO_SPEC, register_echo
from agentloop.tools.executor import ToolExecutor, ToolRegistry
from agentloop.types import ToolCall, ToolSpec


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_echo(reg)
    return reg


class TestRegistry:
    def test_satisfies_protocol(self, registry):
        assert isinstance(registry, ToolExecutor)

    def test_specs(self, registry):
        assert registry.specs() == (ECHO_SPEC,)

    def test_duplicate_registration_rejected(self, registry):
        with pytest.raises(ValueError, match="already registered"):
            register_echo(registry)

    async def test_echo_executes(self, registry):
        result = await registry.execute(
            ToolCall(id="c1", name="echo", arguments={"text": "hi"})
        )
        assert result.content == "hi"
        assert result.is_error is False
        assert result.tool_call_id == "c1"

    async def test_unknown_tool_is_error_result(self, registry):
        result = await registry.execute(ToolCall(id="c1", name="nope"))
        assert result.is_error is True
        assert "unknown tool" in result.content

    async def test_handler_exception_is_error_result(self):
        registry = ToolRegistry()

        async def boom(arguments):
            raise RuntimeError("kaput")

        registry.register(ToolSpec(name="boom"), boom)
        result = await registry.execute(ToolCall(id="c1", name="boom"))
        assert result.is_error is True
        assert "RuntimeError" in result.content and "kaput" in result.content
