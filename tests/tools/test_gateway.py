"""ToolGateway: gating, jail, timeout, truncation — the one dispatch path."""

from __future__ import annotations

import asyncio

import pytest

from agentloop.tools.builtin.echo import register_echo
from agentloop.tools.executor import ToolGateway, ToolRegistry
from agentloop.tools.sandbox import Sandbox
from agentloop.types import ToolCall, ToolSpec


class RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind, payload) -> None:
        self.events.append((kind, dict(payload)))


@pytest.fixture
def emitter() -> RecordingEmitter:
    return RecordingEmitter()


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_echo(registry)

    async def read_file(arguments):
        return f"contents of {arguments['path']}"

    registry.register(
        ToolSpec(
            name="fs__read_file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            source="mcp_server",
            permissions_tag="fs.read_file",
            path_hints=("path",),
        ),
        read_file,
    )

    async def hang(arguments):
        await asyncio.Event().wait()
        return "never"

    registry.register(ToolSpec(name="hang"), hang)

    async def firehose(arguments):
        return "y" * 1_000_000

    registry.register(ToolSpec(name="firehose"), firehose)
    return registry


class TestGating:
    async def test_allowed_tool_executes(self, emitter):
        gateway = ToolGateway(
            [make_registry()], allowlist=["echo"], emitter=emitter
        )
        result = await gateway.execute(
            ToolCall(id="c1", name="echo", arguments={"text": "hi"})
        )
        assert result.content == "hi" and not result.is_error

    async def test_denied_tool_yields_error_and_event(self, emitter):
        gateway = ToolGateway(
            [make_registry()], allowlist=["echo"], emitter=emitter
        )
        result = await gateway.execute(
            ToolCall(id="c1", name="fs__read_file", arguments={"path": "x"})
        )
        assert result.is_error is True
        assert "denied by policy" in result.content
        assert "fs.read_file" in result.content
        kinds = [k for k, _ in emitter.events]
        assert kinds == ["tool.denied"]

    async def test_specs_expose_only_allowlisted_tools(self, emitter):
        gateway = ToolGateway(
            [make_registry()], allowlist=["fs.read_*"], emitter=emitter
        )
        assert [s.name for s in gateway.specs()] == ["fs__read_file"]

    async def test_empty_allowlist_denies_everything(self, emitter):
        gateway = ToolGateway([make_registry()], allowlist=[], emitter=emitter)
        assert gateway.specs() == ()
        result = await gateway.execute(ToolCall(id="c1", name="echo"))
        assert result.is_error is True

    async def test_unknown_tool_fails_closed(self, emitter):
        gateway = ToolGateway([make_registry()], allowlist=["*"], emitter=emitter)
        result = await gateway.execute(ToolCall(id="c1", name="hallucinated"))
        assert result.is_error is True and "unknown tool" in result.content

    async def test_name_collision_rejected_at_construction(self):
        a, b = ToolRegistry(), ToolRegistry()
        register_echo(a)
        register_echo(b)
        with pytest.raises(ValueError, match="collision"):
            ToolGateway([a, b], allowlist=["*"])


class TestSandboxIntegration:
    async def test_jail_violation_via_gateway(self, tmp_path, emitter):
        (tmp_path / "jail").mkdir()
        gateway = ToolGateway(
            [make_registry()],
            allowlist=["fs.read_*"],
            sandbox=Sandbox(roots=[tmp_path / "jail"]),
            emitter=emitter,
        )
        result = await gateway.execute(
            ToolCall(id="c1", name="fs__read_file", arguments={"path": "/etc/passwd"})
        )
        assert result.is_error is True and "escapes" in result.content
        assert emitter.events[0][0] == "tool.denied"

    async def test_timeout_kills_hung_tool(self, emitter):
        gateway = ToolGateway(
            [make_registry()], allowlist=["hang"], tool_timeout_s=0.05,
            emitter=emitter,
        )
        result = await gateway.execute(ToolCall(id="c1", name="hang"))
        assert result.is_error is True and "timed out" in result.content

    async def test_oversized_result_is_capped(self, emitter):
        gateway = ToolGateway(
            [make_registry()],
            allowlist=["firehose"],
            sandbox=Sandbox(max_result_chars=1000),
            emitter=emitter,
        )
        result = await gateway.execute(ToolCall(id="c1", name="firehose"))
        assert len(result.content) < 1100
        assert "truncated" in result.content
