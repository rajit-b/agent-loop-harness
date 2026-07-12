"""Integration: a real MCP stdio server (subprocess) through the full stack.

Spawns tests/tools/fixture_server.py via the stdio transport — real
discovery, real dispatch, real env scrubbing — and finally wires it into
Phase 3's AgentLoop.
"""

from __future__ import annotations

import sys
from pathlib import Path


from agentloop.config.manifest import MCPServerConfig
from agentloop.loop.machine import AgentLoop
from agentloop.tools.executor import ToolGateway
from agentloop.tools.mcp import MCPClient
from agentloop.types import ToolCall

from ..loop.conftest import RecordingEmitter, ScriptedProvider, result

FIXTURE = Path(__file__).parent / "fixture_server.py"


def server_config(**env: str) -> MCPServerConfig:
    return MCPServerConfig(
        name="testsrv",
        transport="stdio",
        command=sys.executable,
        args=(str(FIXTURE),),
        env=env,
    )


class TestDiscoveryAndDispatch:
    async def test_discovery_yields_namespaced_specs(self):
        async with MCPClient([server_config()]) as client:
            specs = {s.name: s for s in client.specs()}
            assert set(specs) == {
                "testsrv__add",
                "testsrv__read_file",
                "testsrv__getenv",
                "testsrv__explode",
            }
            add = specs["testsrv__add"]
            assert add.source == "mcp_server"
            assert add.permissions_tag == "testsrv.add"
            assert add.description == "Add two integers."
            assert set(add.parameters["properties"]) == {"a", "b"}
            # path-hint heuristic tagged read_file's path argument
            assert specs["testsrv__read_file"].path_hints == ("path",)
            assert specs["testsrv__add"].path_hints == ()

    async def test_call_round_trip(self):
        async with MCPClient([server_config()]) as client:
            result_ = await client.execute(
                ToolCall(id="c1", name="testsrv__add", arguments={"a": 1, "b": 2})
            )
            assert result_.content == "3"
            assert result_.is_error is False

    async def test_server_side_tool_error_maps_to_is_error(self):
        async with MCPClient([server_config()]) as client:
            result_ = await client.execute(
                ToolCall(id="c1", name="testsrv__explode", arguments={})
            )
            assert result_.is_error is True
            assert "deliberate failure" in result_.content

    async def test_env_is_scrubbed_except_manifest_listed(self, monkeypatch):
        monkeypatch.setenv("AGENTLOOP_TEST_SECRET", "leaked!")
        async with MCPClient([server_config(ALLOWED_VAR="visible")]) as client:
            secret = await client.execute(
                ToolCall(id="c1", name="testsrv__getenv",
                         arguments={"name": "AGENTLOOP_TEST_SECRET"})
            )
            allowed = await client.execute(
                ToolCall(id="c2", name="testsrv__getenv",
                         arguments={"name": "ALLOWED_VAR"})
            )
        assert secret.content == ""  # parent env did NOT pass through
        assert allowed.content == "visible"  # manifest-listed env did


class TestLoopIntegration:
    async def test_real_mcp_tool_through_the_agent_loop(self, tmp_path):
        (tmp_path / "notes.txt").write_text("the loader lives in config/")
        emitter = RecordingEmitter()
        provider = ScriptedProvider(
            [
                result(
                    tool_calls=(
                        ToolCall(
                            id="c1",
                            name="testsrv__read_file",
                            arguments={"path": str(tmp_path / "notes.txt")},
                        ),
                    )
                ),
                result("It lives in config/."),
            ]
        )
        async with MCPClient([server_config()]) as client:
            from agentloop.tools.sandbox import Sandbox

            gateway = ToolGateway(
                [client],
                allowlist=["testsrv.*"],
                sandbox=Sandbox(roots=[tmp_path]),
                emitter=emitter,
            )
            loop = AgentLoop(
                provider, gateway, intent="Answer questions.", emitter=emitter
            )
            turn = await loop.run_turn("where is the loader?")

        assert turn.status == "completed"
        assert turn.text == "It lives in config/."
        # the real file contents flowed through MCP into the model request
        tool_messages = [m for m in provider.requests[1].messages if m.role == "tool"]
        assert tool_messages[0].text == "the loader lives in config/"
        # gating was live: the model only saw allowlisted, namespaced tools
        first_tools = {t.name for t in provider.requests[0].tools}
        assert "testsrv__read_file" in first_tools

    async def test_jail_blocks_escape_through_full_stack(self, tmp_path):
        (tmp_path / "jail").mkdir()
        emitter = RecordingEmitter()
        async with MCPClient([server_config()]) as client:
            from agentloop.tools.sandbox import Sandbox

            gateway = ToolGateway(
                [client],
                allowlist=["testsrv.*"],
                sandbox=Sandbox(roots=[tmp_path / "jail"]),
                emitter=emitter,
            )
            result_ = await gateway.execute(
                ToolCall(id="c1", name="testsrv__read_file",
                         arguments={"path": "/etc/passwd"})
            )
        assert result_.is_error is True and "escapes" in result_.content
        assert any(k == "tool.denied" for k, _ in emitter.events)
