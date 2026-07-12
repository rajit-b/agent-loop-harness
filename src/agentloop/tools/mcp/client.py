"""MCP client: server lifecycle, discovery, dispatch (§8, A2).

The official `mcp` SDK is confined to this package; the rest of the
framework sees only ToolSpec/ToolCall/ToolResult through the ToolExecutor
interface. Per connection:

- stdio: spawn with a scrubbed environment — the SDK's minimal safe set
  (PATH, HOME, …) plus ONLY the manifest-listed variables (A7);
- http: streamable-HTTP transport against the configured URL;
- discovery: tools/list → ToolSpecs named '{server}__{tool}' (wire-safe;
  providers reject dots), canonical '{server}.{tool}' in permissions_tag,
  path_hints from a well-known argument-name heuristic;
- dispatch: content blocks flatten to text; isError maps to is_error.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from agentloop.config.manifest import MCPServerConfig
from agentloop.types import AgentLoopError, ToolCall, ToolResult, ToolSpec

WIRE_SEPARATOR = "__"

# MCP schemas don't tag path arguments; this heuristic marks the usual
# suspects for the sandbox's path jail. Builtins can pass explicit hints.
PATH_ARG_NAMES = frozenset(
    {
        "path", "paths", "file", "files", "file_path", "filename",
        "dir", "directory", "directories", "root", "cwd",
        "source", "destination", "target",
    }
)


def detect_path_hints(input_schema: dict[str, Any]) -> tuple[str, ...]:
    properties = input_schema.get("properties", {})
    return tuple(name for name in properties if name in PATH_ARG_NAMES)


def _flatten_content(blocks: Sequence[Any]) -> str:
    texts = [b.text for b in blocks if getattr(b, "type", None) == "text"]
    return "\n".join(texts)


class MCPServerConnection:
    """One long-lived session to one MCP server."""

    def __init__(self, config: MCPServerConfig, *, init_timeout_s: float = 30.0):
        self.config = config
        self._init_timeout_s = init_timeout_s
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        try:
            if self.config.transport == "stdio":
                params = StdioServerParameters(
                    command=self.config.command or "",
                    args=list(self.config.args),
                    env={**get_default_environment(), **self.config.env},
                )
                read, write = await self._stack.enter_async_context(
                    stdio_client(params)
                )
            else:
                read, write, _ = await self._stack.enter_async_context(
                    streamablehttp_client(self.config.url or "")
                )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await asyncio.wait_for(self._session.initialize(), self._init_timeout_s)
        except BaseException:
            await self._stack.aclose()
            self._stack = None
            raise

    async def aclose(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise AgentLoopError(
                f"MCP server {self.config.name!r} is not connected"
            )
        return self._session

    async def discover(self) -> tuple[ToolSpec, ...]:
        listing = await self.session.list_tools()
        specs = []
        for tool in listing.tools:
            schema = tool.inputSchema or {"type": "object", "properties": {}}
            specs.append(
                ToolSpec(
                    name=f"{self.config.name}{WIRE_SEPARATOR}{tool.name}",
                    description=tool.description or "",
                    parameters=schema,
                    source="mcp_server",
                    permissions_tag=f"{self.config.name}.{tool.name}",
                    path_hints=detect_path_hints(schema),
                )
            )
        return tuple(specs)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        result = await self.session.call_tool(tool_name, arguments=arguments)
        return _flatten_content(result.content), bool(result.isError)


class MCPClient:
    """ToolExecutor over any number of MCP servers. Start before use:

        async with MCPClient(configs) as client: ...
    """

    def __init__(self, servers: Sequence[MCPServerConfig]):
        self._connections = {cfg.name: MCPServerConnection(cfg) for cfg in servers}
        self._specs: dict[str, ToolSpec] = {}  # wire name -> spec

    async def start(self) -> None:
        for connection in self._connections.values():
            await connection.start()
            for spec in await connection.discover():
                self._specs[spec.name] = spec

    async def aclose(self) -> None:
        for connection in reversed(list(self._connections.values())):
            await connection.aclose()

    async def __aenter__(self) -> MCPClient:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs.values())

    async def execute(self, call: ToolCall) -> ToolResult:
        server_name, _, tool_name = call.name.partition(WIRE_SEPARATOR)
        connection = self._connections.get(server_name)
        if not tool_name or connection is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"unknown MCP tool: {call.name!r}",
                is_error=True,
            )
        try:
            content, is_error = await connection.call(tool_name, call.arguments)
        except Exception as exc:  # noqa: BLE001 — transport failures feed the model
            return ToolResult(
                tool_call_id=call.id,
                content=f"MCP call {call.name!r} failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResult(tool_call_id=call.id, content=content, is_error=is_error)


@asynccontextmanager
async def connect(servers: Sequence[MCPServerConfig]) -> AsyncIterator[MCPClient]:
    async with MCPClient(servers) as client:
        yield client
