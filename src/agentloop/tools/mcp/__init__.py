"""MCP client (Phase 4). The `mcp` SDK never escapes this package."""

from agentloop.tools.mcp.client import MCPClient, MCPServerConnection, connect

__all__ = ["MCPClient", "MCPServerConnection", "connect"]
