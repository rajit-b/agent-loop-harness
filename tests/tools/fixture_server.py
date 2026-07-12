"""A real MCP stdio server used as a subprocess fixture by test_mcp.py."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("testsrv")


@server.tool()
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


@server.tool()
def read_file(path: str) -> str:
    """Read a text file and return its contents."""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


@server.tool()
def getenv(name: str) -> str:
    """Return the value of an environment variable (empty if unset)."""
    return os.environ.get(name, "")


@server.tool()
def explode() -> str:
    """Always fails."""
    raise ValueError("deliberate failure")


if __name__ == "__main__":
    server.run()  # stdio transport
