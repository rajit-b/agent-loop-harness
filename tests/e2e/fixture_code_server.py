"""Fixture MCP server standing in for filesystem + ripgrep tools, so the
worked example runs offline. Exposes `code.search` and `code.read_file`.
`search` echoes the query it received — that echo is how the end-to-end
test proves the secret-redaction hook scrubbed arguments before execution."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("code")


@server.tool()
def search(query: str) -> str:
    """Search the repository for a regex or symbol."""
    return (
        "1 match:\n"
        "src/app/config/loader.py:12: def load_config(path)\n"
        f"[server received query={query!r}]"
    )


@server.tool()
def read_file(path: str) -> str:
    """Read a source file from the repository."""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


if __name__ == "__main__":
    server.run()
