"""Allowlist gating: dotted globs, wire-name normalization, fail closed."""

from __future__ import annotations

from agentloop.tools.permissions import canonical_name, is_allowed
from agentloop.types import ToolSpec


def mcp_spec(server: str, tool: str) -> ToolSpec:
    return ToolSpec(
        name=f"{server}__{tool}",
        source="mcp_server",
        permissions_tag=f"{server}.{tool}",
    )


class TestCanonicalName:
    def test_mcp_spec_uses_permissions_tag(self):
        assert canonical_name(mcp_spec("fs", "read_file")) == "fs.read_file"

    def test_builtin_uses_bare_name(self):
        assert canonical_name(ToolSpec(name="echo")) == "echo"


class TestIsAllowed:
    def test_exact_match(self):
        assert is_allowed(mcp_spec("jira", "issue_lookup"), ["jira.issue_lookup"])

    def test_glob_match(self):
        spec = mcp_spec("fs", "read_file")
        assert is_allowed(spec, ["fs.read_*"])
        assert not is_allowed(spec, ["fs.write_*"])

    def test_server_wildcard(self):
        assert is_allowed(mcp_spec("ripgrep", "search"), ["ripgrep.*"])

    def test_empty_allowlist_denies_all(self):
        assert not is_allowed(mcp_spec("fs", "read_file"), [])
        assert not is_allowed(ToolSpec(name="echo"), [])

    def test_builtin_bare_name(self):
        assert is_allowed(ToolSpec(name="echo"), ["echo"])
        assert not is_allowed(ToolSpec(name="echo"), ["fs.*"])

    def test_dotted_glob_does_not_match_wire_name_accidentally(self):
        # the wire name has '__'; only the canonical tag is matched
        spec = mcp_spec("fs", "read_file")
        assert not is_allowed(spec, ["fs__read_file"])
