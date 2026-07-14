"""Worked-example plugin: registers a Jira issue-lookup tool.

A plugin's only power is registration (§10). This one adds a single tool;
its config carries the Jira base URL. In a real deployment the handler would
call the Jira REST API — here it returns a deterministic stub so the example
runs offline.
"""

from __future__ import annotations

from typing import Any

from agentloop.types import ToolSpec

PLUGIN_API_VERSION = 1


class JiraPlugin:
    name = "jira"
    version = "0.1.0"
    api_version = PLUGIN_API_VERSION

    def __init__(self, config: dict[str, Any]):
        self.base_url = config.get("base_url", "https://example.atlassian.net")

    def register(self, registrar: Any) -> None:
        async def issue_lookup(arguments: dict[str, Any]) -> str:
            key = str(arguments.get("key", "")).strip()
            if not key:
                raise ValueError("issue_lookup requires a 'key' argument")
            # stub: a real handler would GET {base_url}/rest/api/2/issue/{key}
            return (
                f"{key} [Open] Fix config loader precedence — "
                f"assignee: unassigned ({self.base_url})"
            )

        registrar.add_tool(
            ToolSpec(
                name="issue_lookup",
                description="Look up a Jira issue by key (e.g. PROJ-123).",
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            ),
            issue_lookup,
        )

    def dispose(self) -> None:
        return None


def agentloop_plugin(config: dict[str, Any]) -> JiraPlugin:
    return JiraPlugin(config)
