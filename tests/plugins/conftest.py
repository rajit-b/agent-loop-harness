from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agentloop.config.manifest import PluginConfig
from agentloop.hooks.bus import HookBus
from agentloop.plugins.loader import PluginManager
from agentloop.tools.executor import ToolRegistry

from ..loop.conftest import RecordingEmitter

#: A well-behaved plugin registering one of everything (tool, hook, CLI).
GOOD_PLUGIN = """
class JiraPlugin:
    name = "jira"
    version = "0.1.5"
    api_version = 1

    def __init__(self, config):
        self.config = config
        self.disposed = False

    def register(self, registrar):
        from agentloop.types import ToolSpec

        async def issue_lookup(arguments):
            return f"JIRA-{arguments.get('key', '?')}: fake issue from " \\
                   f"{self.config.get('base_url', 'nowhere')}"

        registrar.add_tool(
            ToolSpec(
                name="issue_lookup",
                description="Look up a Jira issue by key.",
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            ),
            issue_lookup,
        )

        def audit(payload, ctx):
            return None  # observe-only hook

        registrar.add_hook("pre_tool", audit, priority=5)
        registrar.add_cli_command(
            "jira-status", lambda: "ok", help="Show Jira connectivity."
        )

    def dispose(self):
        self.disposed = True


_instances = []


def agentloop_plugin(config):
    plugin = JiraPlugin(config)
    _instances.append(plugin)
    return plugin
"""


def write_plugin(parent: Path, name: str, body: str) -> Path:
    directory = parent / name
    directory.mkdir(parents=True)
    (directory / "plugin.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return directory


def plugin_config(name: str, source: Path | str, **kwargs) -> PluginConfig:
    return PluginConfig(name=name, source=str(source), **kwargs)


@pytest.fixture
def emitter() -> RecordingEmitter:
    return RecordingEmitter()


@pytest.fixture
def manager(emitter) -> PluginManager:
    return PluginManager(tools=ToolRegistry(), hooks=HookBus(), emitter=emitter)
