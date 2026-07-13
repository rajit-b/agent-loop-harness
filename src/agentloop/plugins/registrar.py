"""The registrar: a plugin's entire capability surface (§10).

Stage-then-commit: add_* calls validate eagerly (collisions against live
targets AND the staged set) but mutate nothing. The manager commits only
after the plugin's register() returns — so a plugin that throws mid-way
has, by construction, registered nothing. Fails closed without any
unregister machinery.

Plugin tools are namespaced exactly like MCP tools: wire name
'{plugin}__{tool}', canonical '{plugin}.{tool}' in permissions_tag — the
manifest allowlist speaks one dotted dialect for both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentloop.hooks.bus import HookBus
from agentloop.hooks.contract import HookHandler
from agentloop.config.manifest import HookEvent
from agentloop.plugins.contract import CliCommand
from agentloop.skills.loader import Skill, load_skill
from agentloop.tools.executor import ToolHandler, ToolRegistry
from agentloop.types import PluginError, ToolSpec


class Registrar:
    def __init__(
        self,
        plugin_name: str,
        *,
        tools: ToolRegistry,
        hooks: HookBus,
        skills: list[Skill],
        cli: dict[str, CliCommand],
    ):
        self._plugin = plugin_name
        self._tools = tools
        self._hooks = hooks
        self._skills = skills
        self._cli = cli
        self._staged_tools: list[tuple[ToolSpec, ToolHandler]] = []
        self._staged_hooks: list[dict[str, Any]] = []
        self._staged_skills: list[Skill] = []
        self._staged_cli: list[CliCommand] = []

    # ------------------------------------------------------------------
    # the four capabilities
    # ------------------------------------------------------------------

    def add_tool(self, spec: ToolSpec, handler: ToolHandler) -> None:
        prefix = f"{self._plugin}__"
        bare = spec.name.removeprefix(prefix)
        namespaced = spec.model_copy(
            update={
                "name": f"{prefix}{bare}",
                "permissions_tag": f"{self._plugin}.{bare}",
                "source": "plugin",
            }
        )
        taken = {s.name for s in self._tools.specs()}
        taken |= {s.name for s, _ in self._staged_tools}
        if namespaced.name in taken:
            raise PluginError(
                f"plugin {self._plugin!r}: tool {namespaced.name!r} already exists"
            )
        self._staged_tools.append((namespaced, handler))

    def add_skill(self, skill: Skill | Path | str) -> None:
        if not isinstance(skill, Skill):
            skill = load_skill(skill)
        staged_names = {s.name for s in self._staged_skills}
        existing_names = {s.name for s in self._skills}
        if skill.name in staged_names | existing_names:
            raise PluginError(
                f"plugin {self._plugin!r}: skill {skill.name!r} already exists"
            )
        self._staged_skills.append(skill)

    def add_hook(
        self,
        event: HookEvent,
        handler: HookHandler,
        *,
        priority: int = 100,
        config: dict[str, Any] | None = None,
        blocking: bool = False,
    ) -> None:
        self._staged_hooks.append(
            {
                "event": event,
                "handler": handler,
                "priority": priority,
                "name": f"{self._plugin}:{getattr(handler, '__qualname__', 'hook')}",
                "config": config or {},
                "blocking": blocking,
            }
        )

    def add_cli_command(self, name: str, handler, *, help: str = "") -> None:
        if name in self._cli or any(c.name == name for c in self._staged_cli):
            raise PluginError(
                f"plugin {self._plugin!r}: CLI command {name!r} already exists"
            )
        self._staged_cli.append(
            CliCommand(name=name, handler=handler, help=help, plugin=self._plugin)
        )

    # ------------------------------------------------------------------

    def commit(self) -> dict[str, int]:
        """Apply everything staged. Called by the manager, never plugins."""
        for spec, handler in self._staged_tools:
            self._tools.register(spec, handler)
        for hook in self._staged_hooks:
            self._hooks.register(
                hook["event"],
                hook["handler"],
                priority=hook["priority"],
                name=hook["name"],
                config=hook["config"],
                blocking=hook["blocking"],
            )
        self._skills.extend(self._staged_skills)
        for command in self._staged_cli:
            self._cli[command.name] = command
        return {
            "tools": len(self._staged_tools),
            "hooks": len(self._staged_hooks),
            "skills": len(self._staged_skills),
            "cli_commands": len(self._staged_cli),
        }
