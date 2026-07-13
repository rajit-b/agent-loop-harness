"""Plugin loading: resolve → validate → register → commit; fail closed.

Source resolution: a `source` that exists on disk is a local plugin (a
directory containing plugin.py, or a .py file) whose module exposes the
`agentloop_plugin(config)` factory; anything else is an entry-point name
in group 'agentloop.plugin'.

An incompatible or failing plugin is skipped with a `plugin.failed`
trace event and never partially loaded — registration is staged and only
committed after register() returns (see registrar.py).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from agentloop.config.manifest import PluginConfig
from agentloop.hooks.bus import HookBus
from agentloop.plugins.contract import (
    ENTRY_POINT_GROUP,
    FACTORY_ATTR,
    PLUGIN_API_VERSION,
    CliCommand,
    Plugin,
    PluginFactory,
)
from agentloop.plugins.registrar import Registrar
from agentloop.skills.loader import Skill
from agentloop.tools.executor import ToolRegistry
from agentloop.types import NullEmitter, PluginError, TraceEmitter


@dataclass(slots=True)
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (name, error)


class PluginManager:
    def __init__(
        self,
        *,
        tools: ToolRegistry,
        hooks: HookBus,
        skills: list[Skill] | None = None,
        cli: dict[str, CliCommand] | None = None,
        emitter: TraceEmitter | None = None,
    ):
        self.tools = tools
        self.hooks = hooks
        self.skills: list[Skill] = skills if skills is not None else []
        self.cli: dict[str, CliCommand] = cli if cli is not None else {}
        self._emitter = emitter or NullEmitter()
        self._loaded: list[Plugin] = []

    def load_all(self, configs: Sequence[PluginConfig]) -> LoadReport:
        report = LoadReport()
        for config in configs:
            try:
                plugin = self.load_one(config)
            except PluginError as exc:
                self._emitter.emit(
                    "plugin.failed", {"name": config.name, "error": str(exc)}
                )
                report.failed.append((config.name, str(exc)))
                continue
            report.loaded.append(plugin.name)
        return report

    def load_one(self, config: PluginConfig) -> Plugin:
        factory = _resolve_factory(config)
        try:
            plugin = factory(dict(config.config))
        except Exception as exc:  # noqa: BLE001 — a broken factory fails closed
            raise PluginError(
                f"plugin {config.name!r} factory raised: {type(exc).__name__}: {exc}"
            ) from exc
        _validate(config, plugin)

        registrar = Registrar(
            plugin.name,
            tools=self.tools,
            hooks=self.hooks,
            skills=self.skills,
            cli=self.cli,
        )
        try:
            plugin.register(registrar)
        except PluginError:
            raise
        except Exception as exc:  # noqa: BLE001
            # nothing was committed: the staged registrations die here
            raise PluginError(
                f"plugin {config.name!r} failed during register(): "
                f"{type(exc).__name__}: {exc} (rolled back)"
            ) from exc
        counts = registrar.commit()
        self._loaded.append(plugin)
        self._emitter.emit(
            "plugin.loaded", {"name": plugin.name, "version": plugin.version, **counts}
        )
        return plugin

    def dispose_all(self) -> None:
        for plugin in reversed(self._loaded):
            try:
                plugin.dispose()
            except Exception as exc:  # noqa: BLE001 — dispose failures are isolated
                self._emitter.emit(
                    "plugin.failed",
                    {"name": plugin.name, "error": f"dispose: {exc}"},
                )
            else:
                self._emitter.emit("plugin.disposed", {"name": plugin.name})
        self._loaded.clear()


# ---------------------------------------------------------------------------


def _resolve_factory(config: PluginConfig) -> PluginFactory:
    path = Path(config.source)
    if path.exists():
        return _factory_from_path(config, path)
    return _factory_from_entry_point(config)


def _factory_from_path(config: PluginConfig, path: Path) -> PluginFactory:
    module_file = path / "plugin.py" if path.is_dir() else path
    if not module_file.is_file() or module_file.suffix != ".py":
        raise PluginError(
            f"plugin {config.name!r}: source {config.source!r} has no plugin.py"
        )
    digest = hashlib.sha1(str(module_file.resolve()).encode()).hexdigest()[:8]
    module_name = f"_agentloop_plugin_{config.name}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise PluginError(f"plugin {config.name!r}: cannot load {module_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise PluginError(
            f"plugin {config.name!r}: import failed: {type(exc).__name__}: {exc}"
        ) from exc
    factory = getattr(module, FACTORY_ATTR, None)
    if not callable(factory):
        raise PluginError(
            f"plugin {config.name!r}: {module_file} does not expose a callable "
            f"{FACTORY_ATTR!r}"
        )
    return factory


def _factory_from_entry_point(config: PluginConfig) -> PluginFactory:
    matches = [
        ep
        for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        if ep.name == config.source
    ]
    if not matches:
        raise PluginError(
            f"plugin {config.name!r}: source {config.source!r} is neither a "
            f"path nor an {ENTRY_POINT_GROUP!r} entry point"
        )
    try:
        factory = matches[0].load()
    except Exception as exc:  # noqa: BLE001
        raise PluginError(
            f"plugin {config.name!r}: entry point failed to load: {exc}"
        ) from exc
    if not callable(factory):
        raise PluginError(f"plugin {config.name!r}: entry point is not callable")
    return factory


def _validate(config: PluginConfig, plugin: Any) -> None:
    for attribute in ("name", "version", "api_version"):
        if not hasattr(plugin, attribute):
            raise PluginError(
                f"plugin {config.name!r} is missing attribute {attribute!r}"
            )
    if not callable(getattr(plugin, "register", None)):
        raise PluginError(f"plugin {config.name!r} has no register() method")
    if not callable(getattr(plugin, "dispose", None)):
        raise PluginError(f"plugin {config.name!r} has no dispose() method")
    if plugin.api_version != PLUGIN_API_VERSION:
        raise PluginError(
            f"plugin {config.name!r} targets api_version {plugin.api_version}, "
            f"this framework provides {PLUGIN_API_VERSION}"
        )
    if config.version is not None:
        try:
            specifier = SpecifierSet(config.version)
            actual = Version(plugin.version)
        except (InvalidSpecifier, InvalidVersion) as exc:
            raise PluginError(
                f"plugin {config.name!r}: bad version data: {exc}"
            ) from exc
        if actual not in specifier:
            raise PluginError(
                f"plugin {config.name!r} version {plugin.version} does not "
                f"satisfy the manifest constraint {config.version!r}"
            )
