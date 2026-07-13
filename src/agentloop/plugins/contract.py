"""Plugin packaging contract (§10).

A plugin ships either as an installed distribution exposing an entry
point in group 'agentloop.plugin', or as a local directory/file whose
module exposes an `agentloop_plugin(config) -> Plugin` factory. The
returned object satisfies the Plugin protocol below.

Lifecycle: load (import + version/API-compat checks) → register(registrar)
→ dispose(). Registration is the plugin's ONLY power: the registrar it
receives can add skills, tools, hooks, and CLI commands — and nothing
else. It gets no loop, no model, no stores, no other plugins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

#: Bumped on breaking registrar/contract changes; plugins declare the
#: version they were written against and must match exactly.
PLUGIN_API_VERSION = 1

#: Entry-point group for installed distributions.
ENTRY_POINT_GROUP = "agentloop.plugin"

#: Module attribute local-path plugins must expose.
FACTORY_ATTR = "agentloop_plugin"


@runtime_checkable
class Plugin(Protocol):
    name: str
    version: str  # the plugin's own semver, checked against the manifest constraint
    api_version: int  # must equal PLUGIN_API_VERSION

    def register(self, registrar: Any) -> None: ...

    def dispose(self) -> None: ...


PluginFactory = Callable[[dict[str, Any]], Plugin]


@dataclass(frozen=True, slots=True)
class CliCommand:
    """A plugin-contributed CLI command; mounted when the CLI ships."""

    name: str
    handler: Callable[..., Any]
    help: str = ""
    plugin: str = ""
