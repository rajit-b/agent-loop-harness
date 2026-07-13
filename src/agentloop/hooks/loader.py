"""Install manifest-declared hooks onto a bus.

Handler paths are 'pkg.mod:attr' or 'pkg.mod.attr' (validated by the
manifest schema). Manifest hooks register before any plugin's, which is
exactly the §9 tie-break: same priority → manifest wins.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence

from agentloop.config.manifest import HookEntry, HookEvent
from agentloop.hooks.bus import HookBus
from agentloop.hooks.contract import HookHandler
from agentloop.types import ConfigError


def resolve_handler(path: str) -> HookHandler:
    module_path, sep, attr = path.partition(":")
    if not sep:
        module_path, _, attr = path.rpartition(".")
    if not module_path or not attr:
        raise ConfigError(f"hook handler path {path!r} is not importable")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigError(f"cannot import hook module {module_path!r}: {exc}") from exc
    try:
        handler = getattr(module, attr)
    except AttributeError:
        raise ConfigError(
            f"hook module {module_path!r} has no attribute {attr!r}"
        ) from None
    if not callable(handler):
        raise ConfigError(f"hook handler {path!r} is not callable")
    return handler


def install_manifest_hooks(
    bus: HookBus, hooks_config: Mapping[HookEvent, Sequence[HookEntry]]
) -> None:
    for event, entries in hooks_config.items():
        for entry in entries:
            bus.register(
                event,
                resolve_handler(entry.handler),
                priority=entry.priority,
                name=entry.handler,
                config=dict(entry.config),
            )
