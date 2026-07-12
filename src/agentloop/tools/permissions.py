"""Allowlist gating (§8): glob match on canonical 'server.tool' ids.

The manifest allowlist speaks the dotted syntax ('fs.read_*'). Wire names
sent to models use '__' (provider tool-name rules forbid dots), so each
ToolSpec carries its canonical id in permissions_tag; specs without one
(builtins) match on their bare name. No allowlist entry, no execution —
fail closed.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from collections.abc import Sequence

from agentloop.types import ToolSpec


def canonical_name(spec: ToolSpec) -> str:
    return spec.permissions_tag or spec.name


def is_allowed(spec: ToolSpec, allowlist: Sequence[str]) -> bool:
    name = canonical_name(spec)
    return any(fnmatchcase(name, pattern) for pattern in allowlist)
