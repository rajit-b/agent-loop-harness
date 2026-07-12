"""Sandbox enforcement (A7): path jail and result caps.

This is a trust boundary, not containment — an MCP server is a foreign
process we cannot jail from the client side. What IS enforced here:
path-hinted arguments must resolve (symlinks included) inside the
manifest-declared roots, and results are size-capped. Timeouts are
enforced by the gateway; env scrubbing by the stdio transport.

With no roots configured the path jail is inert (documented in §5's
sandbox key): gating via the allowlist remains the primary control.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from agentloop.types import ToolCall, ToolSpec

TRUNCATION_NOTICE = "\n…[truncated {dropped} of {total} chars by sandbox policy]"


class Sandbox:
    def __init__(self, *, roots: Sequence[str | Path] = (), max_result_chars: int = 100_000):
        self._roots = [Path(r).expanduser().resolve() for r in roots]
        self._max_result_chars = max_result_chars

    def check_paths(self, spec: ToolSpec, call: ToolCall) -> str | None:
        """Return a violation message, or None if the call is clean."""
        if not self._roots:
            return None
        for arg_name in spec.path_hints:
            raw = call.arguments.get(arg_name)
            values = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
            for value in values:
                if not isinstance(value, str) or not value:
                    continue
                if not self._within_roots(value):
                    return (
                        f"path {value!r} (argument {arg_name!r}) escapes the "
                        f"sandbox roots"
                    )
        return None

    def _within_roots(self, value: str) -> bool:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self._roots[0] / candidate
        resolved = candidate.resolve()  # collapses .. and follows symlinks
        return any(resolved.is_relative_to(root) for root in self._roots)

    def cap_result(self, content: str) -> str:
        if len(content) <= self._max_result_chars:
            return content
        kept = content[: self._max_result_chars]
        return kept + TRUNCATION_NOTICE.format(
            dropped=len(content) - len(kept), total=len(content)
        )
