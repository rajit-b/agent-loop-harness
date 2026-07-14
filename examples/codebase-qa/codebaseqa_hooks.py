"""Worked-example hook: redact secrets from tool-call arguments (§ worked
example). A pre_tool hook returns a modified copy of the payload (never
mutates in place) so the trace can diff it. Config carries the patterns.

The example manifest references this by module path
`codebaseqa_hooks:redact_secrets`; put this directory on PYTHONPATH (or run
from it) so the loader can import it.
"""

from __future__ import annotations

import re

from agentloop.hooks.contract import Continue, HookContext, PreToolPayload

DEFAULT_PATTERNS = (
    r"sk-[A-Za-z0-9_-]{6,}",           # OpenAI-style keys
    r"(?i)ghp_[A-Za-z0-9]{20,}",       # GitHub tokens
    r"(?i)AKIA[0-9A-Z]{12,}",          # AWS access key ids
    r"(?i)bearer\s+[A-Za-z0-9._-]{8,}",
)


def redact_secrets(payload: PreToolPayload, ctx: HookContext):
    patterns = ctx.config.get("patterns") or DEFAULT_PATTERNS
    compiled = [re.compile(p) for p in patterns]

    def scrub(value):
        if isinstance(value, str):
            for pattern in compiled:
                value = pattern.sub("[REDACTED]", value)
            return value
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        return value

    scrubbed = {k: scrub(v) for k, v in payload.call.arguments.items()}
    if scrubbed == payload.call.arguments:
        return None  # nothing to redact — pass through unchanged
    return Continue(
        payload=payload.model_copy(
            update={"call": payload.call.model_copy(update={"arguments": scrubbed})}
        )
    )
