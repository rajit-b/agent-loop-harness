"""Handlers referenced by dotted path in loader tests."""

from __future__ import annotations

from agentloop.hooks.contract import Continue, HookContext, PreToolPayload


def redact_secrets(payload: PreToolPayload, ctx: HookContext):
    """Replace configured patterns in string arguments with [REDACTED]."""
    patterns = ctx.config.get("patterns", [])
    arguments = dict(payload.call.arguments)
    changed = False
    for key, value in arguments.items():
        if isinstance(value, str):
            for pattern in patterns:
                if pattern in value:
                    arguments[key] = value.replace(pattern, "[REDACTED]")
                    value = arguments[key]
                    changed = True
    if not changed:
        return None
    return Continue(
        payload=payload.model_copy(
            update={"call": payload.call.model_copy(update={"arguments": arguments})}
        )
    )


not_callable = "just a string"
