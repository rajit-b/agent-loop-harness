"""Hooks: mutate/veto contract, priority bus, manifest loader (Phase 5)."""

from agentloop.hooks.bus import DispatchOutcome, HookBus
from agentloop.hooks.contract import (
    Continue,
    ErrorPayload,
    HookContext,
    HookDecision,
    PostModelPayload,
    PostRetrievalPayload,
    PostToolPayload,
    PreModelPayload,
    PreRetrievalPayload,
    PreToolPayload,
    TurnEndPayload,
    Veto,
)
from agentloop.hooks.loader import install_manifest_hooks, resolve_handler

__all__ = [
    "Continue",
    "DispatchOutcome",
    "ErrorPayload",
    "HookBus",
    "HookContext",
    "HookDecision",
    "PostModelPayload",
    "PostRetrievalPayload",
    "PostToolPayload",
    "PreModelPayload",
    "PreRetrievalPayload",
    "PreToolPayload",
    "TurnEndPayload",
    "Veto",
    "install_manifest_hooks",
    "resolve_handler",
]
