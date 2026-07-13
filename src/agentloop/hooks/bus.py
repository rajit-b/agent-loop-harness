"""HookBus: priority-ordered dispatch with the §9 execution semantics.

- Ascending priority, ties broken by registration order (manifest hooks
  install first, plugins follow in load order — Phase 7).
- Sync and async handlers both accepted; a sync handler flagged
  `blocking=True` runs via asyncio.to_thread, otherwise inline.
- A hook raising is itself an on_error event: the failing hook is skipped,
  the chain continues (a broken observer must not take down the run), and
  the failure is traced. on_error's own hooks never re-enter on_error.
- Veto on a non-vetoable event (on_error, on_turn_end) is ignored and
  traced as decision "veto_ignored".
- Every execution emits a `hook.executed` trace event carrying the
  decision and the names of payload fields the hook changed.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from agentloop.config.manifest import HookEvent
from agentloop.hooks.contract import (
    VETOABLE_EVENTS,
    Continue,
    ErrorPayload,
    HookContext,
    HookHandler,
    Veto,
)
from agentloop.types import NullEmitter, TraceEmitter

P = TypeVar("P")


@dataclass(slots=True)
class RegisteredHook:
    name: str
    handler: HookHandler
    priority: int
    config: dict[str, Any]
    blocking: bool
    order: int  # registration sequence — the tie-breaker


@dataclass(slots=True)
class DispatchOutcome(Generic[P]):
    payload: P
    veto: Veto | None = None
    vetoed_by: str | None = None

    @property
    def vetoed(self) -> bool:
        return self.veto is not None


def _changed_fields(before: Any, after: Any) -> list[str]:
    if after is before:
        return []
    if isinstance(before, BaseModel) and type(after) is type(before):
        return [
            name
            for name in type(before).model_fields
            if getattr(before, name) != getattr(after, name)
        ]
    return ["payload"] if before != after else []


@dataclass(slots=True)
class HookBus:
    emitter: TraceEmitter = field(default_factory=NullEmitter)
    _hooks: dict[HookEvent, list[RegisteredHook]] = field(default_factory=dict)
    _counter: int = 0

    def register(
        self,
        event: HookEvent,
        handler: HookHandler,
        *,
        priority: int = 100,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        blocking: bool = False,
    ) -> None:
        self._counter += 1
        self._hooks.setdefault(event, []).append(
            RegisteredHook(
                name=name or getattr(handler, "__qualname__", repr(handler)),
                handler=handler,
                priority=priority,
                config=config or {},
                blocking=blocking,
                order=self._counter,
            )
        )

    def hooks_for(self, event: HookEvent) -> list[RegisteredHook]:
        return sorted(self._hooks.get(event, []), key=lambda h: (h.priority, h.order))

    async def dispatch(
        self, event: HookEvent, payload: P, *, run_id: str = ""
    ) -> DispatchOutcome[P]:
        current = payload
        for hook in self.hooks_for(event):
            ctx = HookContext(event=event, run_id=run_id, config=dict(hook.config))
            try:
                decision = await self._invoke(hook, current, ctx)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — a broken hook must not kill the run
                self.emitter.emit(
                    "hook.error",
                    {"event": event, "hook": hook.name, "error": f"{type(exc).__name__}: {exc}"},
                )
                if event != "on_error":  # a hook raising is itself an on_error event
                    await self.dispatch(
                        "on_error",
                        ErrorPayload(
                            error=str(exc),
                            kind=type(exc).__name__,
                            source=f"hook:{event}:{hook.name}",
                        ),
                        run_id=run_id,
                    )
                continue

            if isinstance(decision, Veto):
                if event in VETOABLE_EVENTS:
                    self._trace(event, hook, "veto", reason=decision.reason)
                    return DispatchOutcome(
                        payload=current, veto=decision, vetoed_by=hook.name
                    )
                self._trace(event, hook, "veto_ignored", reason=decision.reason)
                continue

            if isinstance(decision, Continue) and decision.payload is not None:
                changed = _changed_fields(current, decision.payload)
                self._trace(event, hook, "mutated", changed=changed)
                current = decision.payload
            else:  # Continue() or None: pass through unchanged
                self._trace(event, hook, "continue")
        return DispatchOutcome(payload=current)

    async def _invoke(self, hook: RegisteredHook, payload: Any, ctx: HookContext):
        if inspect.iscoroutinefunction(hook.handler):
            return await hook.handler(payload, ctx)
        if hook.blocking:
            return await asyncio.to_thread(hook.handler, payload, ctx)
        result = hook.handler(payload, ctx)
        if inspect.isawaitable(result):
            return await result
        return result

    def _trace(
        self,
        event: HookEvent,
        hook: RegisteredHook,
        decision: str,
        *,
        reason: str | None = None,
        changed: list[str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": event,
            "hook": hook.name,
            "priority": hook.priority,
            "decision": decision,
        }
        if reason is not None:
            payload["reason"] = reason
        if changed:
            payload["changed_fields"] = changed
        self.emitter.emit("hook.executed", payload)
