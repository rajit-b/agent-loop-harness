"""FallbackChain — a composite ModelProvider (§6).

Policy: TransientProviderError → exponential backoff + jitter, up to
`retries` per provider, then advance. Non-transient ProviderError (auth,
bad request) → advance immediately, no retries — a broken key on provider
A must not kill a run that provider B could serve. Every advance emits a
`model.fallback` trace event. All providers exhausted → ProviderExhaustedError.

Streaming: fallback applies until the first event is obtained; a failure
mid-stream propagates (partial text has already been surfaced and cannot
be transparently restarted).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from agentloop.models.protocol import (
    CompletionRequest,
    CompletionResult,
    ModelProvider,
    StreamEvent,
)
from agentloop.types import (
    Message,
    NullEmitter,
    ProviderError,
    ProviderExhaustedError,
    ToolSpec,
    TraceEmitter,
    TransientProviderError,
)


class FallbackChain:
    def __init__(
        self,
        providers: Sequence[ModelProvider],
        *,
        retries: int = 2,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        emitter: TraceEmitter | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,  # test seam
    ):
        if not providers:
            raise ValueError("FallbackChain requires at least one provider")
        self._providers = list(providers)
        self._retries = retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._emitter = emitter or NullEmitter()
        self._sleep = sleep

    # The chain fronts its primary; results carry the provider that
    # actually served them.
    @property
    def provider(self) -> str:
        return self._providers[0].provider

    @property
    def model(self) -> str:
        return self._providers[0].model

    def tool_call_schema(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        return self._providers[0].tool_call_schema(tools)

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return self._providers[0].count_tokens(messages)

    def _backoff(self, attempt: int) -> float:
        delay = min(self._backoff_cap, self._backoff_base * (2**attempt))
        return delay * (0.5 + random.random() / 2)  # jitter: 50-100% of delay

    def _emit_fallback(self, provider: ModelProvider, error: Exception) -> None:
        self._emitter.emit(
            "model.fallback",
            {
                "provider": provider.provider,
                "model": provider.model,
                "error": str(error),
                "retryable": isinstance(error, TransientProviderError),
            },
        )

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        attempts: list[str] = []
        for provider in self._providers:
            for attempt in range(self._retries + 1):
                try:
                    return await provider.complete(request)
                except TransientProviderError as exc:
                    attempts.append(f"{provider.provider}/{provider.model}: {exc}")
                    if attempt < self._retries:
                        await self._sleep(self._backoff(attempt))
                        continue
                    self._emit_fallback(provider, exc)
                    break
                except ProviderError as exc:  # non-retryable → advance now
                    attempts.append(f"{provider.provider}/{provider.model}: {exc}")
                    self._emit_fallback(provider, exc)
                    break
        raise ProviderExhaustedError(attempts)

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        attempts: list[str] = []
        for provider in self._providers:
            for attempt in range(self._retries + 1):
                events = provider.stream(request)
                try:
                    first = await anext(events)
                except TransientProviderError as exc:
                    attempts.append(f"{provider.provider}/{provider.model}: {exc}")
                    if attempt < self._retries:
                        await self._sleep(self._backoff(attempt))
                        continue
                    self._emit_fallback(provider, exc)
                    break
                except ProviderError as exc:
                    attempts.append(f"{provider.provider}/{provider.model}: {exc}")
                    self._emit_fallback(provider, exc)
                    break
                yield first
                async for event in events:  # mid-stream failures propagate
                    yield event
                return
        raise ProviderExhaustedError(attempts)
