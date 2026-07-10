"""FallbackChain: retry, advance, exhaustion, trace events, streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from agentloop.models.fallback import FallbackChain
from agentloop.models.protocol import (
    CompletionRequest,
    CompletionResult,
    ModelProvider,
    StreamCompleted,
    StreamEvent,
    TextDelta,
)
from agentloop.types import (
    Cost,
    Message,
    ProviderError,
    ProviderExhaustedError,
    ToolSpec,
    TransientProviderError,
    Usage,
)


def _result(provider: str, model: str) -> CompletionResult:
    return CompletionResult(
        message=Message.assistant("ok"),
        stop_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1),
        cost=Cost(),
        provider=provider,
        model=model,
    )


class FakeProvider:
    """Scripted provider: a list of exceptions to raise before succeeding."""

    def __init__(self, name: str, failures: Sequence[Exception] = ()):
        self.provider = name
        self.model = "fake-model"
        self.calls = 0
        self._failures = list(failures)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        return _result(self.provider, self.model)

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        yield TextDelta(text="ok")
        yield StreamCompleted(result=_result(self.provider, self.model))

    def tool_call_schema(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        return []

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return 0


class RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind: str, payload) -> None:
        self.events.append((kind, dict(payload)))


REQUEST = CompletionRequest(messages=(Message.user("hi"),))


async def _no_sleep(_: float) -> None:
    return None


def _chain(providers, *, retries=1, emitter=None, sleep=_no_sleep) -> FallbackChain:
    return FallbackChain(providers, retries=retries, emitter=emitter, sleep=sleep)


class TestComplete:
    async def test_satisfies_protocol(self):
        assert isinstance(_chain([FakeProvider("a")]), ModelProvider)

    async def test_transient_retries_then_succeeds_on_same_provider(self):
        provider = FakeProvider("a", [TransientProviderError("a", "429")])
        result = await _chain([provider]).complete(REQUEST)
        assert result.provider == "a"
        assert provider.calls == 2  # one failure + one retry, no fallback needed

    async def test_sleeps_between_retries_with_backoff(self):
        sleeps: list[float] = []

        async def record(s: float) -> None:
            sleeps.append(s)

        provider = FakeProvider(
            "a",
            [TransientProviderError("a", "1"), TransientProviderError("a", "2")],
        )
        await _chain([provider], retries=2, sleep=record).complete(REQUEST)
        assert len(sleeps) == 2
        assert sleeps[1] > sleeps[0] * 0.9  # exponential-ish despite jitter

    async def test_transient_exhaustion_advances_to_next_provider(self):
        primary = FakeProvider(
            "a",
            [TransientProviderError("a", "x"), TransientProviderError("a", "y")],
        )
        backup = FakeProvider("b")
        result = await _chain([primary, backup]).complete(REQUEST)
        assert result.provider == "b"
        assert primary.calls == 2  # initial + 1 retry
        assert backup.calls == 1

    async def test_non_retryable_advances_immediately(self):
        primary = FakeProvider("a", [ProviderError("a", "401 bad key")])
        backup = FakeProvider("b")
        result = await _chain([primary, backup], retries=3).complete(REQUEST)
        assert result.provider == "b"
        assert primary.calls == 1  # no retries wasted on auth failure

    async def test_all_exhausted_raises_with_attempt_log(self):
        providers = [
            FakeProvider("a", [ProviderError("a", "boom")]),
            FakeProvider("b", [TransientProviderError("b", "503")] * 2),
        ]
        with pytest.raises(ProviderExhaustedError) as exc_info:
            await _chain(providers).complete(REQUEST)
        assert len(exc_info.value.attempts) == 3  # a once, b twice
        assert "a/fake-model" in exc_info.value.attempts[0]

    async def test_fallback_emits_trace_event(self):
        emitter = RecordingEmitter()
        providers = [FakeProvider("a", [ProviderError("a", "nope")]), FakeProvider("b")]
        await _chain(providers, emitter=emitter).complete(REQUEST)
        assert len(emitter.events) == 1
        kind, payload = emitter.events[0]
        assert kind == "model.fallback"
        assert payload["provider"] == "a"
        assert payload["retryable"] is False


class TestStream:
    async def test_stream_falls_back_before_first_token(self):
        primary = FakeProvider("a", [TransientProviderError("a", "x")] * 2)
        backup = FakeProvider("b")
        events = [e async for e in _chain([primary, backup]).stream(REQUEST)]
        assert isinstance(events[0], TextDelta)
        final = events[-1]
        assert isinstance(final, StreamCompleted)
        assert final.result.provider == "b"

    async def test_stream_all_exhausted(self):
        primary = FakeProvider("a", [ProviderError("a", "denied")])
        with pytest.raises(ProviderExhaustedError):
            async for _ in _chain([primary]).stream(REQUEST):
                pass

    async def test_empty_chain_rejected(self):
        with pytest.raises(ValueError):
            FallbackChain([])
