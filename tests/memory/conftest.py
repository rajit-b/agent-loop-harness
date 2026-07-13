from __future__ import annotations

import json

import pytest

from agentloop.config.manifest import MemoryConfig, ShortTermMemoryConfig
from agentloop.memory.manager import MemoryManager
from agentloop.memory.store import MemoryStore
from agentloop.storage.sqlite import connect

from ..loop.conftest import RecordingEmitter, ScriptedProvider, result
from ..rag.conftest import VOCAB, VocabEmbedder

DAY = 86_400.0


class FakeNow:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance_days(self, days: float) -> None:
        self.now += days * DAY


def extraction(*facts: dict) -> str:
    """A model completion carrying an extractor JSON payload."""
    return json.dumps(list(facts))


def candidate(
    text: str,
    *,
    type: str = "preference",
    confidence: float = 0.9,
    explicit: bool = False,
    quote: str = "",
) -> dict:
    return {
        "text": text, "type": type, "confidence": confidence,
        "explicit": explicit, "quote": quote,
    }


@pytest.fixture
def now() -> FakeNow:
    return FakeNow()


@pytest.fixture
def mem_store(now) -> MemoryStore:
    store = MemoryStore(connect(":memory:"), now=now)
    store.initialize(embedding_model="fake", dimensions=len(VOCAB))
    return store


@pytest.fixture
def embedder() -> VocabEmbedder:
    return VocabEmbedder()


@pytest.fixture
def emitter() -> RecordingEmitter:
    return RecordingEmitter()


def make_manager(
    mem_store,
    embedder,
    script: list,
    *,
    session_id: str = "session-1",
    max_tokens: int = 8000,
    emitter=None,
) -> tuple[MemoryManager, ScriptedProvider]:
    provider = ScriptedProvider([result(text) if isinstance(text, str) else text
                                 for text in script])
    manager = MemoryManager(
        mem_store,
        embedder,
        provider,
        config=MemoryConfig(short_term=ShortTermMemoryConfig(max_tokens=max_tokens)),
        session_id=session_id,
        emitter=emitter,
    )
    return manager, provider
