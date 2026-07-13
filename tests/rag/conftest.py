from __future__ import annotations

import re

import pytest

from agentloop.rag.store import RagStore
from agentloop.storage.sqlite import connect

from ..loop.conftest import RecordingEmitter

#: Vocabulary the fake embedder can "see". Deliberately EXCLUDES rare
#: identifiers like error codes — that blindness is what the
#: hybrid-beats-vector test exploits.
VOCAB = [
    "config", "loader", "manifest", "resolve", "precedence",
    "budget", "token", "wall", "clock", "cost",
    "hook", "veto", "mutate", "payload",
    "error", "retry", "provider",
]


class VocabEmbedder:
    """Deterministic bag-of-words embedder; counts embed() calls."""

    def __init__(self, vocab: list[str] | None = None):
        self.vocab = vocab or VOCAB
        self.calls = 0
        self.texts_embedded = 0

    async def embed(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        vectors = []
        for text in texts:
            words = re.findall(r"[a-z]+", text.lower())
            # +tiny epsilon so no vector is all-zero (vec0 accepts it fine
            # but cosine-ish distances stay meaningful)
            vectors.append(
                [float(words.count(term)) + 0.001 for term in self.vocab]
            )
        return vectors


DOC_CONFIG = """# Configuration

## The loader

The config loader reads the manifest and applies resolution precedence:
defaults, then manifest, then environment, then CLI flags.

## Validation

Unknown keys fail closed. The loader validates the manifest eagerly.
"""

DOC_BUDGETS = """# Budgets

The loop enforces a token budget, a wall clock budget, and a cost budget.
Budgets are checked at transition boundaries.

When a budget is exhausted the loop wraps up with error code ZQXW-7741.
"""


@pytest.fixture
def store() -> RagStore:
    store = RagStore(connect(":memory:"))
    store.initialize(embedding_model="fake", dimensions=len(VOCAB))
    return store


@pytest.fixture
def embedder() -> VocabEmbedder:
    return VocabEmbedder()


@pytest.fixture
def emitter() -> RecordingEmitter:
    return RecordingEmitter()


@pytest.fixture
def docs_dir(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "config.md").write_text(DOC_CONFIG, encoding="utf-8")
    (docs / "budgets.md").write_text(DOC_BUDGETS, encoding="utf-8")
    return docs
