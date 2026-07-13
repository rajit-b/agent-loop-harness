"""Indexing with hash-based incremental reindex (§11): unchanged skip,
changed replace, deleted swept."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentloop.rag.chunk import chunk_text
from agentloop.rag.sources import Source
from agentloop.rag.store import RagStore
from agentloop.types import EmbeddingProvider

EMBED_BATCH = 64


@dataclass(slots=True)
class IndexReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


class Indexer:
    def __init__(
        self,
        store: RagStore,
        embedder: EmbeddingProvider,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self._store = store
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def index(self, source: Source) -> IndexReport:
        report = IndexReport()
        seen: set[str] = set()
        for document in source.documents():
            seen.add(document.uri)
            existing_hash = self._store.document_hash(document.uri)
            if existing_hash == document.content_hash:
                report.skipped.append(document.uri)
                continue
            if existing_hash is not None:
                self._store.delete_document(document.uri)
                report.updated.append(document.uri)
            else:
                report.added.append(document.uri)
            chunks = chunk_text(
                document.content, size=self._chunk_size, overlap=self._chunk_overlap
            )
            embeddings: list[list[float]] = []
            for batch_start in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[batch_start : batch_start + EMBED_BATCH]
                embeddings.extend(await self._embedder.embed([c.text for c in batch]))
            self._store.add_document(
                document.uri, document.content_hash, chunks, embeddings
            )
        for uri in self._store.all_uris() - seen:  # deletion sweep
            self._store.delete_document(uri)
            report.removed.append(uri)
        return report
