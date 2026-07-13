"""Hybrid retrieval (§11): vec KNN + FTS5 BM25 → RRF fusion → optional
rerank. Deterministic by default (A6): RRF costs no model call. Every
retrieved chunk carries a full Citation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agentloop.rag.store import RagStore
from agentloop.types import Citation, EmbeddingProvider


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: int
    text: str
    uri: str
    heading: str
    score: float
    citation: Citation


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]: ...


def rrf_fuse(rankings: Sequence[Sequence[int]], *, k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: score(d) = Σ 1/(k + rank_i(d))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class Retriever:
    def __init__(
        self,
        store: RagStore,
        embedder: EmbeddingProvider,
        *,
        top_k: int = 8,
        vector_k: int = 20,
        fts_k: int = 20,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
    ):
        self._store = store
        self._embedder = embedder
        self.top_k = top_k
        self._vector_k = vector_k
        self._fts_k = fts_k
        self._rrf_k = rrf_k
        self._reranker = reranker

    async def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        limit = top_k or self.top_k
        [query_vector] = await self._embedder.embed([query])
        vector_ids = self._store.vector_search(query_vector, self._vector_k)
        keyword_ids = self._store.keyword_search(query, self._fts_k)
        scores = rrf_fuse([vector_ids, keyword_ids], k=self._rrf_k)
        ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:limit]
        chunks = [
            RetrievedChunk(
                chunk_id=stored.id,
                text=stored.text,
                uri=stored.uri,
                heading=stored.heading,
                score=scores[stored.id],
                citation=Citation(
                    source_uri=stored.uri,
                    start=stored.start,
                    end=stored.end,
                    content_hash=stored.content_hash,
                    score=scores[stored.id],
                ),
            )
            for stored in self._store.get_chunks(ranked_ids)
        ]
        if self._reranker is not None:
            chunks = await self._reranker.rerank(query, chunks)
        return chunks


def format_context_block(chunks: Sequence[RetrievedChunk]) -> str:
    """The cited context block appended to the user turn (§11)."""
    lines = ["## Retrieved context"]
    for n, chunk in enumerate(chunks, start=1):
        location = f"{chunk.uri}"
        if chunk.heading:
            location += f" § {chunk.heading}"
        lines.append(f"[{n}] {location} (chars {chunk.citation.start}-{chunk.citation.end})")
        lines.append(chunk.text.strip())
    lines.append("Cite sources with their [n] markers where relevant.")
    return "\n\n".join(lines)
