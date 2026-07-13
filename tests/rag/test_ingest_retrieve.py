"""Ingest→retrieve round-trip, incremental reindex, hybrid vs vector-only."""

from __future__ import annotations

import pytest

from agentloop.rag.ingest import Indexer
from agentloop.rag.retrieve import Retriever, rrf_fuse
from agentloop.rag.sources import FileSource


@pytest.fixture
def indexer(store, embedder) -> Indexer:
    return Indexer(store, embedder, chunk_size=128, chunk_overlap=16)


class TestRoundTripWithCitations:
    async def test_ingest_then_retrieve_carries_citations(
        self, indexer, store, embedder, docs_dir
    ):
        report = await indexer.index(FileSource([docs_dir]))
        assert len(report.added) == 2

        retriever = Retriever(store, embedder, top_k=3)
        chunks = await retriever.retrieve("how does the config loader resolve precedence?")
        assert chunks, "retrieval returned nothing"
        top = chunks[0]
        assert "loader" in top.text.lower()
        # citation is complete and verifiable against the source file
        assert top.citation.source_uri.endswith("config.md")
        source_text = (docs_dir / "config.md").read_text()
        assert source_text[top.citation.start : top.citation.end] == top.text
        assert top.citation.content_hash  # document hash rides along
        assert top.citation.score > 0


class TestIncrementalReindex:
    async def test_unchanged_files_are_skipped_without_reembedding(
        self, indexer, embedder, docs_dir
    ):
        await indexer.index(FileSource([docs_dir]))
        embedded_first_pass = embedder.texts_embedded

        report = await indexer.index(FileSource([docs_dir]))  # nothing changed
        assert len(report.skipped) == 2
        assert report.added == [] and report.updated == []
        assert embedder.texts_embedded == embedded_first_pass  # zero new embeds

    async def test_changed_file_is_reindexed(self, indexer, store, embedder, docs_dir):
        await indexer.index(FileSource([docs_dir]))
        (docs_dir / "config.md").write_text(
            "# Configuration\n\nThe loader now also supports profiles.\n"
        )
        report = await indexer.index(FileSource([docs_dir]))
        assert [u.endswith("config.md") for u in report.updated] == [True]
        assert len(report.skipped) == 1  # budgets.md untouched
        # old content gone, new content searchable
        assert store.get_chunks(store.keyword_search("precedence", 5)) == []
        [hit] = store.get_chunks(store.keyword_search("profiles", 5))
        assert "profiles" in hit.text

    async def test_deleted_file_is_swept(self, indexer, store, docs_dir):
        await indexer.index(FileSource([docs_dir]))
        (docs_dir / "budgets.md").unlink()
        report = await indexer.index(FileSource([docs_dir]))
        assert [u.endswith("budgets.md") for u in report.removed] == [True]
        assert store.keyword_search("ZQXW", 5) == []


class TestHybridBeatsVectorOnly:
    async def test_keyword_heavy_query_needs_bm25(
        self, indexer, store, embedder, docs_dir
    ):
        """'ZQXW-7741' is outside the embedder's vocabulary: pure vector
        search cannot find the budgets doc, BM25 nails it, and the fused
        ranking puts it first."""
        await indexer.index(FileSource([docs_dir]))
        query = "what is ZQXW-7741?"

        # vector-only: the embedder is blind to the code, so the top hit
        # does NOT contain it
        [query_vector] = await embedder.embed([query])
        vector_ids = store.vector_search(query_vector, 5)
        vector_top = store.get_chunks(vector_ids[:1])
        assert vector_top and "ZQXW" not in vector_top[0].text

        # hybrid: RRF fusion surfaces the exact-match chunk at rank 1
        retriever = Retriever(store, embedder, top_k=3)
        chunks = await retriever.retrieve(query)
        assert "ZQXW-7741" in chunks[0].text


class TestRRF:
    def test_rrf_math(self):
        scores = rrf_fuse([[1, 2, 3], [3, 4]], k=60)
        assert scores[1] == pytest.approx(1 / 61)
        assert scores[3] == pytest.approx(1 / 63 + 1 / 61)  # in both lists
        assert scores[3] > scores[2]

    def test_rrf_empty_rankings(self):
        assert rrf_fuse([[], []]) == {}


class TestReranker:
    async def test_reranker_is_applied(self, indexer, store, embedder, docs_dir):
        await indexer.index(FileSource([docs_dir]))

        class Reverser:
            async def rerank(self, query, chunks):
                return list(reversed(chunks))

        plain = Retriever(store, embedder, top_k=3)
        reversed_ = Retriever(store, embedder, top_k=3, reranker=Reverser())
        base = await plain.retrieve("config loader")
        flipped = await reversed_.retrieve("config loader")
        assert [c.chunk_id for c in flipped] == [c.chunk_id for c in base][::-1]
