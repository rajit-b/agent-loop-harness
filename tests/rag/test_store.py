"""RagStore: tables, search paths, deletion hygiene, embedding-space guard."""

from __future__ import annotations

from agentloop.rag.chunk import Chunk
from agentloop.rag.store import RagStore
from agentloop.storage.sqlite import connect

DIM = 3


def make_store() -> RagStore:
    store = RagStore(connect(":memory:"))
    store.initialize(embedding_model="m", dimensions=DIM)
    return store


def chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(text=text, index=index, start=0, end=len(text))


class TestRoundTrip:
    def test_add_and_get(self):
        store = make_store()
        store.add_document("doc.md", "h1", [chunk("hello world")], [[1.0, 0, 0]])
        [stored] = store.get_chunks(store.keyword_search("hello", 5))
        assert stored.text == "hello world"
        assert stored.uri == "doc.md"
        assert stored.content_hash == "h1"

    def test_vector_search_orders_by_distance(self):
        store = make_store()
        store.add_document(
            "a.md", "h",
            [chunk("close", 0), chunk("far", 1)],
            [[1.0, 0, 0], [0, 1.0, 0]],
        )
        ids = store.vector_search([0.9, 0.1, 0], k=2)
        chunks = store.get_chunks(ids)
        assert [c.text for c in chunks] == ["close", "far"]

    def test_keyword_search_uses_bm25(self):
        store = make_store()
        store.add_document(
            "a.md", "h",
            [chunk("the ZQXW error code", 0), chunk("nothing relevant", 1)],
            [[1.0, 0, 0], [0, 1.0, 0]],
        )
        ids = store.keyword_search("ZQXW", 5)
        [stored] = store.get_chunks(ids)
        assert "ZQXW" in stored.text

    def test_punctuation_heavy_query_does_not_crash(self):
        store = make_store()
        store.add_document("a.md", "h", [chunk("text")], [[1.0, 0, 0]])
        assert store.keyword_search('"co*de" OR (x AND -y)!', 5) is not None
        assert store.keyword_search("!!!", 5) == []


class TestDeletion:
    def test_delete_document_purges_all_tables(self):
        store = make_store()
        store.add_document("a.md", "h", [chunk("findme unique")], [[1.0, 0, 0]])
        store.delete_document("a.md")
        assert store.chunk_count() == 0
        assert store.keyword_search("findme", 5) == []  # gone from FTS
        assert store.vector_search([1.0, 0, 0], k=5) == []  # gone from vec
        assert store.document_hash("a.md") is None


class TestEmbeddingSpaceGuard:
    def test_dimension_change_wipes_index(self):
        conn = connect(":memory:")
        store = RagStore(conn)
        assert store.initialize(embedding_model="m", dimensions=3) is False
        store.add_document("a.md", "h", [chunk("text")], [[1.0, 0, 0]])
        assert store.chunk_count() == 1
        # same model, new dimensions → forced wipe (A5)
        assert store.initialize(embedding_model="m", dimensions=4) is True
        assert store.chunk_count() == 0

    def test_model_change_wipes_index(self):
        conn = connect(":memory:")
        store = RagStore(conn)
        store.initialize(embedding_model="nomic", dimensions=3)
        store.add_document("a.md", "h", [chunk("text")], [[1.0, 0, 0]])
        assert store.initialize(embedding_model="other", dimensions=3) is True
        assert store.chunk_count() == 0

    def test_matching_space_preserves_index(self):
        conn = connect(":memory:")
        store = RagStore(conn)
        store.initialize(embedding_model="m", dimensions=3)
        store.add_document("a.md", "h", [chunk("text")], [[1.0, 0, 0]])
        assert store.initialize(embedding_model="m", dimensions=3) is False
        assert store.chunk_count() == 1
