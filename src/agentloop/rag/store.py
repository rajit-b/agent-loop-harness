"""RAG store (§11): rag_documents, rag_chunks, sqlite-vec KNN, FTS5 BM25.

One embedding space per store (A5): the embedding model and dimension
are recorded in rag_meta; a mismatch at initialize() wipes the index and
returns True so the caller re-ingests rather than silently mixing spaces.
FTS5 is external-content against rag_chunks and maintained explicitly on
add/delete — no triggers.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlite_vec import serialize_float32

from agentloop.rag.chunk import Chunk


@dataclass(frozen=True, slots=True)
class StoredChunk:
    id: int
    uri: str
    heading: str
    text: str
    start: int
    end: int
    content_hash: str  # the parent document's hash


class RagStore:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._dimensions: int | None = None

    def initialize(self, *, embedding_model: str, dimensions: int) -> bool:
        """Create tables; wipe if the embedding space changed. Returns True
        when a wipe happened (caller must re-ingest)."""
        conn = self._conn
        conn.execute("CREATE TABLE IF NOT EXISTS rag_meta (key TEXT PRIMARY KEY, value TEXT)")
        stored = dict(conn.execute("SELECT key, value FROM rag_meta"))
        wiped = False
        if stored and (
            stored.get("embedding_model") != embedding_model
            or stored.get("dimensions") != str(dimensions)
        ):
            for table in ("rag_chunks_vec", "rag_chunks_fts", "rag_chunks", "rag_documents"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute("DELETE FROM rag_meta")
            wiped = True
        conn.execute(
            "INSERT OR REPLACE INTO rag_meta VALUES ('embedding_model', ?), ('dimensions', ?)",
            (embedding_model, str(dimensions)),
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_documents ("
            " uri TEXT PRIMARY KEY, content_hash TEXT NOT NULL, indexed_at REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_chunks ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " uri TEXT NOT NULL, chunk_index INTEGER, heading TEXT, text TEXT,"
            " start INTEGER, end INTEGER)"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_vec "
            f"USING vec0(embedding float[{dimensions}])"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts "
            "USING fts5(text, content='rag_chunks', content_rowid='id')"
        )
        conn.commit()
        self._dimensions = dimensions
        return wiped

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------

    def document_hash(self, uri: str) -> str | None:
        row = self._conn.execute(
            "SELECT content_hash FROM rag_documents WHERE uri = ?", (uri,)
        ).fetchone()
        return row["content_hash"] if row else None

    def all_uris(self) -> set[str]:
        return {r["uri"] for r in self._conn.execute("SELECT uri FROM rag_documents")}

    def add_document(
        self,
        uri: str,
        content_hash: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        assert len(chunks) == len(embeddings)
        conn = self._conn
        conn.execute(
            "INSERT OR REPLACE INTO rag_documents VALUES (?, ?, ?)",
            (uri, content_hash, time.time()),
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            cursor = conn.execute(
                "INSERT INTO rag_chunks (uri, chunk_index, heading, text, start, end)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (uri, chunk.index, chunk.heading, chunk.text, chunk.start, chunk.end),
            )
            chunk_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO rag_chunks_vec (rowid, embedding) VALUES (?, ?)",
                (chunk_id, serialize_float32(list(embedding))),
            )
            conn.execute(
                "INSERT INTO rag_chunks_fts (rowid, text) VALUES (?, ?)",
                (chunk_id, chunk.text),
            )
        conn.commit()

    def delete_document(self, uri: str) -> None:
        conn = self._conn
        rows = conn.execute(
            "SELECT id, text FROM rag_chunks WHERE uri = ?", (uri,)
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO rag_chunks_fts (rag_chunks_fts, rowid, text)"
                " VALUES ('delete', ?, ?)",
                (row["id"], row["text"]),
            )
            conn.execute("DELETE FROM rag_chunks_vec WHERE rowid = ?", (row["id"],))
        conn.execute("DELETE FROM rag_chunks WHERE uri = ?", (uri,))
        conn.execute("DELETE FROM rag_documents WHERE uri = ?", (uri,))
        conn.commit()

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def vector_search(self, query: Sequence[float], k: int) -> list[int]:
        rows = self._conn.execute(
            "SELECT rowid FROM rag_chunks_vec WHERE embedding MATCH ? AND k = ?"
            " ORDER BY distance",
            (serialize_float32(list(query)), k),
        ).fetchall()
        return [row["rowid"] for row in rows]

    def keyword_search(self, query: str, k: int) -> list[int]:
        tokens = re.findall(r"\w+", query)
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        rows = self._conn.execute(
            "SELECT rowid FROM rag_chunks_fts WHERE rag_chunks_fts MATCH ?"
            " ORDER BY rank LIMIT ?",
            (match, k),
        ).fetchall()
        return [row["rowid"] for row in rows]

    def get_chunks(self, ids: Sequence[int]) -> list[StoredChunk]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT c.*, d.content_hash FROM rag_chunks c"
            f" JOIN rag_documents d ON d.uri = c.uri"
            f" WHERE c.id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        by_id = {
            row["id"]: StoredChunk(
                id=row["id"],
                uri=row["uri"],
                heading=row["heading"],
                text=row["text"],
                start=row["start"],
                end=row["end"],
                content_hash=row["content_hash"],
            )
            for row in rows
        }
        return [by_id[i] for i in ids if i in by_id]

    def chunk_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
