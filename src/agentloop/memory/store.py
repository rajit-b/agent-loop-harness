"""Memory persistence (§11): mem_episodic + rolling summary (short-term),
mem_facts + mem_facts_vec + mem_kv (long-term).

The episodic log is append-only; overflow summarization FOLDS rows
(folded=1) into the rolling summary rather than deleting them — history
assembly reads summary + unfolded tail, consolidation reads everything
past its cursor. Facts are never deleted: candidate → active → archived.
The facts vec table uses cosine distance so the dedup threshold (A5/§11:
cosine ≥ 0.92) maps directly to distance ≤ 0.08.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sqlite_vec import serialize_float32


@dataclass(frozen=True, slots=True)
class EpisodicRow:
    id: int
    session_id: str
    turn_id: str
    role: str
    content: str
    folded: bool


@dataclass(frozen=True, slots=True)
class FactRecord:
    id: int
    text: str
    type: str
    status: str  # candidate | active | archived
    confidence: float
    support_count: int
    session_count: int
    last_session_id: str
    source_quote: str
    source_session: str
    source_turn: str
    last_accessed: float


class MemoryStore:
    def __init__(
        self, conn: sqlite3.Connection, *, now: Callable[[], float] = time.time
    ):
        self._conn = conn
        self._now = now

    def initialize(self, *, embedding_model: str, dimensions: int) -> bool:
        """Create tables; wipe the FACT index if the embedding space changed
        (episodic text carries no embeddings and always survives)."""
        conn = self._conn
        conn.execute("CREATE TABLE IF NOT EXISTS mem_meta (key TEXT PRIMARY KEY, value TEXT)")
        stored = dict(conn.execute("SELECT key, value FROM mem_meta"))
        wiped = False
        if stored and (
            stored.get("embedding_model") != embedding_model
            or stored.get("dimensions") != str(dimensions)
        ):
            conn.execute("DROP TABLE IF EXISTS mem_facts_vec")
            conn.execute("DROP TABLE IF EXISTS mem_facts")
            conn.execute("DELETE FROM mem_meta")
            wiped = True
        conn.execute(
            "INSERT OR REPLACE INTO mem_meta VALUES ('embedding_model', ?), ('dimensions', ?)",
            (embedding_model, str(dimensions)),
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS mem_episodic ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, turn_id TEXT,"
            " role TEXT, content TEXT, folded INTEGER DEFAULT 0, created_at REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS mem_summary (session_id TEXT PRIMARY KEY, summary TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS mem_facts ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " text TEXT NOT NULL, type TEXT, status TEXT DEFAULT 'candidate',"
            " confidence REAL, support_count INTEGER DEFAULT 1,"
            " session_count INTEGER DEFAULT 1, last_session_id TEXT,"
            " source_quote TEXT, source_session TEXT, source_turn TEXT,"
            " created_at REAL, last_accessed REAL)"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS mem_facts_vec "
            f"USING vec0(embedding float[{dimensions}] distance_metric=cosine)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS mem_kv (key TEXT PRIMARY KEY, value TEXT)")
        conn.commit()
        return wiped

    # ------------------------------------------------------------------
    # short-term: episodic log + rolling summary
    # ------------------------------------------------------------------

    def append_episodic(
        self, session_id: str, turn_id: str, role: str, content: str
    ) -> int:
        cursor = self._conn.execute(
            "INSERT INTO mem_episodic (session_id, turn_id, role, content, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (session_id, turn_id, role, content, self._now()),
        )
        self._conn.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def episodic_rows(
        self, session_id: str, *, include_folded: bool = False, after_id: int = 0
    ) -> list[EpisodicRow]:
        query = "SELECT * FROM mem_episodic WHERE session_id = ? AND id > ?"
        if not include_folded:
            query += " AND folded = 0"
        rows = self._conn.execute(query + " ORDER BY id", (session_id, after_id))
        return [
            EpisodicRow(
                id=r["id"], session_id=r["session_id"], turn_id=r["turn_id"],
                role=r["role"], content=r["content"], folded=bool(r["folded"]),
            )
            for r in rows
        ]

    def fold_rows(self, ids: Sequence[int]) -> None:
        self._conn.executemany(
            "UPDATE mem_episodic SET folded = 1 WHERE id = ?", [(i,) for i in ids]
        )
        self._conn.commit()

    def get_summary(self, session_id: str) -> str:
        row = self._conn.execute(
            "SELECT summary FROM mem_summary WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row["summary"] if row else ""

    def set_summary(self, session_id: str, summary: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO mem_summary VALUES (?, ?)", (session_id, summary)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # long-term: facts
    # ------------------------------------------------------------------

    def insert_fact(
        self,
        *,
        text: str,
        type: str,
        confidence: float,
        quote: str,
        session_id: str,
        turn_id: str,
        embedding: Sequence[float],
    ) -> int:
        now = self._now()
        cursor = self._conn.execute(
            "INSERT INTO mem_facts (text, type, confidence, last_session_id,"
            " source_quote, source_session, source_turn, created_at, last_accessed)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (text, type, confidence, session_id, quote, session_id, turn_id, now, now),
        )
        fact_id = cursor.lastrowid
        self._conn.execute(
            "INSERT INTO mem_facts_vec (rowid, embedding) VALUES (?, ?)",
            (fact_id, serialize_float32(list(embedding))),
        )
        self._conn.commit()
        assert fact_id is not None
        return fact_id

    def merge_mention(self, fact_id: int, session_id: str, confidence: float) -> None:
        """Dedup hit: same fact seen again — recency, support, sessions."""
        row = self._get_row(fact_id)
        new_session = session_id != row["last_session_id"]
        self._conn.execute(
            "UPDATE mem_facts SET support_count = support_count + 1,"
            " session_count = session_count + ?, last_session_id = ?,"
            " confidence = MAX(confidence, ?), last_accessed = ? WHERE id = ?",
            (1 if new_session else 0, session_id, confidence, self._now(), fact_id),
        )
        self._conn.commit()

    def set_status(self, fact_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE mem_facts SET status = ? WHERE id = ?", (status, fact_id)
        )
        self._conn.commit()

    def touch_facts(self, ids: Sequence[int]) -> None:
        now = self._now()
        self._conn.executemany(
            "UPDATE mem_facts SET last_accessed = ? WHERE id = ?",
            [(now, i) for i in ids],
        )
        self._conn.commit()

    def get_fact(self, fact_id: int) -> FactRecord:
        return self._to_record(self._get_row(fact_id))

    def nearest_fact(self, embedding: Sequence[float]) -> tuple[int, float] | None:
        """(fact_id, cosine_similarity) of the closest fact of ANY status."""
        row = self._conn.execute(
            "SELECT rowid, distance FROM mem_facts_vec WHERE embedding MATCH ?"
            " AND k = 1",
            (serialize_float32(list(embedding)),),
        ).fetchone()
        if row is None:
            return None
        return row["rowid"], 1.0 - row["distance"]

    def knn_active_facts(
        self, embedding: Sequence[float], n: int
    ) -> list[tuple[FactRecord, float]]:
        rows = self._conn.execute(
            "SELECT rowid, distance FROM mem_facts_vec WHERE embedding MATCH ?"
            " AND k = ?",
            (serialize_float32(list(embedding)), n),
        ).fetchall()
        results = []
        for row in rows:
            fact_row = self._get_row(row["rowid"])
            if fact_row["status"] == "active":
                results.append((self._to_record(fact_row), 1.0 - row["distance"]))
        return results

    def active_facts(self) -> list[FactRecord]:
        rows = self._conn.execute("SELECT * FROM mem_facts WHERE status = 'active'")
        return [self._to_record(r) for r in rows]

    def fact_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM mem_facts").fetchone()[0]

    # ------------------------------------------------------------------
    # KV
    # ------------------------------------------------------------------

    def kv_get(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM mem_kv WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO mem_kv VALUES (?, ?)", (key, value))
        self._conn.commit()

    # ------------------------------------------------------------------

    def _get_row(self, fact_id: int) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM mem_facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no fact with id {fact_id}")
        return row

    @staticmethod
    def _to_record(row: sqlite3.Row) -> FactRecord:
        return FactRecord(
            id=row["id"], text=row["text"], type=row["type"], status=row["status"],
            confidence=row["confidence"], support_count=row["support_count"],
            session_count=row["session_count"], last_session_id=row["last_session_id"],
            source_quote=row["source_quote"], source_session=row["source_session"],
            source_turn=row["source_turn"], last_accessed=row["last_accessed"],
        )
