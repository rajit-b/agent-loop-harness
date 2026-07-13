"""Trace sinks (§12): JSONL per run (source of truth, flushed per event)
plus a SQLite run index. Console rendering would be one more sink."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from agentloop.observability.events import TraceEvent
from agentloop.types import TraceEmitter


class JsonlWriter:
    """TraceEmitter writing runs/{run_id}.jsonl, one envelope per line,
    flushed on every event so a crash loses nothing."""

    def __init__(self, directory: Path | str, *, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{self.run_id}.jsonl"
        self._file: TextIO = self.path.open("a", encoding="utf-8")
        self._seq = 0
        self.started_at = time.time()

    def emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        self._seq += 1
        event = TraceEvent(
            run_id=self.run_id, seq=self._seq, ts=time.time(),
            kind=kind, payload=dict(payload),
        )
        self._file.write(json.dumps(event.model_dump(), default=str) + "\n")
        self._file.flush()

    @property
    def event_count(self) -> int:
        return self._seq

    def close(self, index: RunIndex | None = None) -> None:
        self._file.close()
        if index is not None:
            index.record(self)


def read_events(path: Path | str) -> list[TraceEvent]:
    events = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(TraceEvent.model_validate(json.loads(line)))
    return events


class RunIndex:
    """SQLite index over recorded runs — for listing, not replay."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trace_runs ("
            " run_id TEXT PRIMARY KEY, path TEXT, started_at REAL,"
            " ended_at REAL, event_count INTEGER)"
        )
        conn.commit()

    def record(self, writer: JsonlWriter) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO trace_runs VALUES (?, ?, ?, ?, ?)",
            (
                writer.run_id, str(writer.path), writer.started_at,
                time.time(), writer.event_count,
            ),
        )
        self._conn.commit()

    def runs(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM trace_runs ORDER BY started_at"
        ).fetchall()


class TeeEmitter:
    """Fan one event stream out to several sinks."""

    def __init__(self, *emitters: TraceEmitter):
        self._emitters = emitters

    def emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        for emitter in self._emitters:
            emitter.emit(kind, payload)
