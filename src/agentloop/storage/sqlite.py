"""Shared SQLite plumbing (§13): connection setup, WAL, capability checks.

A10: FTS5 and loadable-extension support are hard requirements, checked
at connection time with pointed errors — not discovered mid-query.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from agentloop.types import ConfigError


def connect(path: Path | str = ":memory:", *, load_vec: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _assert_fts5(conn)
    if load_vec:
        _load_vec(conn)
    return conn


def _assert_fts5(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
    except sqlite3.OperationalError as exc:
        raise ConfigError(
            "this SQLite build lacks FTS5, which agentloop requires (A10); "
            f"probe failed with: {exc}"
        ) from exc


def _load_vec(conn: sqlite3.Connection) -> None:
    try:
        import sqlite_vec
    except ImportError as exc:
        raise ConfigError(
            "the sqlite-vec package is not installed (pip install sqlite-vec)"
        ) from exc
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError) as exc:
        raise ConfigError(
            "this Python's sqlite3 cannot load extensions, so sqlite-vec is "
            f"unavailable (A10): {exc}"
        ) from exc
