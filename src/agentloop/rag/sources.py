"""Ingest sources (§11). Files/dirs in v1; URL sources are an extension."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

DEFAULT_EXTENSIONS = (".md", ".txt", ".rst")


@dataclass(frozen=True, slots=True)
class RawDocument:
    uri: str
    content: str
    content_hash: str  # sha256 hex of the content


def _make_document(path: Path) -> RawDocument:
    content = path.read_text(encoding="utf-8")
    return RawDocument(
        uri=str(path.resolve()),
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


@runtime_checkable
class Source(Protocol):
    def documents(self) -> Iterator[RawDocument]: ...


class FileSource:
    """Files and directories (recursive), filtered by extension."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    ):
        self._paths = [Path(p) for p in paths]
        self._extensions = {ext.lower() for ext in extensions}

    def documents(self) -> Iterator[RawDocument]:
        for path in self._paths:
            if path.is_file():
                yield _make_document(path)
            elif path.is_dir():
                for file in sorted(path.rglob("*")):
                    if file.is_file() and file.suffix.lower() in self._extensions:
                        yield _make_document(file)
