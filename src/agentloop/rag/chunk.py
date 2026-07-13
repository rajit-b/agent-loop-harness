"""Markdown-heading-aware chunking with token-window fallback (§11).

Sizes are in tokens (chars/4 heuristic, consistent with estimate_tokens).
Every chunk carries character offsets into the source document — the
substance of its citation. Invariant: content[start:end] == text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    index: int
    start: int  # char offset into the document
    end: int
    heading: str = ""


def chunk_text(content: str, *, size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Split by markdown sections; oversized sections get sliding windows."""
    max_chars = size * _CHARS_PER_TOKEN
    step = max(1, (size - overlap) * _CHARS_PER_TOKEN)
    chunks: list[Chunk] = []
    for heading, section_start, section_end in _sections(content):
        position = section_start
        while position < section_end:
            window_end = min(position + max_chars, section_end)
            text = content[position:window_end]
            if text.strip():
                chunks.append(
                    Chunk(
                        text=text,
                        index=len(chunks),
                        start=position,
                        end=window_end,
                        heading=heading,
                    )
                )
            if window_end >= section_end:
                break
            position += step
    return chunks


def _sections(content: str) -> list[tuple[str, int, int]]:
    """(heading, start, end) per markdown section; whole doc if no headings."""
    matches = list(_HEADING.finditer(content))
    if not matches:
        return [("", 0, len(content))] if content else []
    sections: list[tuple[str, int, int]] = []
    if matches[0].start() > 0:  # preamble before the first heading
        sections.append(("", 0, matches[0].start()))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append((match.group(1).strip(), match.start(), end))
    return sections
