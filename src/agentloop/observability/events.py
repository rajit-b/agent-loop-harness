"""The TraceEvent envelope (§4/§12).

envelope.run_id identifies the recorded run (one JSONL file); the loop's
per-turn id travels inside payloads (as `run_id` on loop events — one
run may span several turns). seq is monotonic per writer; ts is wall
time and is masked out of behavioral comparisons. span ids are carried
for forward compatibility; nothing sets them yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    seq: int
    ts: float
    kind: str
    payload: dict[str, Any]
    span_id: str | None = None
    parent_span_id: str | None = None


def comparable(events: Sequence[TraceEvent]) -> list[tuple[str, dict[str, Any]]]:
    """The behavioral projection of a stream: (kind, payload) with the
    envelope's timing metadata masked. Two runs are byte-identical iff
    their projections are equal."""
    return [(event.kind, event.payload) for event in events]
