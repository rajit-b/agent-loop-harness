"""Observability: trace envelope, sinks, deterministic replay (Phase 10).

Per §3 rule 3, NO framework component imports this package — everything
emits through the injected TraceEmitter Protocol (types.py). This package
is composed at the application boundary.
"""

from agentloop.observability.events import TraceEvent, comparable
from agentloop.observability.replay import (
    RecordingClock,
    RecordingEmbedder,
    RecordingExecutor,
    RecordingProvider,
    ReplayClock,
    ReplayDivergence,
    ReplayEmbedder,
    ReplayExecutor,
    ReplayLog,
    ReplayProvider,
    first_difference,
    stable_hash,
)
from agentloop.observability.sinks import (
    JsonlWriter,
    RunIndex,
    TeeEmitter,
    read_events,
)

__all__ = [
    "JsonlWriter",
    "RecordingClock",
    "RecordingEmbedder",
    "RecordingExecutor",
    "RecordingProvider",
    "ReplayClock",
    "ReplayDivergence",
    "ReplayEmbedder",
    "ReplayExecutor",
    "ReplayLog",
    "ReplayProvider",
    "RunIndex",
    "TeeEmitter",
    "TraceEvent",
    "comparable",
    "first_difference",
    "read_events",
    "stable_hash",
]
