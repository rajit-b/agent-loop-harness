"""The Phase 10 gate: byte-identical replay, pointed divergence."""

from __future__ import annotations

import pytest

from agentloop.loop.machine import AgentLoop
from agentloop.observability.events import comparable
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
)
from agentloop.observability.sinks import JsonlWriter, read_events
from agentloop.tools.builtin.echo import register_echo
from agentloop.tools.executor import ToolRegistry
from agentloop.types import ToolCall

from ..loop.conftest import ScriptedProvider, echo_call, result
from ..rag.conftest import VocabEmbedder

INTENT = "Answer questions about the codebase."


def scripted() -> ScriptedProvider:
    return ScriptedProvider(
        [result(tool_calls=(echo_call("hello"),)), result("The answer is hello.")]
    )


def registry() -> ToolRegistry:
    reg = ToolRegistry()
    register_echo(reg)
    return reg


async def record_run(tmp_path, *, user_input: str = "what echoes?") -> JsonlWriter:
    writer = JsonlWriter(tmp_path / "runs", run_id="recorded")
    loop = AgentLoop(
        RecordingProvider(scripted(), writer),
        RecordingExecutor(registry(), writer),
        intent=INTENT,
        emitter=writer,
        clock=RecordingClock(__import__("time").monotonic, writer),
    )
    turn = await loop.run_turn(user_input, turn_id="turn-1")
    assert turn.status == "completed"
    writer.close()
    return writer


class TestByteIdenticalReplay:
    async def test_recorded_run_replays_byte_identically(self, tmp_path):
        recorded_writer = await record_run(tmp_path)
        recorded = read_events(recorded_writer.path)

        log = ReplayLog(recorded)
        replay_writer = JsonlWriter(tmp_path / "replays", run_id="replayed")
        loop = AgentLoop(
            ReplayProvider(log, replay_writer),
            ReplayExecutor(log, replay_writer),
            intent=INTENT,
            emitter=replay_writer,
            clock=ReplayClock(log, replay_writer),
        )
        turn = await loop.run_turn("what echoes?", turn_id="turn-1")
        replay_writer.close()

        # identical behavior: same result...
        assert turn.status == "completed"
        assert turn.text == "The answer is hello."
        # ...and the full (kind, payload) stream matches event for event
        replayed = read_events(replay_writer.path)
        assert comparable(replayed) == comparable(recorded)
        # every recorded boundary was consumed
        assert log.remaining() == {}

    async def test_replay_needs_no_live_backends(self, tmp_path):
        """The replay stubs ARE the providers: nothing real is touched."""
        recorded = read_events((await record_run(tmp_path)).path)
        log = ReplayLog(recorded)
        writer = JsonlWriter(tmp_path / "replays2", run_id="r2")
        loop = AgentLoop(
            ReplayProvider(log, writer),
            ReplayExecutor(log, writer),
            intent=INTENT,
            emitter=writer,
            clock=ReplayClock(log, writer),
        )
        turn = await loop.run_turn("what echoes?", turn_id="turn-1")
        assert turn.usage.input_tokens > 0  # usage came from the recording


class TestDivergence:
    async def test_changed_behavior_diverges_with_pointed_assertion(self, tmp_path):
        """The 'deliberate code change': the prompt the code builds no
        longer matches what was recorded."""
        recorded = read_events((await record_run(tmp_path)).path)
        log = ReplayLog(recorded)
        writer = JsonlWriter(tmp_path / "replays", run_id="diverged")
        loop = AgentLoop(
            ReplayProvider(log, writer),
            ReplayExecutor(log, writer),
            intent=INTENT,
            persona="Be terse now.",  # ← the code change
            emitter=writer,
            clock=ReplayClock(log, writer),
        )
        turn = await loop.run_turn("what echoes?", turn_id="turn-1")
        # the loop surfaces it as an errored turn carrying the divergence
        assert turn.status == "error"
        assert turn.error is not None
        assert "model call #1 diverged" in turn.error
        assert "first difference at" in turn.error
        assert "Be terse now." in turn.error  # the offending live value, named

    async def test_extra_calls_beyond_recording_diverge(self, tmp_path):
        recorded = read_events((await record_run(tmp_path)).path)
        log = ReplayLog(recorded)
        writer = JsonlWriter(tmp_path / "replays", run_id="extra")
        provider = ReplayProvider(log, writer)
        loop = AgentLoop(
            provider, ReplayExecutor(log, writer), intent=INTENT,
            emitter=writer, clock=ReplayClock(log, writer),
        )
        await loop.run_turn("what echoes?", turn_id="turn-1")
        from agentloop.models.protocol import CompletionRequest
        from agentloop.types import Message

        with pytest.raises(ReplayDivergence, match="more record.model calls"):
            await provider.complete(
                CompletionRequest(messages=(Message.user("one more"),))
            )

    async def test_tool_argument_divergence_is_pointed(self, tmp_path):
        recorded = read_events((await record_run(tmp_path)).path)
        log = ReplayLog(recorded)
        writer = JsonlWriter(tmp_path / "replays", run_id="tooldiv")
        executor = ReplayExecutor(log, writer)
        executor.specs()
        with pytest.raises(ReplayDivergence) as exc_info:
            await executor.execute(
                ToolCall(id="call_0", name="echo", arguments={"text": "TAMPERED"})
            )
        message = str(exc_info.value)
        assert "tool call #1 diverged" in message
        assert "TAMPERED" in message and "hello" in message  # both sides named


class TestEmbedderRecordReplay:
    async def test_round_trip_and_divergence(self, tmp_path):
        from ..loop.conftest import RecordingEmitter

        emitter = RecordingEmitter()
        recording = RecordingEmbedder(VocabEmbedder(), emitter)
        original = await recording.embed(["config loader", "budgets"])

        from agentloop.observability.events import TraceEvent

        events = [
            TraceEvent(run_id="r", seq=i + 1, ts=0.0, kind=kind, payload=payload)
            for i, (kind, payload) in enumerate(emitter.events)
        ]
        replay = ReplayEmbedder(ReplayLog(events), RecordingEmitter())
        assert await replay.embed(["config loader", "budgets"]) == original

        replay2 = ReplayEmbedder(ReplayLog(events), RecordingEmitter())
        with pytest.raises(ReplayDivergence, match="embedding call #1"):
            await replay2.embed(["different text"])
