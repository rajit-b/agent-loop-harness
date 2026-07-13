"""Envelope, JSONL sink, run index, tee."""

from __future__ import annotations

import json

from agentloop.observability.events import comparable
from agentloop.observability.sinks import JsonlWriter, RunIndex, TeeEmitter, read_events
from agentloop.storage.sqlite import connect

from ..loop.conftest import RecordingEmitter


class TestJsonlWriter:
    def test_envelope_fields_and_monotonic_seq(self, tmp_path):
        writer = JsonlWriter(tmp_path, run_id="run-1")
        writer.emit("a.b", {"x": 1})
        writer.emit("c.d", {"y": "z"})
        writer.close()
        events = read_events(writer.path)
        assert [e.kind for e in events] == ["a.b", "c.d"]
        assert [e.seq for e in events] == [1, 2]
        assert all(e.run_id == "run-1" for e in events)
        assert all(e.ts > 0 for e in events)

    def test_flushed_per_event(self, tmp_path):
        writer = JsonlWriter(tmp_path, run_id="run-2")
        writer.emit("k", {"n": 1})
        # readable BEFORE close: nothing sits in a buffer
        assert len(read_events(writer.path)) == 1
        writer.close()

    def test_non_json_native_values_serialize(self, tmp_path):
        from decimal import Decimal

        writer = JsonlWriter(tmp_path, run_id="run-3")
        writer.emit("k", {"cost": Decimal("1.50")})
        writer.close()
        [event] = read_events(writer.path)
        assert event.payload["cost"] == "1.50"

    def test_one_line_per_event(self, tmp_path):
        writer = JsonlWriter(tmp_path, run_id="run-4")
        for i in range(5):
            writer.emit("k", {"i": i})
        writer.close()
        lines = writer.path.read_text().strip().splitlines()
        assert len(lines) == 5
        assert all(json.loads(line)["kind"] == "k" for line in lines)


class TestRunIndex:
    def test_close_records_run(self, tmp_path):
        index = RunIndex(connect(":memory:", load_vec=False))
        writer = JsonlWriter(tmp_path, run_id="indexed-run")
        writer.emit("k", {})
        writer.emit("k", {})
        writer.close(index)
        [row] = index.runs()
        assert row["run_id"] == "indexed-run"
        assert row["event_count"] == 2
        assert row["path"].endswith("indexed-run.jsonl")
        assert row["ended_at"] >= row["started_at"]


class TestTee:
    def test_fan_out(self, tmp_path):
        a, b = RecordingEmitter(), RecordingEmitter()
        tee = TeeEmitter(a, b)
        tee.emit("k", {"x": 1})
        assert a.events == b.events == [("k", {"x": 1})]


def test_comparable_masks_only_envelope_metadata(tmp_path):
    writer = JsonlWriter(tmp_path, run_id="r")
    writer.emit("k", {"x": 1})
    writer.close()
    assert comparable(read_events(writer.path)) == [("k", {"x": 1})]
