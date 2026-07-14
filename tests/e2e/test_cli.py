"""CLI entrypoint: config resolution, replay summary, run wiring."""

from __future__ import annotations

import json

import pytest

from agentloop.cli import _parse_overrides, build_parser, main
from agentloop.observability.sinks import JsonlWriter


class TestParsing:
    def test_overrides_parse(self):
        assert _parse_overrides(["a.b=1", "c=hi"]) == {"a.b": "1", "c": "hi"}

    def test_bad_override_exits(self):
        with pytest.raises(SystemExit):
            _parse_overrides(["nokey"])

    def test_run_requires_a_question(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run"])


class TestConfigCommand:
    def test_prints_resolved_config_and_provenance(self, tmp_path, capsys):
        import yaml

        manifest = tmp_path / "m.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "version": "1.0",
                    "intent": "Do things.",
                    "model": {"provider": "ollama", "name": "qwen2.5:14b"},
                }
            )
        )
        code = main(["config", "-m", str(manifest), "--set", "model.provider=openai"])
        assert code == 0
        output = json.loads(capsys.readouterr().out)
        assert output["manifest"]["model"]["provider"] == "openai"  # override applied
        assert output["provenance"]["model.provider"] == "cli"  # provenance tracked

    def test_invalid_manifest_reports_error(self, tmp_path, capsys):
        manifest = tmp_path / "bad.yaml"
        manifest.write_text("intent: [unclosed")
        code = main(["config", "-m", str(manifest)])
        assert code == 2
        assert "error:" in capsys.readouterr().err


class TestReplayCommand:
    def test_summarizes_a_recorded_run(self, tmp_path, capsys):
        writer = JsonlWriter(tmp_path, run_id="cli-run")
        writer.emit("loop.transition", {"to": "plan"})
        writer.emit("loop.transition", {"to": "act"})
        writer.emit("model.complete", {"model": "x"})
        writer.close()
        code = main(["replay", str(writer.path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "cli-run" in out
        assert "events: 3" in out
        assert "loop.transition" in out and "2" in out
