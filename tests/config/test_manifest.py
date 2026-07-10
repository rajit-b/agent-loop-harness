"""Manifest validation: valid and invalid shapes (Phase 1 gate)."""

from __future__ import annotations

import pydantic
import pytest

from agentloop.config import Manifest, parse_manifest, read_manifest_file
from agentloop.types import ConfigError

from .conftest import ANNOTATED_EXAMPLE


class TestValid:
    def test_minimal(self, minimal):
        m = parse_manifest(minimal)
        assert m.intent == "Answer questions about the codebase."
        assert m.model.provider == "ollama"
        # defaults materialize
        assert m.limits.max_steps == 16
        assert m.tools.allowlist == ()  # deny all by default

    def test_annotated_example(self):
        m = parse_manifest(read_manifest_file(ANNOTATED_EXAMPLE))
        assert [a.name for a in m.agents] == ["qa", "reviewer"]
        assert m.hooks["pre_tool"][0].priority == 10
        assert m.rag.reranker is None

    def test_fallback_string_shorthand(self, minimal):
        minimal["model"]["fallback"] = ["anthropic/claude-sonnet-5"]
        m = parse_manifest(minimal)
        assert m.model.fallback[0].provider == "anthropic"
        assert m.model.fallback[0].name == "claude-sonnet-5"

    def test_fallback_name_may_contain_slash(self, minimal):
        minimal["model"]["fallback"] = ["ollama/library/llama3"]
        m = parse_manifest(minimal)
        assert m.model.fallback[0].name == "library/llama3"

    def test_manifest_is_frozen(self, minimal):
        m = parse_manifest(minimal)
        with pytest.raises(pydantic.ValidationError):
            m.intent = "mutated"  # type: ignore[misc]


class TestInvalid:
    def _expect(self, data, *needles: str):
        with pytest.raises(ConfigError) as exc_info:
            parse_manifest(data)
        message = str(exc_info.value)
        for needle in needles:
            assert needle in message, f"{needle!r} not in:\n{message}"

    def test_missing_intent(self, minimal):
        del minimal["intent"]
        self._expect(minimal, "intent", "required")

    def test_blank_intent(self, minimal):
        minimal["intent"] = "   "
        self._expect(minimal, "intent", "non-empty")

    def test_unknown_top_level_key(self, minimal):
        minimal["modle"] = {}  # typo must fail loudly, not be ignored
        self._expect(minimal, "modle", "Extra inputs")

    def test_unquoted_version_float(self, minimal):
        minimal["version"] = 1.0  # YAML `version: 1.0` without quotes
        self._expect(minimal, "version")

    def test_unsupported_major_version(self, minimal):
        minimal["version"] = "2.0"
        self._expect(minimal, "version")

    def test_stdio_requires_command(self, minimal):
        minimal["tools"] = {"mcp_servers": [{"name": "fs", "transport": "stdio"}]}
        self._expect(minimal, "stdio transport requires 'command'")

    def test_http_requires_url(self, minimal):
        minimal["tools"] = {"mcp_servers": [{"name": "j", "transport": "http"}]}
        self._expect(minimal, "http transport requires 'url'")

    def test_http_rejects_command(self, minimal):
        minimal["tools"] = {
            "mcp_servers": [
                {"name": "j", "transport": "http", "url": "http://x", "command": "rm"}
            ]
        }
        self._expect(minimal, "http transport does not take")

    def test_duplicate_server_names(self, minimal):
        server = {"name": "fs", "transport": "stdio", "command": "x"}
        minimal["tools"] = {"mcp_servers": [server, dict(server)]}
        self._expect(minimal, "duplicate MCP server names", "fs")

    def test_duplicate_agent_names(self, minimal):
        minimal["agents"] = [{"name": "qa"}, {"name": "qa"}]
        self._expect(minimal, "duplicate agent names", "qa")

    def test_unknown_hook_event(self, minimal):
        minimal["hooks"] = {"pre_flight": [{"handler": "x.y"}]}
        self._expect(minimal, "pre_flight")

    def test_bad_handler_path(self, minimal):
        minimal["hooks"] = {"pre_tool": [{"handler": "not a path!"}]}
        self._expect(minimal, "handler")

    def test_bad_fallback_string(self, minimal):
        minimal["model"]["fallback"] = ["no-slash-here"]
        self._expect(minimal, "provider/name")

    def test_negative_limit(self, minimal):
        minimal["limits"] = {"max_steps": 0}
        self._expect(minimal, "max_steps", "greater than 0")

    def test_chunk_overlap_must_be_smaller_than_size(self, minimal):
        minimal["rag"] = {"chunk": {"size": 100, "overlap": 100}}
        self._expect(minimal, "overlap must be smaller")


class TestFileLoading:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            read_manifest_file(tmp_path / "nope.yaml")

    def test_malformed_yaml(self, write_manifest):
        path = write_manifest("intent: [unclosed")
        with pytest.raises(ConfigError, match="malformed YAML"):
            read_manifest_file(path)

    def test_non_mapping_root(self, write_manifest):
        path = write_manifest("- just\n- a list\n")
        with pytest.raises(ConfigError, match="must be a mapping"):
            read_manifest_file(path)

    def test_json_schema_marks_required_fields(self):
        schema = Manifest.model_json_schema()
        assert set(schema["required"]) == {"version", "intent", "model"}
