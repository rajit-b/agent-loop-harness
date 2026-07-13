"""Manifest hooks{} → bus, via dotted handler paths."""

from __future__ import annotations

import pytest

from agentloop.config.manifest import HookEntry
from agentloop.hooks.bus import HookBus
from agentloop.hooks.contract import PreToolPayload
from agentloop.hooks.loader import install_manifest_hooks, resolve_handler
from agentloop.types import ConfigError, ToolCall


class TestResolveHandler:
    def test_colon_form(self):
        handler = resolve_handler("tests.hooks.sample_handlers:redact_secrets")
        assert callable(handler)

    def test_dotted_form(self):
        handler = resolve_handler("tests.hooks.sample_handlers.redact_secrets")
        assert callable(handler)

    def test_missing_module(self):
        with pytest.raises(ConfigError, match="cannot import"):
            resolve_handler("no.such.module:handler")

    def test_missing_attribute(self):
        with pytest.raises(ConfigError, match="no attribute"):
            resolve_handler("tests.hooks.sample_handlers:nonexistent")

    def test_non_callable(self):
        with pytest.raises(ConfigError, match="not callable"):
            resolve_handler("tests.hooks.sample_handlers:not_callable")


class TestInstall:
    async def test_manifest_hook_runs_with_its_config(self):
        bus = HookBus()
        install_manifest_hooks(
            bus,
            {
                "pre_tool": (
                    HookEntry(
                        handler="tests.hooks.sample_handlers:redact_secrets",
                        priority=10,
                        config={"patterns": ["sk-secret-123"]},
                    ),
                )
            },
        )
        payload = PreToolPayload(
            call=ToolCall(
                id="c1", name="jira__issue_lookup",
                arguments={"query": "auth uses sk-secret-123 ok?"},
            )
        )
        outcome = await bus.dispatch("pre_tool", payload)
        assert outcome.payload.call.arguments["query"] == "auth uses [REDACTED] ok?"

    def test_priority_and_name_carried_over(self):
        bus = HookBus()
        install_manifest_hooks(
            bus,
            {
                "pre_tool": (
                    HookEntry(
                        handler="tests.hooks.sample_handlers:redact_secrets",
                        priority=7,
                    ),
                )
            },
        )
        [hook] = bus.hooks_for("pre_tool")
        assert hook.priority == 7
        assert hook.name == "tests.hooks.sample_handlers:redact_secrets"
