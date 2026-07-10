"""Resolution order and provenance: defaults < manifest < env < CLI (A3)."""

from __future__ import annotations

import pytest

from agentloop.config import load_config
from agentloop.types import ConfigError


@pytest.fixture
def manifest_path(minimal, write_manifest):
    minimal["model"]["params"] = {"temperature": 0.2}
    minimal["tools"] = {"allowlist": ["fs.read_*"]}
    return write_manifest(minimal)


class TestPrecedence:
    def test_manifest_only(self, manifest_path):
        cfg = load_config(manifest_path, env={})
        assert cfg.manifest.model.provider == "ollama"

    def test_env_overrides_manifest(self, manifest_path):
        cfg = load_config(manifest_path, env={"AGENTLOOP_MODEL__PROVIDER": "openai"})
        assert cfg.manifest.model.provider == "openai"

    def test_cli_overrides_env(self, manifest_path):
        cfg = load_config(
            manifest_path,
            env={"AGENTLOOP_MODEL__PROVIDER": "openai"},
            cli_overrides={"model.provider": "anthropic"},
        )
        assert cfg.manifest.model.provider == "anthropic"

    def test_env_values_are_typed(self, manifest_path):
        cfg = load_config(
            manifest_path,
            env={
                "AGENTLOOP_LIMITS__MAX_STEPS": "42",
                "AGENTLOOP_MEMORY__ENABLED": "false",
            },
        )
        assert cfg.manifest.limits.max_steps == 42
        assert cfg.manifest.memory.enabled is False

    def test_lists_replace_wholesale(self, manifest_path):
        cfg = load_config(
            manifest_path, env={"AGENTLOOP_TOOLS__ALLOWLIST": "[ripgrep.*, jira.issue_lookup]"}
        )
        # manifest's fs.read_* is gone, not merged
        assert cfg.manifest.tools.allowlist == ("ripgrep.*", "jira.issue_lookup")

    def test_mappings_deep_merge(self, manifest_path):
        cfg = load_config(manifest_path, env={"AGENTLOOP_MODEL__PARAMS__TOP_P": "0.9"})
        # env adds top_p without clobbering the manifest's temperature
        assert cfg.manifest.model.params == {"temperature": 0.2, "top_p": 0.9}

    def test_unrelated_env_ignored(self, manifest_path):
        cfg = load_config(manifest_path, env={"PATH": "/usr/bin", "HOME": "/Users/x"})
        assert cfg.manifest.model.provider == "ollama"


class TestFailClosed:
    def test_env_addressing_unknown_field_fails(self, manifest_path):
        with pytest.raises(ConfigError, match="max_stepz"):
            load_config(manifest_path, env={"AGENTLOOP_LIMITS__MAX_STEPZ": "5"})

    def test_cli_addressing_unknown_field_fails(self, manifest_path):
        with pytest.raises(ConfigError, match="nonsense"):
            load_config(manifest_path, env={}, cli_overrides={"nonsense": "1"})

    def test_override_producing_invalid_value_fails(self, manifest_path):
        with pytest.raises(ConfigError, match="max_steps"):
            load_config(manifest_path, env={"AGENTLOOP_LIMITS__MAX_STEPS": "-1"})


class TestProvenance:
    def test_all_four_origins(self, manifest_path):
        cfg = load_config(
            manifest_path,
            env={"AGENTLOOP_LIMITS__MAX_STEPS": "42"},
            cli_overrides={"model.provider": "anthropic"},
        )
        assert cfg.origin("model.name") == "manifest"
        assert cfg.origin("limits.max_steps") == "env"
        assert cfg.origin("model.provider") == "cli"
        assert cfg.origin("limits.max_tokens") == "default"

    def test_later_layer_wins_provenance(self, manifest_path):
        cfg = load_config(
            manifest_path,
            env={"AGENTLOOP_MODEL__PROVIDER": "openai"},
            cli_overrides={"model.provider": "anthropic"},
        )
        assert cfg.origin("model.provider") == "cli"

    def test_wholesale_list_provenance(self, manifest_path):
        cfg = load_config(manifest_path, env={"AGENTLOOP_TOOLS__ALLOWLIST": "[a.b]"})
        assert cfg.origin("tools.allowlist") == "env"
