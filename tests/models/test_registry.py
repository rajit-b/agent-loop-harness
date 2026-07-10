"""Registry: manifest ModelConfig → providers/chain, fail-fast credentials."""

from __future__ import annotations

import pytest

from agentloop.config.manifest import ModelConfig
from agentloop.models.adapters.anthropic import AnthropicProvider
from agentloop.models.adapters.ollama import OllamaProvider
from agentloop.models.registry import build_chain, build_provider
from agentloop.types import ConfigError


class TestBuildProvider:
    def test_ollama_defaults(self):
        provider = build_provider("ollama", "qwen2.5:14b", secrets={})
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "qwen2.5:14b"

    def test_ollama_host_without_scheme_is_normalized(self):
        provider = build_provider(
            "ollama", "m", secrets={"OLLAMA_HOST": "gpu-box:11434"}
        )
        assert provider._base_url == "http://gpu-box:11434"  # noqa: SLF001

    def test_anthropic_requires_api_key(self):
        with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
            build_provider("anthropic", "claude-opus-4-8", secrets={})

    def test_anthropic_with_key(self):
        provider = build_provider(
            "anthropic", "claude-opus-4-8", secrets={"ANTHROPIC_API_KEY": "sk-x"}
        )
        assert isinstance(provider, AnthropicProvider)

    def test_unknown_provider(self):
        with pytest.raises(ConfigError, match="unknown model provider 'openai'"):
            build_provider("openai", "gpt-4o", secrets={})


class TestBuildChain:
    def test_chain_from_manifest_config(self):
        config = ModelConfig(
            provider="ollama",
            name="qwen2.5:14b",
            fallback=("anthropic/claude-sonnet-5",),
        )
        chain = build_chain(config, secrets={"ANTHROPIC_API_KEY": "sk-x"})
        assert chain.provider == "ollama"  # chain fronts its primary
        assert chain.model == "qwen2.5:14b"
        assert len(chain._providers) == 2  # noqa: SLF001

    def test_missing_fallback_credential_fails_at_build_time(self):
        config = ModelConfig(
            provider="ollama", name="m", fallback=("anthropic/claude-sonnet-5",)
        )
        with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
            build_chain(config, secrets={})
