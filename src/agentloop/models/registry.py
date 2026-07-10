"""Build providers and fallback chains from manifest ModelConfig.

API keys and endpoints come from the environment by design (never the
manifest): OLLAMA_HOST, ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL. A `secrets`
mapping can replace os.environ for tests. Missing credentials fail at
build time, not on the first request.

The manifest's shared `model.params` apply to every provider in the chain;
provider-specific tuning belongs in per-model pricing/params overrides
when a real need appears (deliberate v1 simplification).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from agentloop.config.manifest import ModelConfig, ModelRef
from agentloop.models.adapters.anthropic import AnthropicProvider
from agentloop.models.adapters.ollama import DEFAULT_BASE_URL as OLLAMA_DEFAULT_URL
from agentloop.models.adapters.ollama import OllamaProvider
from agentloop.models.fallback import FallbackChain
from agentloop.models.pricing import PricingTable
from agentloop.models.protocol import ModelProvider
from agentloop.types import ConfigError, TraceEmitter

KNOWN_PROVIDERS = ("ollama", "anthropic")  # grows as adapters land


def build_provider(
    provider: str,
    model: str,
    *,
    params: dict[str, Any] | None = None,
    pricing: PricingTable | None = None,
    secrets: Mapping[str, str] | None = None,
) -> ModelProvider:
    secrets = os.environ if secrets is None else secrets
    if provider == "ollama":
        base_url = secrets.get("OLLAMA_HOST", OLLAMA_DEFAULT_URL)
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        return OllamaProvider(model, base_url=base_url, params=params, pricing=pricing)
    if provider == "anthropic":
        api_key = secrets.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set; the anthropic provider needs it "
                "(API keys live in the environment, never the manifest)"
            )
        return AnthropicProvider(
            model,
            api_key=api_key,
            base_url=secrets.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            params=params,
            pricing=pricing,
        )
    raise ConfigError(
        f"unknown model provider {provider!r}; known: {', '.join(KNOWN_PROVIDERS)}"
    )


def build_chain(
    config: ModelConfig,
    *,
    secrets: Mapping[str, str] | None = None,
    emitter: TraceEmitter | None = None,
    retries: int = 2,
) -> FallbackChain:
    pricing = PricingTable(config.pricing)
    refs = [ModelRef(provider=config.provider, name=config.name), *config.fallback]
    providers = [
        build_provider(
            ref.provider, ref.name, params=dict(config.params), pricing=pricing,
            secrets=secrets,
        )
        for ref in refs
    ]
    return FallbackChain(providers, retries=retries, emitter=emitter)
