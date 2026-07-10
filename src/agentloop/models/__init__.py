"""Model abstraction: Protocol, adapters, fallback, pricing (Phase 2)."""

from agentloop.models.fallback import FallbackChain
from agentloop.models.pricing import BUILTIN_PRICING, PricingTable
from agentloop.models.protocol import (
    CompletionRequest,
    CompletionResult,
    ModelProvider,
    StopReason,
    StreamCompleted,
    StreamEvent,
    TextDelta,
)
from agentloop.models.registry import build_chain, build_provider

__all__ = [
    "BUILTIN_PRICING",
    "CompletionRequest",
    "CompletionResult",
    "FallbackChain",
    "ModelProvider",
    "PricingTable",
    "StopReason",
    "StreamCompleted",
    "StreamEvent",
    "TextDelta",
    "build_chain",
    "build_provider",
]
