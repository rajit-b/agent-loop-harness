"""Pricing table and cost computation (A9).

Rates are USD per million tokens, cached from the provider price lists on
2026-07-10 (Anthropic rates via the platform docs; Sonnet 5 uses sticker
rates, not the time-limited introductory pricing). Manifest `model.pricing`
entries override these. Local providers (ollama) cost $0.00 but their
tokens still count against budgets.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from agentloop.config.manifest import PricingEntry
from agentloop.types import Cost, Usage

FREE_PROVIDERS = frozenset({"ollama"})

_MTOK = Decimal(1_000_000)


def _entry(inp: str, out: str, cache_read: str) -> PricingEntry:
    return PricingEntry(
        input_per_mtok=Decimal(inp),
        output_per_mtok=Decimal(out),
        cache_read_per_mtok=Decimal(cache_read),
    )


# Cache reads are ~0.1x input price on Anthropic.
BUILTIN_PRICING: dict[str, PricingEntry] = {
    "anthropic/claude-fable-5": _entry("10.00", "50.00", "1.00"),
    "anthropic/claude-opus-4-8": _entry("5.00", "25.00", "0.50"),
    "anthropic/claude-opus-4-7": _entry("5.00", "25.00", "0.50"),
    "anthropic/claude-opus-4-6": _entry("5.00", "25.00", "0.50"),
    "anthropic/claude-sonnet-5": _entry("3.00", "15.00", "0.30"),
    "anthropic/claude-sonnet-4-6": _entry("3.00", "15.00", "0.30"),
    "anthropic/claude-haiku-4-5": _entry("1.00", "5.00", "0.10"),
}


class PricingTable:
    """Resolves rates (overrides > builtin > free-provider rule) into Cost."""

    def __init__(self, overrides: Mapping[str, PricingEntry] | None = None):
        self._overrides = dict(overrides or {})

    def lookup(self, provider: str, model: str) -> PricingEntry | None:
        key = f"{provider}/{model}"
        return self._overrides.get(key) or BUILTIN_PRICING.get(key)

    def cost_for(self, provider: str, model: str, usage: Usage) -> Cost:
        entry = self.lookup(provider, model)
        if entry is not None:
            usd = (
                Decimal(usage.input_tokens) * entry.input_per_mtok
                + Decimal(usage.output_tokens) * entry.output_per_mtok
                + Decimal(usage.cache_read_tokens) * entry.cache_read_per_mtok
            ) / _MTOK
            return Cost(usd=usd, known=True)
        if provider in FREE_PROVIDERS:
            return Cost(usd=Decimal("0"), known=True)
        return Cost(usd=Decimal("0"), known=False)  # unknown model: don't invent a price
