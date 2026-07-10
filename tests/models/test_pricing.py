"""Cost accounting: exact Decimal math, overrides, the free/unknown split."""

from __future__ import annotations

from decimal import Decimal

from agentloop.config.manifest import PricingEntry
from agentloop.models.pricing import PricingTable
from agentloop.types import Cost, Usage


class TestCostComputation:
    def test_exact_decimal_math(self):
        table = PricingTable()
        usage = Usage(input_tokens=123_456, output_tokens=7_890, cache_read_tokens=50_000)
        cost = table.cost_for("anthropic", "claude-sonnet-5", usage)
        expected = (
            Decimal(123_456) * Decimal("3.00")
            + Decimal(7_890) * Decimal("15.00")
            + Decimal(50_000) * Decimal("0.30")
        ) / Decimal(1_000_000)
        assert cost.usd == expected
        assert cost.known is True

    def test_free_provider_costs_zero_and_is_known(self):
        cost = PricingTable().cost_for("ollama", "whatever:7b", Usage(input_tokens=10**6))
        assert cost.usd == 0 and cost.known is True

    def test_unknown_model_is_flagged_not_invented(self):
        cost = PricingTable().cost_for("anthropic", "claude-99", Usage(input_tokens=100))
        assert cost.usd == 0 and cost.known is False

    def test_manifest_override_wins_over_builtin(self):
        table = PricingTable(
            {
                "anthropic/claude-sonnet-5": PricingEntry(
                    input_per_mtok=Decimal("2.00"), output_per_mtok=Decimal("10.00")
                )
            }
        )
        cost = table.cost_for(
            "anthropic", "claude-sonnet-5", Usage(input_tokens=1_000_000)
        )
        assert cost.usd == Decimal("2.00")

    def test_override_can_price_local_models(self):
        table = PricingTable(
            {"ollama/big:70b": PricingEntry(
                input_per_mtok=Decimal("0.10"), output_per_mtok=Decimal("0.10"))}
        )
        cost = table.cost_for("ollama", "big:70b", Usage(output_tokens=1_000_000))
        assert cost.usd == Decimal("0.10")


class TestAccumulation:
    def test_usage_addition(self):
        total = Usage(input_tokens=10, output_tokens=5) + Usage(
            input_tokens=1, output_tokens=2, cache_read_tokens=3
        )
        assert (total.input_tokens, total.output_tokens, total.cache_read_tokens) == (11, 7, 3)
        assert total.total_tokens == 21

    def test_cost_addition_propagates_unknown(self):
        known = Cost(usd=Decimal("1.50"))
        unknown = Cost(usd=Decimal("0"), known=False)
        assert (known + known).usd == Decimal("3.00")
        assert (known + known).known is True
        assert (known + unknown).known is False
