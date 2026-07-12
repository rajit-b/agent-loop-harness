"""Budget tracking (§7): step cap, token, wall-clock, and cost budgets.

Checked at transition boundaries by the state machine, never inside
states. The clock is injectable so wall-clock tests are deterministic.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from decimal import Decimal

from agentloop.config.manifest import LimitsConfig
from agentloop.types import Cost, Usage


class BudgetTracker:
    def __init__(
        self, limits: LimitsConfig, *, clock: Callable[[], float] = time.monotonic
    ):
        self._limits = limits
        self._clock = clock
        self._started = clock()
        self.steps = 0  # PLAN entries
        self.usage = Usage()
        self.cost = Cost(usd=Decimal("0"))

    def record_plan_entry(self) -> None:
        self.steps += 1

    def add(self, usage: Usage, cost: Cost) -> None:
        self.usage = self.usage + usage
        self.cost = self.cost + cost

    def elapsed(self) -> float:
        return self._clock() - self._started

    def exceeded(self, *, entering_plan: bool = False) -> str | None:
        """First exhausted budget, or None. The step cap only blocks a
        *new* PLAN entry; other budgets bind at every boundary."""
        if entering_plan and self.steps >= self._limits.max_steps:
            return "max_steps"
        if self.usage.total_tokens > self._limits.max_tokens:
            return "max_tokens"
        if self.elapsed() > self._limits.max_wall_clock_s:
            return "max_wall_clock_s"
        if self.cost.usd > self._limits.max_cost_usd:
            return "max_cost_usd"
        return None
