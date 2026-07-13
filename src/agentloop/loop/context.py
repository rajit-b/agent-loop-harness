"""TurnContext (the working-memory buffer) and prompt assembly.

build_system_prompt is the one place the §11 injection order is coded;
later phases extend it (long-term memory facts, skill index, skill
bodies) without the loop changing shape.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from agentloop.loop.budgets import BudgetTracker
from agentloop.types import Message

TurnStatus = str  # "completed" | "budget_exceeded" | "error" | "cancelled" | "vetoed"


def build_system_prompt(
    intent: str,
    persona: str = "",
    *,
    memory_block: str = "",
    skill_index: str = "",
    skill_bodies: tuple[tuple[str, str], ...] = (),
) -> str:
    # §11 order (fixed): intent → persona → long-term memory facts
    #                    → skill index → selected skill bodies
    parts = [intent.strip()]
    if persona.strip():
        parts.append(persona.strip())
    if memory_block:
        parts.append(memory_block)
    if skill_index:
        parts.append(skill_index)
    for name, body in skill_bodies:
        parts.append(f"## Active skill: {name}\n\n{body.strip()}")
    return "\n\n".join(parts)


@dataclass(slots=True)
class TurnContext:
    """Mutable state of one turn. The messages list IS working memory —
    ephemeral, never persisted beyond the trace (§11)."""

    user_input: str
    budgets: BudgetTracker
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[Message] = field(default_factory=list)
    budget_reason: str | None = None  # set when a budget forced wrap-up
    status: TurnStatus = "completed"
    error: str | None = None
    seq: int = 0
    # prompt components stashed by PERCEIVE so RETRIEVE can rebuild the
    # system message with the recalled-memory block (§11 order)
    skill_index: str = ""
    skill_bodies: tuple[tuple[str, str], ...] = ()

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    @property
    def final_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "assistant" and message.text:
                return message.text
        return ""
