#!/usr/bin/env python
"""Run the research-digest agent via the programmatic API.

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/research-digest/run.py

Builds the agent from agent.manifest.yaml (Claude Sonnet + the digest skill +
the Sonnet-backed summarizer plugin) and digests a sample document. Pass your
own text as the first CLI argument to digest something else.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agentloop import Agent
from agentloop.observability.sinks import JsonlWriter

MANIFEST = Path(__file__).with_name("agent.manifest.yaml")

SAMPLE = """
Subject: Q3 planning sync — notes

We agreed to ship the new billing dashboard by the end of Q3. Priya will lead
the frontend and needs one more engineer; Marcus offered to help part-time.
The main risk is the payments migration, which is blocked on the vendor's API
review — Dana is chasing them and expects an answer by Friday. We decided to
cut the CSV-export feature from this release to protect the date; it moves to
Q4. Finance flagged that the current pricing table is out of date in three
places, so someone needs to reconcile it before launch. Next sync is Thursday.
""".strip()


async def main() -> int:
    text = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    trace = JsonlWriter(Path(__file__).with_name("runs"))
    async with await Agent.from_manifest(MANIFEST, emitter=trace) as agent:
        turn = await agent.run(f"Give me a digest of this:\n\n{text}")
    trace.close()

    print(turn.text)
    print(
        f"\n[status={turn.status}  steps={turn.steps}  "
        f"tokens={turn.usage.total_tokens}  cost=${turn.cost.usd}  "
        f"trace={trace.path}]"
    )
    return 0 if turn.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
