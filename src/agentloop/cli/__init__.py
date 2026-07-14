"""CLI entrypoint (§ "Ship a CLI"). Thin argparse over the programmatic API.

    agentloop run     -m agent.manifest.yaml "your question"
    agentloop config  -m agent.manifest.yaml        # resolved config + provenance
    agentloop replay  runs/<run_id>.jsonl           # re-run a recorded trace

Config resolution order (manifest → env → CLI) is honored: --set k=v flags
become CLI overrides, AGENTLOOP_* env is read by the loader.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from agentloop.api import Agent
from agentloop.config.resolve import load_config
from agentloop.observability.events import comparable
from agentloop.observability.sinks import JsonlWriter, read_events
from agentloop.types import AgentLoopError


def _parse_overrides(pairs: Sequence[str]) -> dict[str, str]:
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        overrides[key.strip()] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentloop")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one turn against a manifest")
    run.add_argument("question")
    run.add_argument("-m", "--manifest", default="agent.manifest.yaml")
    run.add_argument("-a", "--agent", default=None, help="active agent name")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--storage", default=None, help="durable DB directory")
    run.add_argument("--trace", default=None, help="write a JSONL trace here")

    show = sub.add_parser("config", help="print the resolved config + provenance")
    show.add_argument("-m", "--manifest", default="agent.manifest.yaml")
    show.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    replay = sub.add_parser("replay", help="summarize a recorded run")
    replay.add_argument("trace", help="path to a runs/<id>.jsonl file")

    return parser


async def _run(args: argparse.Namespace) -> int:
    emitter = None
    writer = None
    if args.trace:
        writer = JsonlWriter(Path(args.trace).parent or ".")
        emitter = writer
    agent = await Agent.from_manifest(
        args.manifest,
        agent_name=args.agent,
        cli_overrides=_parse_overrides(args.set),
        emitter=emitter,
        storage_dir=args.storage,
    )
    async with agent:
        turn = await agent.run(args.question)
    if writer is not None:
        writer.close()
    print(turn.text)
    if turn.status != "completed":
        print(f"[status: {turn.status}]", file=sys.stderr)
        if turn.error:
            print(f"[error: {turn.error}]", file=sys.stderr)
    return 0 if turn.status in ("completed", "budget_exceeded") else 1


def _config(args: argparse.Namespace) -> int:
    config = load_config(args.manifest, cli_overrides=_parse_overrides(args.set))
    output = {
        "manifest": config.manifest.model_dump(mode="json"),
        "provenance": config.provenance,
    }
    print(json.dumps(output, indent=2, default=str))
    return 0


def _replay(args: argparse.Namespace) -> int:
    events = read_events(args.trace)
    projection = comparable(events)
    kinds: dict[str, int] = {}
    for kind, _ in projection:
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"run: {events[0].run_id if events else '(empty)'}  events: {len(events)}")
    for kind, count in sorted(kinds.items()):
        print(f"  {count:4d}  {kind}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "config":
            return _config(args)
        if args.command == "replay":
            return _replay(args)
    except AgentLoopError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
