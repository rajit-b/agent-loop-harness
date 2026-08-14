# agent-loop-harness

A local-first, model-agnostic **agent loop framework** in Python. One
`agent.manifest.yaml` declares the whole agent — model, tools, skills, plugins,
hooks, RAG, and memory — and the framework assembles and runs it. No cloud
services, no vendor lock-in: the only persistence is SQLite (with
[`sqlite-vec`](https://github.com/asg017/sqlite-vec) and FTS5) plus JSONL trace
files, and every external system sits behind a `typing.Protocol`.

- **Python 3.11+, asyncio-native, fully typed** (Protocols for seams, pydantic
  v2 for every schema).
- **Model-agnostic.** Ollama (default) and Anthropic adapters normalize their
  different tool-calling formats to one internal representation, with ordered
  fallback and per-call cost accounting.
- **Zero infrastructure.** SQLite + `sqlite-vec` + FTS5. Nothing to deploy.
- **Deterministic and replayable.** Every loop step emits a structured trace
  event; a recorded run replays byte-for-byte, and a behavior change surfaces
  as a pointed divergence.

> Status: all layers implemented, **313 passing tests**, with an
> `import-linter` layer contract enforced across the codebase. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

Requires a SQLite build with **FTS5** and **loadable extensions** (the system
Python on macOS/Linux qualifies) — both are checked at startup with a clear
error if missing. For real model/embedding calls you'll want a local
[Ollama](https://ollama.com) (`ollama pull qwen2.5:14b`,
`ollama pull nomic-embed-text`) or an `ANTHROPIC_API_KEY` in the environment.

---

## Quick start

### CLI

```bash
# Run one turn against a manifest
agentloop run -m examples/codebase-qa/agent.manifest.yaml "where is the config loader?"

# Show the fully resolved config with per-value provenance
agentloop config -m examples/codebase-qa/agent.manifest.yaml --set limits.max_steps=5

# Summarize a recorded trace
agentloop replay runs/<run_id>.jsonl
```

Config resolves in the order **defaults → manifest → environment
(`AGENTLOOP_*`) → CLI (`--set key=value`)**, and `agentloop config` reports
where each value came from.

### Programmatic API

```python
from agentloop import Agent

async with await Agent.from_manifest("agent.manifest.yaml") as agent:
    turn = await agent.run("how does the config loader resolve precedence?")
    print(turn.text)          # the answer, with [n] citations
    print(turn.usage, turn.cost, turn.status)
```

Every external backend is injectable (`provider=`, `embedder=`,
`memory_provider=`, `emitter=`), so the same assembly runs against real models
in production and scripted/deterministic stubs in tests.

---

## The manifest

One `agent.manifest.yaml` is the root of all configuration. Abridged:

```yaml
version: "1.0"
intent: >
  Answer questions about this codebase, citing files. Search and read the
  code before answering.

model:
  provider: ollama
  name: qwen2.5:14b
  fallback: [anthropic/claude-sonnet-5]   # ordered fallback chain

agents:
  - name: qa
    persona: You are a precise, citation-driven codebase assistant.

tools:
  mcp_servers:
    - {name: code, transport: stdio, command: mcp-server-filesystem, args: ["--root", "."]}
  allowlist: [code.search, code.read_file, jira.issue_lookup]  # fail closed
  sandbox: {roots: ["."], max_result_chars: 100000}            # path jail

skills:  [../../skills/code-search]
plugins: [{name: jira, source: ../../plugins/jira, version: ">=0.1,<0.2"}]

hooks:
  pre_tool:
    - {handler: codebaseqa_hooks:redact_secrets, priority: 10}

rag:    {sources: ["./docs"], top_k: 4}
memory: {enabled: true}
limits: {max_steps: 12, max_tokens: 200000, max_cost_usd: 1.00}
```

The formal JSON Schema is generated from the pydantic models
(`python -m agentloop.config.schema` → [`src/agentloop/config/schema.json`](src/agentloop/config/schema.json)),
and a fully-annotated example lives at
[`examples/manifests/annotated.manifest.yaml`](examples/manifests/annotated.manifest.yaml).

---

## How it works

The agent loop is an explicit state machine — **perceive → retrieve → plan →
act → observe → reflect → terminate** — with step, token, wall-clock, and cost
budgets checked at every transition boundary. Exceeding any budget forces a
graceful wrap-up. Each subsystem sits in its own layer, and dependencies only
ever point downward (enforced mechanically by `import-linter`):

| Layer | Responsibility |
|---|---|
| **Manifest** (`config/`) | Load, validate, and resolve `agent.manifest.yaml`; generate its JSON Schema. |
| **Model abstraction** (`models/`) | `ModelProvider` Protocol; Ollama + Anthropic adapters; `FallbackChain`; pricing/cost. |
| **Agent loop** (`loop/`) | The seven-state machine, budgets, cancellation, prompt assembly. |
| **Tools / MCP** (`tools/`) | MCP client (stdio + HTTP), allowlist gating (fail closed), path jail, timeouts. |
| **Hooks** (`hooks/`) | `pre/post_model`, `pre/post_tool`, `pre/post_retrieval`, `on_error`, `on_turn_end` — each may mutate or veto. |
| **Skills** (`skills/`) | Directory-based skills with progressive disclosure; explicit → trigger → semantic selection. |
| **Plugins** (`plugins/`) | Packaged extensions whose only power is registration (tools, skills, hooks, CLI); fail-closed with rollback. |
| **RAG** (`rag/`) | Ingest → chunk → embed → store → hybrid retrieve (vector + BM25, RRF fusion) → cited chunks; incremental reindex. |
| **Memory** (`memory/`) | Three tiers — working, short-term (episodic + rolling summary), long-term (semantic facts with consolidation & decay). |
| **Observability** (`observability/`) | Structured trace envelope, JSONL + SQLite sinks, deterministic replay. |

Two design rules are load-bearing:

- **One internal representation.** Provider tool-calling formats
  (Anthropic `tool_use` blocks, OpenAI-dialect `tool_calls`, …) are translated
  only at the adapter boundary and never leak inward.
- **Observability by injection.** No component imports the observability
  package; each receives a `TraceEmitter` Protocol. That is what makes
  substituting recorders for live backends — i.e. deterministic replay — clean.

---

## Examples

Two worked examples show the harness composing only the layers a use case
needs — they are deliberately opposite shapes:

- **[`examples/codebase-qa/`](examples/codebase-qa/)** — a local (Ollama) agent
  with filesystem/ripgrep MCP tools, a Jira plugin, a secret-redaction hook,
  RAG over the repo docs, and long-term memory.
- **[`examples/research-digest/`](examples/research-digest/)** — a **Claude
  Sonnet** agent that digests documents. No MCP servers, no RAG, no memory —
  just a model, one skill, and a plugin whose `summarize` tool is *itself* a
  Sonnet call. Runs with only an `ANTHROPIC_API_KEY`.

### Codebase-QA in detail

[`examples/codebase-qa/`](examples/codebase-qa/) assembles everything into one
agent: a **code-search skill** ([`skills/code-search/`](skills/code-search/))
backed by filesystem + ripgrep MCP tools, a **Jira plugin**
([`plugins/jira/`](plugins/jira/)) registering an issue-lookup tool, a
**`pre_tool` hook that redacts secrets** from tool arguments, **RAG** over the
repo's docs, and **long-term memory** holding the user's coding conventions.

The end-to-end tests drive one question through the entire stack — skill
selection, RAG citation, secret redaction (verified by a fixture MCP server
echoing back the scrubbed argument), Jira lookup, and a real file read through
the path jail — plus a convention stated in one session recalled into a fresh
session from durable storage. All backends are scripted, so the suite runs
offline.

---

## Development

```bash
pytest                     # 313 tests
lint-imports               # enforce the layer contract
ruff check src tests       # lint
python -m agentloop.config.schema   # regenerate the manifest JSON Schema
```

The project is built in incremental, independently-testable phases; each is a
single commit and every phase leaves the system running. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §14 for the phase map and the
acceptance test that proves each one.

### Known extension points (not yet built)

- Additional model adapters (OpenAI, Gemini, xAI) — the Protocol and the
  OpenAI-dialect base class are proven by the two existing adapters.
- Streaming under recording/replay (the loop currently uses `complete()`).
- Background/scheduled consolidation (today it is awaited at turn end for
  determinism).
- Inter-agent delegation (`agents[]` selection is in place as the foundation).

---

## License

Not yet specified.
