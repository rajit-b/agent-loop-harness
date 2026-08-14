# Research-digest agent

A second worked example, built to show the harness drives **any** use case —
not just the codebase-QA agent it ships with. This one condenses long
documents and discussion threads into a structured digest (TL;DR → key points
→ action items).

It is deliberately the *opposite shape* from the codebase-QA example:

|                | codebase-QA          | research-digest        |
|----------------|----------------------|------------------------|
| Model          | Ollama (local)       | **Claude Sonnet**      |
| MCP tools      | filesystem + ripgrep | none                   |
| RAG            | over repo docs       | none                   |
| Memory         | long-term facts      | none                   |
| Needs to run   | Ollama + MCP servers | **only an API key**    |

The whole agent is a **model + one skill + one plugin**. That's the point:
you compose only the layers your use case needs.

## The "model inside a skill" pattern

Skills in this framework carry instructions, not tool implementations — a
skill *invokes* tools. So when a skill's core operation should be powered by a
model, that model call lives in a **tool**, and the skill calls it:

```
user question
   → digest skill selected (trigger: "summarize" / "digest" / "TL;DR")
   → skill body tells the agent to call summarizer.summarize on the text
   → summarize tool runs a Claude Sonnet call (via agentloop.models)
   → the summary comes back; the agent shapes it into TL;DR + key points
```

- **The skill** — [`skills/digest/`](../../skills/digest/) — describes the
  operation and how to structure the result.
- **The plugin** — [`plugins/summarizer/`](../../plugins/summarizer/) —
  registers the `summarize` tool, whose handler builds a Claude Sonnet
  provider through the framework's own model abstraction and calls
  `complete()`. Same adapter, pricing, and error handling the main loop uses.
- **The manifest** — [`agent.manifest.yaml`](agent.manifest.yaml) — wires them
  together and sets `model: claude-sonnet-5` for the main loop.

So Claude Sonnet appears twice: as the agent's reasoning model, and as the
engine inside the skill's summarize operation. Either could be a different
model — swap the manifest's `model` or the plugin's `config.model`.

## Run it

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# CLI
agentloop run -m examples/research-digest/agent.manifest.yaml \
  "Summarize this: <paste a long document or thread>"

# or the programmatic API, with a built-in sample document
python examples/research-digest/run.py
```

`run.py` also writes a JSONL trace to `examples/research-digest/runs/` — feed
it to `agentloop replay <file>` to see every loop step, model call, and tool
call.

## How it's tested offline

[`tests/e2e/test_research_digest.py`](../../tests/e2e/test_research_digest.py)
assembles this exact agent from the manifest but injects scripted providers
for both the main model and the summarizer's model, so the full wiring — skill
selection, the plugin tool, the model-backed operation — is verified without a
network or an API key.
