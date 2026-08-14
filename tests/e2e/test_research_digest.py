"""The research-digest example, end to end and offline.

Proves the harness drives a use case with no MCP servers, no RAG, and no
memory — just a Claude Sonnet model, the `digest` skill, and the
`summarizer` plugin whose tool is itself a Sonnet call. Both the main loop
model and the summarizer's model are injected (scripted), so the assembly is
exercised for real without any network.
"""

from __future__ import annotations

from pathlib import Path

from agentloop.api import Agent
from agentloop.config.resolve import load_config
from agentloop.models import registry as model_registry
from agentloop.types import ToolCall

from .conftest import RecordingEmitter, ScriptedProvider, completion

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "examples" / "research-digest" / "agent.manifest.yaml"

THREAD = "We agreed to ship the billing dashboard in Q3. Priya leads frontend."


async def build(main_provider, summarizer_provider, monkeypatch, emitter):
    # the plugin builds its Sonnet provider via registry.build_provider — swap
    # it for a scripted one so no Anthropic key or network is needed
    monkeypatch.setattr(
        model_registry, "build_provider", lambda *a, **k: summarizer_provider
    )
    agent = Agent(
        load_config(MANIFEST, env={}),
        agent_name="analyst",
        provider=main_provider,   # injected → the manifest's Sonnet model is bypassed
        emitter=emitter,
    )
    return agent


class TestDigest:
    async def test_skill_drives_a_sonnet_backed_summarize_operation(
        self, monkeypatch
    ):
        emitter = RecordingEmitter()
        # the summarizer plugin's model (stands in for the real Claude Sonnet)
        summarizer = ScriptedProvider(
            [completion("- Ship billing dashboard in Q3\n- Priya leads frontend")]
        )
        # the main loop model: call the summarize tool, then present the digest
        main = ScriptedProvider(
            [
                completion(
                    tool_calls=(
                        ToolCall(
                            id="c1", name="summarizer__summarize",
                            arguments={"text": THREAD, "style": "bullet points"},
                        ),
                    )
                ),
                completion(
                    "TL;DR: The team will ship the billing dashboard in Q3.\n\n"
                    "Key points:\n- Ship billing dashboard in Q3\n"
                    "- Priya leads frontend"
                ),
            ]
        )
        agent = await build(main, summarizer, monkeypatch, emitter)
        async with agent:
            turn = await agent.run(f"Summarize this thread: {THREAD}")

        assert turn.status == "completed"
        assert "TL;DR" in turn.text

        # 1. the digest SKILL was selected on the "summarize" trigger and its
        #    body entered the system prompt
        system = main.requests[0].messages[0].text
        assert "Active skill: digest" in system
        assert any(
            k == "skill.selected" and p["name"] == "digest"
            for k, p in emitter.events
        )

        # 2. the operation was genuinely a model call: the summarizer provider
        #    received the thread text and a summarizer system prompt
        assert len(summarizer.requests) == 1
        summ_request = summarizer.requests[0]
        assert any("summarizer" in m.text.lower() or "summariz" in m.text.lower()
                   for m in summ_request.messages if m.role == "system")
        assert THREAD in summ_request.messages[-1].text

        # 3. the summary produced by that Sonnet call flowed back to the main
        #    model as the tool result
        tool_msg = [m for m in main.requests[1].messages if m.role == "tool"][0]
        assert "Priya leads frontend" in tool_msg.text

    async def test_minimal_assembly_needs_no_mcp_rag_or_memory(self, monkeypatch):
        summarizer = ScriptedProvider([completion("a summary")])
        main = ScriptedProvider([completion("done")])
        agent = await build(main, summarizer, monkeypatch, RecordingEmitter())
        async with agent:
            assert agent.components is not None
            # the only executable tools are the plugin's summarize + use_skill
            names = {s.name for s in agent.components.gateway.specs()}
            assert names == {"summarizer__summarize", "use_skill"}
            # no RAG, no memory were built for this use case
            assert agent.components.retriever is None
            assert agent.components.memory is None
            assert agent.components.plugin_report.loaded == ["summarizer"]
