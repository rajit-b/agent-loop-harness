"""The Phase 11 gate: the codebase-QA worked example, end to end.

Real skill loading, real plugin loading (fail-closed), real MCP client
against a fixture stdio server, real gating + sandbox, real hook bus, real
RAG over a docs corpus, real three-tier memory — all assembled by the
programmatic API from a manifest. Only the model and embedding backends are
injected (scripted / deterministic) so the whole thing runs offline.
"""

from __future__ import annotations


from agentloop.types import ToolCall

from ..rag.conftest import VocabEmbedder
from .conftest import (
    CannedMemoryProvider,
    RecordingEmitter,
    build_agent,
    completion,
    write_manifest,
)


class TestWorkedExample:
    async def test_search_read_redact_cite_all_in_one_turn(
        self, tmp_path, workspace
    ):
        """question → code-search skill → code MCP tool → secret-redaction
        hook fires → Jira plugin lookup → cited answer."""
        manifest = write_manifest(tmp_path, workspace)
        loader_path = str(workspace / "src" / "app" / "config" / "loader.py")
        emitter = RecordingEmitter()
        script = [
            # PLAN 1: the model searches (with a secret in the query!) and
            # looks up the linked Jira issue, in parallel
            completion(
                tool_calls=(
                    ToolCall(
                        id="c1", name="code__search",
                        arguments={"query": "load_config token=sk-live-DEADBEEF12"},
                    ),
                    ToolCall(
                        id="c2", name="jira__issue_lookup",
                        arguments={"key": "PROJ-42"},
                    ),
                )
            ),
            # PLAN 2: the model reads the file the search pointed at
            completion(
                tool_calls=(
                    ToolCall(
                        id="c3", name="code__read_file",
                        arguments={"path": loader_path},
                    ),
                )
            ),
            # PLAN 3: final, cited answer
            completion(
                "The loader is defined at src/app/config/loader.py:12 and "
                "applies defaults < manifest < env < CLI precedence [1]. "
                "Tracked in PROJ-42."
            ),
        ]
        agent, provider = await build_agent(
            manifest, script, storage_dir=tmp_path / "db",
            embedder=VocabEmbedder(), memory_provider=CannedMemoryProvider(),
            emitter=emitter,
        )
        async with agent:
            turn = await agent.run("where is the config loader?")

        assert turn.status == "completed"
        assert "loader.py:12" in turn.text

        # 1. the code-search SKILL was selected (trigger "where is") and its
        #    body entered the system prompt
        system = provider.requests[0].messages[0].text
        assert "Active skill: code-search" in system
        assert any(
            k == "skill.selected" and p["name"] == "code-search"
            for k, p in emitter.events
        )

        # 2. RAG injected cited context into the user message
        user_message = provider.requests[0].messages[-1].text
        assert "Retrieved context" in user_message
        assert "loader" in user_message.lower()

        # 3. the secret-redaction HOOK scrubbed the search query before the
        #    MCP tool ran — the fixture echoes what it actually received
        tool_messages = [
            m for m in provider.requests[1].messages if m.role == "tool"
        ]
        search_result = next(m for m in tool_messages if "received query" in m.text)
        assert "[REDACTED]" in search_result.text
        assert "sk-live-DEADBEEF12" not in search_result.text
        assert any(
            k == "hook.executed" and p.get("decision") == "mutated"
            for k, p in emitter.events
        )

        # 4. the Jira PLUGIN tool executed and its result reached the model
        jira_result = next(m for m in tool_messages if m.tool_call_id == "c2")
        assert "PROJ-42" in jira_result.text
        assert "example.atlassian.net" in jira_result.text

        # 5. the real file was read through MCP + the path jail (inside root)
        read_back = [
            m for m in provider.requests[2].messages if m.role == "tool"
        ]
        assert any("def load_config" in m.text for m in read_back)

    async def test_convention_recalled_in_a_fresh_session(
        self, tmp_path, workspace
    ):
        """Long-term memory: a convention stated in session A is recalled
        into session B's system prompt, from a durable store."""
        manifest = write_manifest(tmp_path, workspace)
        storage = tmp_path / "db"
        embedder = VocabEmbedder()
        memory_provider = CannedMemoryProvider()

        # session A: the user states a convention as a directive
        agent_a, _ = await build_agent(
            manifest, [completion("Understood — I'll keep config in YAML.")],
            storage_dir=storage, embedder=embedder,
            memory_provider=memory_provider, emitter=RecordingEmitter(),
            session_id="session-A",
        )
        async with agent_a:
            await agent_a.run("From now on always keep configuration in YAML.")

        # the convention was promoted to an active long-term fact
        assert memory_provider.calls >= 1

        # session B: a brand-new agent + session over the SAME durable store
        emitter_b = RecordingEmitter()
        agent_b, provider_b = await build_agent(
            manifest, [completion("Add it to agent.manifest.yaml (YAML).")],
            storage_dir=storage, embedder=embedder,
            memory_provider=CannedMemoryProvider(), emitter=emitter_b,
            session_id="session-B",
        )
        async with agent_b:
            await agent_b.run("how should I add a new config option?")

        system = provider_b.requests[0].messages[0].text
        assert "Known facts" in system
        assert "YAML" in system
        assert "[fact:" in system
        assert any(k == "memory.recalled" for k, _ in emitter_b.events)

        # session A's raw chat log did NOT leak into session B
        assert all(
            "From now on always keep configuration in YAML." != m.text
            for m in provider_b.requests[0].messages
        )


class TestAssembly:
    async def test_plugin_and_skill_and_mcp_all_wired(self, tmp_path, workspace):
        """The gateway exposes exactly the allowlisted tools plus use_skill,
        drawn from all three sources (MCP, plugin, builtin)."""
        manifest = write_manifest(tmp_path, workspace)
        agent, _ = await build_agent(
            manifest, [completion("done")], storage_dir=tmp_path / "db",
            embedder=VocabEmbedder(), memory_provider=CannedMemoryProvider(),
            emitter=RecordingEmitter(),
        )
        async with agent:
            assert agent.components is not None
            names = {s.name for s in agent.components.gateway.specs()}
            assert names == {
                "code__search", "code__read_file",  # MCP
                "jira__issue_lookup",                # plugin
                "use_skill",                         # builtin meta-tool
            }
            assert agent.components.plugin_report.loaded == ["jira"]
            assert agent.components.skills is not None
            assert agent.components.retriever is not None
            assert agent.components.memory is not None

    async def test_denied_tool_never_reaches_the_model(self, tmp_path, workspace):
        """A tool the model hallucinates outside the allowlist fails closed."""
        manifest = write_manifest(tmp_path, workspace)
        emitter = RecordingEmitter()
        script = [
            completion(
                tool_calls=(
                    ToolCall(id="c1", name="code__delete_everything",
                             arguments={}),
                )
            ),
            completion("recovered"),
        ]
        agent, provider = await build_agent(
            manifest, script, storage_dir=tmp_path / "db",
            embedder=VocabEmbedder(), memory_provider=CannedMemoryProvider(),
            emitter=emitter,
        )
        async with agent:
            turn = await agent.run("delete everything")
        assert turn.status == "completed"
        tool_message = [
            m for m in provider.requests[1].messages if m.role == "tool"
        ][0]
        assert tool_message.is_error is True
        assert any(k == "tool.denied" for k, _ in emitter.events)
