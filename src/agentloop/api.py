"""Programmatic API: the composition root (§ "Ship a CLI and a programmatic
API").

Agent.from_manifest builds the whole stack from one agent.manifest.yaml and
the resolution chain, then __aenter__ performs the async wiring that needs a
live event loop (MCP connect, embedding-dimension probe, RAG indexing).
Every external backend is injectable so the same assembly runs hermetically
in tests and against real providers in production:

    async with await Agent.from_manifest("agent.manifest.yaml") as agent:
        turn = await agent.run("where is the config loader?")
        print(turn.text)

Paths in the manifest (skills, plugin sources, rag.sources, sandbox roots)
resolve relative to the manifest's own directory. Credentials come from the
environment, never the manifest (see models.registry).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentloop.config.manifest import AgentConfig, Manifest
from agentloop.config.resolve import ResolvedConfig, load_config
from agentloop.hooks.bus import HookBus
from agentloop.hooks.loader import install_manifest_hooks
from agentloop.loop.machine import AgentLoop, TurnResult
from agentloop.memory.manager import MemoryManager
from agentloop.memory.store import MemoryStore
from agentloop.models.protocol import ModelProvider
from agentloop.models.registry import build_chain
from agentloop.plugins.contract import CliCommand
from agentloop.plugins.loader import LoadReport, PluginManager
from agentloop.rag.embed import OllamaEmbedder
from agentloop.rag.ingest import Indexer
from agentloop.rag.retrieve import Retriever
from agentloop.rag.sources import FileSource
from agentloop.rag.store import RagStore
from agentloop.skills.loader import Skill, load_skill
from agentloop.skills.manager import SkillManager
from agentloop.skills.tooling import register_use_skill
from agentloop.storage.sqlite import connect
from agentloop.tools.executor import ToolGateway, ToolRegistry
from agentloop.tools.mcp import MCPClient
from agentloop.tools.sandbox import Sandbox
from agentloop.types import (
    ConfigError,
    EmbeddingProvider,
    NullEmitter,
    TraceEmitter,
)


@dataclass(slots=True)
class AgentComponents:
    """The assembled pieces — exposed for introspection and testing."""

    loop: AgentLoop
    gateway: ToolGateway
    hooks: HookBus
    skills: SkillManager | None
    retriever: Retriever | None
    memory: MemoryManager | None
    plugin_report: LoadReport
    cli_commands: dict[str, CliCommand]


class Agent:
    def __init__(
        self,
        config: ResolvedConfig,
        *,
        agent_name: str | None = None,
        provider: ModelProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        memory_provider: ModelProvider | None = None,
        emitter: TraceEmitter | None = None,
        session_id: str | None = None,
        secrets: Mapping[str, str] | None = None,
        storage_dir: Path | str | None = None,
    ):
        self.config = config
        self.manifest = config.manifest
        self.active = _select_agent(self.manifest, agent_name)
        self._base_dir = config.path.parent
        self._emitter = emitter or NullEmitter()
        self._secrets = secrets
        self._session_id = session_id or uuid.uuid4().hex[:12]
        # storage_dir set → durable DB files (survive across sessions);
        # None → in-memory, single-process
        self._storage_dir = Path(storage_dir) if storage_dir else None

        self._provider = provider or self._build_provider()
        self._memory_provider = memory_provider or self._provider
        self._embedder = embedder  # may stay None if unused
        self._mcp: MCPClient | None = None
        self._rag_conn = None
        self._mem_conn = None
        self.components: AgentComponents | None = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    async def from_manifest(
        cls,
        manifest_path: Path | str,
        *,
        env: Mapping[str, str] | None = None,
        cli_overrides: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Agent:
        config = load_config(manifest_path, env=env, cli_overrides=cli_overrides)
        return cls(config, **kwargs)

    def _build_provider(self) -> ModelProvider:
        model = self.active.model or self.manifest.model
        return build_chain(model, secrets=self._secrets, emitter=self._emitter)

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self._base_dir / candidate

    def _resolve_plugin_source(self, source: str) -> str:
        """Local plugin paths resolve relative to the manifest, like skills
        and rag.sources; an entry-point name (no such path) passes through."""
        if Path(source).is_absolute():
            return source
        candidate = self._base_dir / source
        return str(candidate) if candidate.exists() else source

    async def _ensure_embedder(self) -> EmbeddingProvider:
        if self._embedder is None:
            self._embedder = OllamaEmbedder(
                self.manifest.rag.embedding.model,
                base_url=(self._secrets or {}).get(
                    "OLLAMA_HOST", "http://localhost:11434"
                ),
            )
        return self._embedder

    async def __aenter__(self) -> Agent:
        registry = ToolRegistry()  # builtins + plugin-registered tools
        hooks = HookBus(emitter=self._emitter)
        skill_objects: list[Skill] = []
        cli_commands: dict[str, CliCommand] = {}

        # plugins first (they may register tools/hooks/skills/cli), fail closed
        plugins = PluginManager(
            tools=registry, hooks=hooks, skills=skill_objects,
            cli=cli_commands, emitter=self._emitter,
        )
        plugin_configs = tuple(
            pc.model_copy(update={"source": self._resolve_plugin_source(pc.source)})
            for pc in self.manifest.plugins
        )
        plugin_report = plugins.load_all(plugin_configs)
        self._plugins = plugins

        # manifest hooks install after plugins → same-priority tie goes to
        # the manifest (§9)
        install_manifest_hooks(hooks, self.manifest.hooks)

        # manifest-declared skills, filtered by the active agent's subset
        for skill_path in self.manifest.skills:
            skill_objects.append(load_skill(self._resolve(skill_path)))
        skills = self._build_skill_manager(skill_objects)

        # MCP servers: connect and discover
        self._mcp = MCPClient(self.manifest.tools.mcp_servers)
        await self._mcp.start()

        # the use_skill meta-tool needs the skill manager
        if skills is not None:
            register_use_skill(registry, skills)

        gateway = ToolGateway(
            [registry, self._mcp],
            allowlist=self._effective_allowlist(skills is not None),
            sandbox=self._build_sandbox(),
            tool_timeout_s=self.manifest.limits.tool_timeout_s,
            emitter=self._emitter,
        )

        retriever = await self._build_retriever()
        memory = await self._build_memory()

        loop = AgentLoop(
            self._provider,
            gateway,
            intent=self.manifest.intent,
            persona=self.active.persona,
            limits=self.manifest.limits,
            hooks=hooks,
            skills=skills,
            retriever=retriever,
            memory=memory,
            emitter=self._emitter,
        )
        self.components = AgentComponents(
            loop=loop, gateway=gateway, hooks=hooks, skills=skills,
            retriever=retriever, memory=memory, plugin_report=plugin_report,
            cli_commands=cli_commands,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if hasattr(self, "_plugins"):
            self._plugins.dispose_all()
        if self._mcp is not None:
            await self._mcp.aclose()
            self._mcp = None
        for conn in (self._rag_conn, self._mem_conn):
            if conn is not None:
                conn.close()
        self._rag_conn = self._mem_conn = None

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    async def run(self, user_input: str, *, turn_id: str | None = None) -> TurnResult:
        if self.components is None:
            raise ConfigError("agent is not started; use 'async with agent:'")
        return await self.components.loop.run_turn(user_input, turn_id=turn_id)

    # ------------------------------------------------------------------
    # wiring helpers
    # ------------------------------------------------------------------

    def _effective_allowlist(self, has_skills: bool) -> tuple[str, ...]:
        base = self.active.tools or self.manifest.tools.allowlist
        # use_skill is a framework builtin, not an MCP/plugin tool — the
        # composition root grants it rather than special-casing the gateway
        return (*base, "use_skill") if has_skills else base

    def _build_sandbox(self) -> Sandbox:
        sandbox = self.manifest.tools.sandbox
        return Sandbox(
            roots=[self._resolve(r) for r in sandbox.roots],
            max_result_chars=sandbox.max_result_chars,
        )

    def _build_skill_manager(self, skills: list[Skill]) -> SkillManager | None:
        if self.active.skills is not None:
            allowed = set(self.active.skills)
            skills = [s for s in skills if s.name in allowed]
        if not skills:
            return None
        return SkillManager(skills, embedder=self._embedder)

    async def _build_retriever(self) -> Retriever | None:
        rag = self.manifest.rag
        if not rag.sources:
            return None
        embedder = await self._ensure_embedder()
        dims = len((await embedder.embed(["dimension probe"]))[0])
        self._rag_conn = connect(self._db_path("rag"))
        store = RagStore(self._rag_conn)
        store.initialize(embedding_model=rag.embedding.model, dimensions=dims)
        indexer = Indexer(
            store, embedder,
            chunk_size=rag.chunk.size, chunk_overlap=rag.chunk.overlap,
        )
        await indexer.index(FileSource([self._resolve(s) for s in rag.sources]))
        return Retriever(
            store, embedder,
            top_k=rag.top_k, vector_k=rag.hybrid.vector_k,
            fts_k=rag.hybrid.fts_k, rrf_k=rag.hybrid.rrf_k,
        )

    async def _build_memory(self) -> MemoryManager | None:
        if not self.manifest.memory.enabled:
            return None
        embedder = await self._ensure_embedder()
        dims = len((await embedder.embed(["dimension probe"]))[0])
        self._mem_conn = connect(self._db_path("memory"))
        store = MemoryStore(self._mem_conn)
        store.initialize(
            embedding_model=self.manifest.rag.embedding.model, dimensions=dims
        )
        return MemoryManager(
            store, embedder, self._memory_provider,
            config=self.manifest.memory, session_id=self._session_id,
            emitter=self._emitter,
        )

    def _db_path(self, name: str) -> str:
        if self._storage_dir is None:
            return ":memory:"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        return str(self._storage_dir / f"{name}.db")


def _select_agent(manifest: Manifest, name: str | None) -> AgentConfig:
    if not manifest.agents:
        return AgentConfig(name="default")  # persona-less default
    if name is None:
        return manifest.agents[0]
    for agent in manifest.agents:
        if agent.name == name:
            return agent
    known = ", ".join(a.name for a in manifest.agents)
    raise ConfigError(f"no agent named {name!r}; known agents: {known}")
