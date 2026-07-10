"""Pydantic models for agent.manifest.yaml (§5 of docs/ARCHITECTURE.md).

The JSON Schema shipped in schema.json is generated from `Manifest`;
these models are the single source of truth. All models are frozen and
reject unknown keys, so a typo in a manifest fails loudly instead of
being silently ignored.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MANIFEST_SCHEMA_VERSION = "1.0"

HookEvent = Literal[
    "pre_model",
    "post_model",
    "pre_tool",
    "post_tool",
    "pre_retrieval",
    "post_retrieval",
    "on_error",
    "on_turn_end",
]

# "pkg.mod.attr" or "pkg.mod:attr"
_DOTTED_PATH = r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*(:[A-Za-z_][A-Za-z0-9_]*)?$"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _duplicates(names: list[str]) -> list[str]:
    return sorted({n for n in names if names.count(n) > 1})


class ModelRef(_Base):
    """A provider/model pair; accepts the string shorthand "provider/name"."""

    provider: str = Field(min_length=1)
    name: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _coerce_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            provider, sep, name = value.partition("/")
            if not sep or not provider or not name:
                raise ValueError(f"model reference must be 'provider/name', got {value!r}")
            return {"provider": provider, "name": name}
        return value


class PricingEntry(_Base):
    """USD per million tokens; overrides the shipped pricing table (A9)."""

    input_per_mtok: Decimal = Field(ge=0)
    output_per_mtok: Decimal = Field(ge=0)
    cache_read_per_mtok: Decimal = Field(default=Decimal("0"), ge=0)


class ModelConfig(_Base):
    provider: str = Field(min_length=1, description="Primary provider, e.g. 'ollama'.")
    name: str = Field(min_length=1, description="Model name as the provider knows it.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Provider-passthrough sampling params."
    )
    fallback: tuple[ModelRef, ...] = Field(
        default=(), description="Ordered fallback chain, tried after the primary is exhausted."
    )
    pricing: dict[str, PricingEntry] = Field(
        default_factory=dict,
        description="Per-'provider/model' price overrides for cost accounting.",
    )

    @field_validator("pricing")
    @classmethod
    def _pricing_keys(cls, v: dict[str, PricingEntry]) -> dict[str, PricingEntry]:
        for key in v:
            if "/" not in key:
                raise ValueError(f"pricing key must be 'provider/model', got {key!r}")
        return v


class MCPServerConfig(_Base):
    name: str = Field(min_length=1)
    transport: Literal["stdio", "http"]
    command: str | None = Field(default=None, description="stdio only: executable to spawn.")
    args: tuple[str, ...] = Field(default=(), description="stdio only: argv for the command.")
    url: str | None = Field(default=None, description="http only: server base URL.")
    env: dict[str, str] = Field(
        default_factory=dict,
        description="stdio only: the ONLY environment variables passed to the server "
        "(scrubbed-env policy, A7).",
    )

    @model_validator(mode="after")
    def _check_transport_fields(self) -> MCPServerConfig:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("stdio transport requires 'command'")
            if self.url is not None:
                raise ValueError("stdio transport does not take 'url'")
        else:
            if not self.url:
                raise ValueError("http transport requires 'url'")
            if self.command is not None or self.args:
                raise ValueError("http transport does not take 'command'/'args'")
        return self


class ToolsConfig(_Base):
    mcp_servers: tuple[MCPServerConfig, ...] = ()
    allowlist: tuple[str, ...] = Field(
        default=(),
        description="Globs of 'server.tool' permitted to execute. "
        "Empty means deny all (fail closed).",
    )

    @model_validator(mode="after")
    def _unique_server_names(self) -> ToolsConfig:
        if dupes := _duplicates([s.name for s in self.mcp_servers]):
            raise ValueError(f"duplicate MCP server names: {dupes}")
        return self


class AgentConfig(_Base):
    """A named agent configuration; one is active per run (A4)."""

    name: str = Field(min_length=1)
    persona: str = Field(default="", description="Appended to the system prompt after intent.")
    model: ModelConfig | None = Field(
        default=None, description="Override of the top-level model; None inherits."
    )
    tools: tuple[str, ...] | None = Field(
        default=None,
        description="Subset of the allowlist this agent may use; None inherits all.",
    )
    skills: tuple[str, ...] | None = Field(
        default=None, description="Subset of top-level skills; None inherits all."
    )


class PluginConfig(_Base):
    name: str = Field(min_length=1)
    source: str = Field(
        min_length=1, description="Local path or installed distribution name."
    )
    version: str | None = Field(
        default=None, description="PEP 440 version constraint, e.g. '>=0.2,<0.3'."
    )
    config: dict[str, Any] = Field(default_factory=dict)


class HookEntry(_Base):
    handler: str = Field(
        pattern=_DOTTED_PATH, description="Import path: 'pkg.mod:attr' or 'pkg.mod.attr'."
    )
    priority: int = Field(default=100, description="Ascending execution order; ties by "
                          "registration order (manifest before plugins).")
    config: dict[str, Any] = Field(default_factory=dict)


class ChunkConfig(_Base):
    size: int = Field(default=512, gt=0, description="Chunk size in tokens.")
    overlap: int = Field(default=64, ge=0)

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> ChunkConfig:
        if self.overlap >= self.size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class EmbeddingConfig(_Base):
    provider: str = Field(default="ollama", min_length=1)
    model: str = Field(default="nomic-embed-text", min_length=1)


class HybridConfig(_Base):
    vector_k: int = Field(default=20, gt=0, description="Candidates from sqlite-vec.")
    fts_k: int = Field(default=20, gt=0, description="Candidates from FTS5 BM25.")
    rrf_k: int = Field(default=60, gt=0, description="RRF rank constant.")


class RagConfig(_Base):
    sources: tuple[str, ...] = Field(default=(), description="Files/dirs to ingest.")
    chunk: ChunkConfig = ChunkConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    top_k: int = Field(default=8, gt=0, description="Chunks injected after fusion/rerank.")
    hybrid: HybridConfig = HybridConfig()
    reranker: str | None = Field(
        default=None, description="Reranker name; None = RRF only (A6)."
    )


class ShortTermMemoryConfig(_Base):
    max_tokens: int = Field(
        default=8_000, gt=0,
        description="Episodic log size that triggers overflow summarization.",
    )


class LongTermMemoryConfig(_Base):
    promotion_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    recurrence_min: int = Field(
        default=2, ge=1, description="Sessions/mentions required for non-explicit promotion."
    )
    dedup_cosine: float = Field(default=0.92, ge=0.0, le=1.0)
    decay_lambda: float = Field(default=0.01, ge=0.0, description="Per-day decay rate.")
    score_floor: float = Field(
        default=0.15, ge=0.0, description="Below this, facts are archived (never deleted)."
    )


class MemoryConfig(_Base):
    enabled: bool = True
    short_term: ShortTermMemoryConfig = ShortTermMemoryConfig()
    long_term: LongTermMemoryConfig = LongTermMemoryConfig()


class LimitsConfig(_Base):
    max_steps: int = Field(default=16, gt=0, description="PLAN entries per turn.")
    max_tokens: int = Field(default=200_000, gt=0, description="Cumulative, per run.")
    max_wall_clock_s: float = Field(default=300.0, gt=0)
    max_cost_usd: Decimal = Field(default=Decimal("1.00"), gt=0)
    tool_timeout_s: float = Field(default=30.0, gt=0, description="Per tool call.")
    reflection_retries: int = Field(default=1, ge=0, description="Max REFLECT→PLAN bounces.")


class Manifest(_Base):
    """Root of all configuration: one agent.manifest.yaml per application."""

    version: str = Field(
        pattern=r"^1(\.\d+)?$",
        description="Manifest schema version (quote it: '1.0', not 1.0).",
    )
    intent: str = Field(
        description="Natural-language purpose; injected verbatim as the first "
        "system-prompt block."
    )
    model: ModelConfig
    agents: tuple[AgentConfig, ...] = ()
    tools: ToolsConfig = ToolsConfig()
    skills: tuple[str, ...] = Field(default=(), description="Skill directory paths or "
                                    "plugin-provided skill names.")
    plugins: tuple[PluginConfig, ...] = ()
    hooks: dict[HookEvent, tuple[HookEntry, ...]] = Field(default_factory=dict)
    rag: RagConfig = RagConfig()
    memory: MemoryConfig = MemoryConfig()
    limits: LimitsConfig = LimitsConfig()

    @field_validator("intent")
    @classmethod
    def _intent_non_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("intent must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _unique_names(self) -> Manifest:
        if dupes := _duplicates([a.name for a in self.agents]):
            raise ValueError(f"duplicate agent names: {dupes}")
        if dupes := _duplicates([p.name for p in self.plugins]):
            raise ValueError(f"duplicate plugin names: {dupes}")
        return self
