from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from agentloop.api import Agent
from agentloop.config.resolve import load_config
from agentloop.models.protocol import (
    CompletionRequest,
    CompletionResult,
)
from agentloop.types import Cost, Message, TextPart, ToolCall, Usage

from ..rag.conftest import VocabEmbedder

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "codebase-qa"
FIXTURE_SERVER = Path(__file__).parent / "fixture_code_server.py"

# the example's redaction hook must be importable by dotted path
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


def completion(
    text: str = "",
    tool_calls: tuple[ToolCall, ...] = (),
) -> CompletionResult:
    return CompletionResult(
        message=Message(
            role="assistant",
            content=(TextPart(text=text),) if text else (),
            tool_calls=tool_calls,
        ),
        stop_reason="tool_calls" if tool_calls else "stop",
        usage=Usage(input_tokens=20, output_tokens=8),
        cost=Cost(usd=Decimal("0.001")),
        provider="scripted",
        model="scripted",
    )


class ScriptedProvider:
    provider = "scripted"
    model = "scripted"

    def __init__(self, script: Sequence[CompletionResult]):
        self.script = list(script)
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        return self.script.pop(0)

    async def stream(self, request):  # pragma: no cover
        raise NotImplementedError
        yield

    def tool_call_schema(self, tools):
        return []

    def count_tokens(self, messages):
        return 0


class CannedMemoryProvider:
    """Consolidation/summarization backend. Content-driven so it is
    order-independent: emits a durable fact whenever the transcript states a
    YAML convention, otherwise extracts nothing."""

    provider = "memory"
    model = "memory"

    def __init__(self):
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.calls += 1
        transcript = " ".join(m.text for m in request.messages)
        if "YAML" in transcript or "yaml" in transcript:
            payload = (
                '[{"text": "User keeps config files in YAML format", '
                '"type": "convention", "confidence": 0.95, "explicit": true, '
                '"quote": "always keep configuration in YAML"}]'
            )
        else:
            payload = "[]"
        return completion(payload)

    async def stream(self, request):  # pragma: no cover
        raise NotImplementedError
        yield

    def tool_call_schema(self, tools):
        return []

    def count_tokens(self, messages):
        return 0


class RecordingEmitter:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind, payload):
        self.events.append((kind, dict(payload)))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.events]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A tmp repo with a docs corpus and a real source file to read."""
    repo = tmp_path / "repo"
    (repo / "src" / "app" / "config").mkdir(parents=True)
    (repo / "src" / "app" / "config" / "loader.py").write_text(
        "def load_config(path):\n"
        "    # defaults < manifest < env < CLI\n"
        "    return resolve(path)\n",
        encoding="utf-8",
    )
    docs = repo / "docs"
    docs.mkdir()
    shutil.copytree(EXAMPLE_DIR / "docs", docs, dirs_exist_ok=True)
    return repo


def write_manifest(tmp_path: Path, workspace: Path) -> Path:
    manifest = {
        "version": "1.0",
        "intent": "Answer questions about this codebase, citing files.",
        "model": {"provider": "ollama", "name": "unused-injected"},
        "agents": [{"name": "qa", "persona": "You are a precise assistant."}],
        "tools": {
            "mcp_servers": [
                {
                    "name": "code",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(FIXTURE_SERVER)],
                }
            ],
            "allowlist": ["code.search", "code.read_file", "jira.issue_lookup"],
            "sandbox": {"roots": [str(workspace)], "max_result_chars": 100000},
        },
        "skills": [str(REPO_ROOT / "skills" / "code-search")],
        "plugins": [
            {
                "name": "jira",
                "source": str(REPO_ROOT / "plugins" / "jira"),
                "version": ">=0.1,<0.2",
                "config": {"base_url": "https://example.atlassian.net"},
            }
        ],
        "hooks": {
            "pre_tool": [
                {
                    "handler": "codebaseqa_hooks:redact_secrets",
                    "priority": 10,
                    "config": {"patterns": ["sk-[A-Za-z0-9_-]{6,}"]},
                }
            ]
        },
        "rag": {
            "sources": [str(workspace / "docs")],
            "embedding": {"provider": "ollama", "model": "fake"},
            "top_k": 3,
            "chunk": {"size": 256, "overlap": 32},
        },
        "memory": {"enabled": True},
        "limits": {"max_steps": 8, "tool_timeout_s": 10},
    }
    path = tmp_path / "agent.manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


async def build_agent(
    manifest_path: Path,
    script: Sequence[CompletionResult],
    *,
    storage_dir: Path,
    embedder: VocabEmbedder,
    memory_provider: CannedMemoryProvider,
    emitter: RecordingEmitter,
    session_id: str | None = None,
) -> tuple[Agent, ScriptedProvider]:
    provider = ScriptedProvider(script)
    agent = Agent(
        load_config(manifest_path, env={}),
        agent_name="qa",
        provider=provider,
        embedder=embedder,
        memory_provider=memory_provider,
        emitter=emitter,
        storage_dir=storage_dir,
        session_id=session_id,
    )
    return agent, provider
