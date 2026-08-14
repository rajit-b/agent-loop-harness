"""Summarizer plugin: registers a `summarize` tool whose operation is itself
a Claude Sonnet call.

This is the framework-native way to "use a model inside a skill": a skill
cannot contain tool implementations, so the model-backed operation lives in a
tool, and the skill invokes it. The tool builds a Claude Sonnet provider
through the framework's own model abstraction (agentloop.models) and calls
`complete()` — the same adapter, pricing, and error taxonomy the main loop
uses. `ANTHROPIC_API_KEY` is read from the environment, never the manifest.

Config (from the manifest's plugin entry):
  model:      Claude model id for the summarizer (default claude-sonnet-5)
  max_tokens: output cap for the summarizer call (default 1024)
"""

from __future__ import annotations

import os
from typing import Any

from agentloop.models import registry  # module import → build_provider is a
from agentloop.models.protocol import CompletionRequest  # test-patchable seam
from agentloop.types import Message, ToolSpec

PLUGIN_API_VERSION = 1

SYSTEM_PROMPT = (
    "You are a precise summarizer. Preserve concrete facts, names, numbers, "
    "and decisions. Never invent information that is not in the source text."
)


class SummarizerPlugin:
    name = "summarizer"
    version = "0.1.0"
    api_version = PLUGIN_API_VERSION

    def __init__(self, config: dict[str, Any]):
        self._model = config.get("model", "claude-sonnet-5")
        self._max_tokens = int(config.get("max_tokens", 1024))
        self._provider = None  # built lazily on first use

    def register(self, registrar: Any) -> None:
        async def summarize(arguments: dict[str, Any]) -> str:
            text = str(arguments.get("text", "")).strip()
            if not text:
                raise ValueError("summarize requires non-empty 'text'")
            style = str(arguments.get("style", "bullet points"))
            focus = str(arguments.get("focus", "")).strip()

            instruction = f"Summarize the text below as {style}."
            if focus:
                instruction += f" Focus on: {focus}."
            prompt = f"{instruction}\n\n---\n{text}\n---"

            provider = self._ensure_provider()
            result = await provider.complete(
                CompletionRequest(
                    messages=(
                        Message.system(SYSTEM_PROMPT),
                        Message.user(prompt),
                    )
                )
            )
            return result.message.text

        registrar.add_tool(
            ToolSpec(
                name="summarize",
                description=(
                    "Summarize a block of text with an LLM. Returns the "
                    "summary. Use for condensing long documents or threads."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to summarize."},
                        "style": {
                            "type": "string",
                            "description": "e.g. 'bullet points', 'one paragraph', "
                            "'three sentences'.",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Optional aspect to emphasize.",
                        },
                    },
                    "required": ["text"],
                },
            ),
            summarize,
        )

    def _ensure_provider(self):
        if self._provider is None:
            self._provider = registry.build_provider(
                "anthropic",
                self._model,
                params={"max_tokens": self._max_tokens},
                secrets=os.environ,
            )
        return self._provider

    def dispose(self) -> None:
        return None


def agentloop_plugin(config: dict[str, Any]) -> SummarizerPlugin:
    return SummarizerPlugin(config)
