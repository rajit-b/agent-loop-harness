"""Ollama adapter (default provider) — /api/chat, no SDK.

Format notes vs the internal representation:
- Ollama tool calls carry no ids; we synthesize "call_{i}" so the loop's
  ToolCall/ToolResult pairing works uniformly.
- `arguments` usually arrives as a dict; older builds send a JSON string —
  both are handled.
- Tool-result messages map to role "tool" with `tool_name` when known.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from agentloop.models.adapters._http import map_status, map_transport
from agentloop.models.pricing import PricingTable
from agentloop.models.protocol import (
    CompletionRequest,
    CompletionResult,
    StopReason,
    StreamCompleted,
    StreamEvent,
    TextDelta,
    estimate_tokens,
)
from agentloop.types import Message, TextPart, ToolCall, ToolSpec, Usage

PROVIDER = "ollama"
DEFAULT_BASE_URL = "http://localhost:11434"


def tool_schema(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _message_to_wire(message: Message) -> dict[str, Any]:
    if message.role == "assistant":
        wire: dict[str, Any] = {"role": "assistant", "content": message.text}
        if message.tool_calls:
            wire["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}}
                for c in message.tool_calls
            ]
        return wire
    if message.role == "tool":
        wire = {"role": "tool", "content": message.text}
        if message.name:
            wire["tool_name"] = message.name
        return wire
    return {"role": message.role, "content": message.text}


def build_chat_payload(
    model: str,
    request: CompletionRequest,
    params: dict[str, Any],
    *,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message_to_wire(m) for m in request.messages],
        "stream": stream,
    }
    if request.tools:
        payload["tools"] = tool_schema(request.tools)
    options = {**params, **request.params}
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if options:
        payload["options"] = options
    return payload


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> tuple[ToolCall, ...]:
    calls = []
    for i, raw in enumerate(raw_calls):
        fn = raw.get("function", {})
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments else {}
        calls.append(ToolCall(id=f"call_{i}", name=fn["name"], arguments=arguments))
    return tuple(calls)


def _stop_reason(done_reason: str | None, has_tool_calls: bool) -> StopReason:
    if has_tool_calls:
        return "tool_calls"
    if done_reason in (None, "stop"):
        return "stop"
    if done_reason == "length":
        return "max_tokens"
    return "other"


def parse_chat_response(
    data: dict[str, Any], model: str, pricing: PricingTable
) -> CompletionResult:
    raw_message = data.get("message", {})
    text = raw_message.get("content", "")
    tool_calls = _parse_tool_calls(raw_message.get("tool_calls", []))
    usage = Usage(
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
    )
    return CompletionResult(
        message=Message(
            role="assistant",
            content=(TextPart(text=text),) if text else (),
            tool_calls=tool_calls,
        ),
        stop_reason=_stop_reason(data.get("done_reason"), bool(tool_calls)),
        usage=usage,
        cost=pricing.cost_for(PROVIDER, model, usage),
        provider=PROVIDER,
        model=model,
    )


class OllamaProvider:
    provider = PROVIDER

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        params: dict[str, Any] | None = None,
        pricing: PricingTable | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._params = params or {}
        self._pricing = pricing or PricingTable()
        self._timeout = timeout
        self._transport = transport  # test seam (httpx.MockTransport)

    @property
    def model(self) -> str:
        return self._model

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout, transport=self._transport
        )

    def tool_call_schema(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        return tool_schema(tools)

    def count_tokens(self, messages: Sequence[Message]) -> int:
        return estimate_tokens(messages)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        payload = build_chat_payload(self._model, request, self._params)
        try:
            async with self._client() as client:
                response = await client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise map_transport(PROVIDER, exc) from exc
        if response.status_code != 200:
            raise map_status(PROVIDER, response)
        return parse_chat_response(response.json(), self._model, self._pricing)

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        payload = build_chat_payload(self._model, request, self._params, stream=True)
        text_parts: list[str] = []
        raw_tool_calls: list[dict[str, Any]] = []
        final: dict[str, Any] = {}
        try:
            async with self._client() as client, client.stream(
                "POST", "/api/chat", json=payload
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise map_status(PROVIDER, response)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    message = chunk.get("message", {})
                    delta = message.get("content", "")
                    if delta:
                        text_parts.append(delta)
                        yield TextDelta(text=delta)
                    raw_tool_calls.extend(message.get("tool_calls", []))
                    if chunk.get("done"):
                        final = chunk
        except httpx.HTTPError as exc:
            raise map_transport(PROVIDER, exc) from exc
        final.setdefault("message", {})
        final["message"]["content"] = "".join(text_parts)
        final["message"]["tool_calls"] = raw_tool_calls
        yield StreamCompleted(
            result=parse_chat_response(final, self._model, self._pricing)
        )
