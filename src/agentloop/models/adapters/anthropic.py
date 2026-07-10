"""Anthropic adapter — the second adapter, chosen because its content-block
format diverges most from the OpenAI dialect and stresses the abstraction.

Raw httpx against POST /v1/messages (no vendor SDK: the framework owns its
retry/fallback policy, and the SDK's built-in retries would double up with
FallbackChain's). Format notes vs the internal representation:

- system messages become the top-level `system` param;
- assistant text + tool calls become `text` / `tool_use` content blocks;
- tool-result messages become `tool_result` blocks in a *user* message, and
  consecutive results are merged into ONE user message (parallel tool
  results split across messages degrade the model's parallel calling);
- streaming is SSE: message_start → content_block_start/delta/stop →
  message_delta → message_stop; text arrives as text_delta, tool arguments
  accumulate from input_json_delta fragments.
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

PROVIDER = "anthropic"
DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


def tool_schema(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


def _tool_result_block(message: Message) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": message.text,
    }
    if message.is_error:
        block["is_error"] = True
    return block


def _assistant_blocks(message: Message) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if message.text:
        blocks.append({"type": "text", "text": message.text})
    for call in message.tool_calls:
        blocks.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        )
    return blocks


def build_messages_payload(
    model: str,
    request: CompletionRequest,
    params: dict[str, Any],
    *,
    stream: bool = False,
) -> dict[str, Any]:
    merged_params = {**params, **request.params}
    params_max_tokens = merged_params.pop("max_tokens", None)  # always pop: it must
    max_tokens = request.max_tokens or params_max_tokens or DEFAULT_MAX_TOKENS  # not resurface via **merged_params

    system_texts: list[str] = []
    wire_messages: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "system":
            system_texts.append(message.text)
        elif message.role == "tool":
            block = _tool_result_block(message)
            last = wire_messages[-1] if wire_messages else None
            # merge consecutive tool results into one user message
            if last and last["role"] == "user" and isinstance(last["content"], list):
                last["content"].append(block)
            else:
                wire_messages.append({"role": "user", "content": [block]})
        elif message.role == "assistant":
            wire_messages.append(
                {"role": "assistant", "content": _assistant_blocks(message)}
            )
        else:
            wire_messages.append({"role": "user", "content": message.text})

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": wire_messages,
        **merged_params,
    }
    if system_texts:
        payload["system"] = "\n\n".join(system_texts)
    if request.tools:
        payload["tools"] = tool_schema(request.tools)
    if stream:
        payload["stream"] = True
    return payload


def _parse_usage(raw: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=raw.get("input_tokens", 0),
        output_tokens=raw.get("output_tokens", 0),
        cache_read_tokens=raw.get("cache_read_input_tokens") or 0,
    )


def parse_messages_response(
    data: dict[str, Any], model: str, pricing: PricingTable
) -> CompletionResult:
    text_parts: list[TextPart] = []
    tool_calls: list[ToolCall] = []
    for block in data.get("content", []):
        if block["type"] == "text":
            text_parts.append(TextPart(text=block["text"]))
        elif block["type"] == "tool_use":
            tool_calls.append(
                ToolCall(id=block["id"], name=block["name"], arguments=block["input"])
            )
        # thinking / other block types carry no internal representation yet
    usage = _parse_usage(data.get("usage", {}))
    return CompletionResult(
        message=Message(
            role="assistant", content=tuple(text_parts), tool_calls=tuple(tool_calls)
        ),
        stop_reason=_STOP_REASONS.get(data.get("stop_reason") or "", "other"),
        usage=usage,
        cost=pricing.cost_for(PROVIDER, model, usage),
        provider=PROVIDER,
        model=model,
    )


class AnthropicProvider:
    provider = PROVIDER

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        params: dict[str, Any] | None = None,
        pricing: PricingTable | None = None,
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._params = params or {}
        self._pricing = pricing or PricingTable()
        self._timeout = timeout
        self._transport = transport

    @property
    def model(self) -> str:
        return self._model

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            transport=self._transport,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
        )

    def tool_call_schema(self, tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        return tool_schema(tools)

    def count_tokens(self, messages: Sequence[Message]) -> int:
        # The count_tokens endpoint exists but a network call per budget
        # check is not worth it; Usage carries the real numbers.
        return estimate_tokens(messages)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        payload = build_messages_payload(self._model, request, self._params)
        try:
            async with self._client() as client:
                response = await client.post("/v1/messages", json=payload)
        except httpx.HTTPError as exc:
            raise map_transport(PROVIDER, exc) from exc
        if response.status_code != 200:
            raise map_status(PROVIDER, response)
        return parse_messages_response(response.json(), self._model, self._pricing)

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        payload = build_messages_payload(self._model, request, self._params, stream=True)
        acc = _StreamAccumulator()
        try:
            async with self._client() as client, client.stream(
                "POST", "/v1/messages", json=payload
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise map_status(PROVIDER, response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[len("data:") :].strip())
                    delta = acc.feed(event)
                    if delta:
                        yield TextDelta(text=delta)
        except httpx.HTTPError as exc:
            raise map_transport(PROVIDER, exc) from exc
        yield StreamCompleted(
            result=parse_messages_response(acc.response(), self._model, self._pricing)
        )


class _StreamAccumulator:
    """Folds SSE events back into a /v1/messages-shaped response dict, so
    streaming and non-streaming share one parse path."""

    def __init__(self) -> None:
        self._blocks: dict[int, dict[str, Any]] = {}
        self._usage: dict[str, Any] = {}
        self._stop_reason: str | None = None

    def feed(self, event: dict[str, Any]) -> str | None:
        """Consume one SSE event; return text to surface as a delta, if any."""
        kind = event.get("type")
        if kind == "message_start":
            self._usage.update(event["message"].get("usage", {}))
        elif kind == "content_block_start":
            block = dict(event["content_block"])
            if block["type"] == "tool_use":
                block.setdefault("input", {})
                block["_partial_json"] = ""
            self._blocks[event["index"]] = block
        elif kind == "content_block_delta":
            delta = event["delta"]
            block = self._blocks[event["index"]]
            if delta["type"] == "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
                return delta["text"]
            if delta["type"] == "input_json_delta":
                block["_partial_json"] += delta["partial_json"]
        elif kind == "content_block_stop":
            block = self._blocks[event["index"]]
            partial = block.pop("_partial_json", None)
            if partial:
                block["input"] = json.loads(partial)
        elif kind == "message_delta":
            self._stop_reason = event["delta"].get("stop_reason") or self._stop_reason
            self._usage.update(event.get("usage", {}))
        return None

    def response(self) -> dict[str, Any]:
        content = [self._blocks[i] for i in sorted(self._blocks)]
        for block in content:
            block.pop("_partial_json", None)
        return {
            "content": content,
            "stop_reason": self._stop_reason,
            "usage": self._usage,
        }
