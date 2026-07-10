"""Ollama adapter: golden request format, response normalization, streaming."""

from __future__ import annotations

import json

import httpx
import pytest

from agentloop.models.adapters.ollama import (
    OllamaProvider,
    build_chat_payload,
    parse_chat_response,
)
from agentloop.models.pricing import PricingTable
from agentloop.models.protocol import CompletionRequest, StreamCompleted, TextDelta
from agentloop.types import Message, ProviderError, TransientProviderError

from .conftest import load_golden

PRICING = PricingTable()


class TestRequestFormat:
    def test_golden_request(self, request_fx):
        payload = build_chat_payload("qwen2.5:14b", request_fx, {"temperature": 0.2})
        assert payload == load_golden("ollama_request.json")

    def test_max_tokens_maps_to_num_predict(self):
        request = CompletionRequest(messages=(Message.user("hi"),), max_tokens=64)
        payload = build_chat_payload("m", request, {})
        assert payload["options"] == {"num_predict": 64}

    def test_request_params_override_adapter_params(self):
        request = CompletionRequest(
            messages=(Message.user("hi"),), params={"temperature": 0.9}
        )
        payload = build_chat_payload("m", request, {"temperature": 0.2, "top_p": 0.5})
        assert payload["options"] == {"temperature": 0.9, "top_p": 0.5}


class TestResponseParsing:
    def test_tool_calls_get_synthesized_ids(self):
        data = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "grep", "arguments": {"pattern": "x"}}},
                    {"function": {"name": "ls", "arguments": {"path": "."}}},
                ],
            },
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 20,
        }
        result = parse_chat_response(data, "m", PRICING)
        assert [c.id for c in result.message.tool_calls] == ["call_0", "call_1"]
        assert result.stop_reason == "tool_calls"  # tool calls win over done_reason
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 20

    def test_string_arguments_are_parsed(self):
        data = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "grep", "arguments": '{"pattern": "x"}'}}
                ],
            }
        }
        result = parse_chat_response(data, "m", PRICING)
        assert result.message.tool_calls[0].arguments == {"pattern": "x"}

    def test_plain_text_response(self):
        data = {"message": {"content": "The loader is in config/."}, "done_reason": "stop"}
        result = parse_chat_response(data, "m", PRICING)
        assert result.message.text == "The loader is in config/."
        assert result.stop_reason == "stop"

    def test_length_maps_to_max_tokens(self):
        data = {"message": {"content": "trunc"}, "done_reason": "length"}
        assert parse_chat_response(data, "m", PRICING).stop_reason == "max_tokens"

    def test_local_models_cost_zero_but_known(self):
        data = {"message": {"content": "hi"}, "prompt_eval_count": 5, "eval_count": 5}
        cost = parse_chat_response(data, "m", PRICING).cost
        assert cost.usd == 0 and cost.known is True


def _provider(handler) -> OllamaProvider:
    return OllamaProvider("m", transport=httpx.MockTransport(handler))


class TestHTTP:
    async def test_complete_end_to_end(self):
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/api/chat"
            body = json.loads(req.content)
            assert body["model"] == "m" and body["stream"] is False
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "hello"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 2,
                },
            )

        result = await _provider(handler).complete(
            CompletionRequest(messages=(Message.user("hi"),))
        )
        assert result.message.text == "hello"
        assert result.provider == "ollama"

    async def test_5xx_is_transient(self):
        def handler(req):
            return httpx.Response(500, text="boom")

        with pytest.raises(TransientProviderError):
            await _provider(handler).complete(
                CompletionRequest(messages=(Message.user("hi"),))
            )

    async def test_4xx_is_not_retryable(self):
        def handler(req):
            return httpx.Response(400, text="bad request")

        with pytest.raises(ProviderError) as exc_info:
            await _provider(handler).complete(
                CompletionRequest(messages=(Message.user("hi"),))
            )
        assert not isinstance(exc_info.value, TransientProviderError)

    async def test_connect_error_is_transient(self):
        def handler(req):
            raise httpx.ConnectError("refused")

        with pytest.raises(TransientProviderError):
            await _provider(handler).complete(
                CompletionRequest(messages=(Message.user("hi"),))
            )

    async def test_streaming(self):
        lines = [
            {"message": {"content": "The "}, "done": False},
            {"message": {"content": "loader"}, "done": False},
            {
                "message": {"content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 3,
            },
        ]
        ndjson = "\n".join(json.dumps(line) for line in lines)

        def handler(req):
            return httpx.Response(200, content=ndjson.encode())

        events = [
            e
            async for e in _provider(handler).stream(
                CompletionRequest(messages=(Message.user("hi"),))
            )
        ]
        deltas = [e for e in events if isinstance(e, TextDelta)]
        assert [d.text for d in deltas] == ["The ", "loader"]
        final = events[-1]
        assert isinstance(final, StreamCompleted)
        assert final.result.message.text == "The loader"
        assert final.result.usage.input_tokens == 12
