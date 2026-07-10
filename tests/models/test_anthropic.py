"""Anthropic adapter: golden request, block normalization, SSE streaming."""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from agentloop.models.adapters.anthropic import (
    AnthropicProvider,
    build_messages_payload,
    parse_messages_response,
)
from agentloop.models.pricing import PricingTable
from agentloop.models.protocol import CompletionRequest, StreamCompleted, TextDelta
from agentloop.types import Message, ToolCall

from .conftest import load_golden

PRICING = PricingTable()


class TestRequestFormat:
    def test_golden_request(self, request_fx):
        payload = build_messages_payload(
            "claude-opus-4-8", request_fx, {"temperature": 0.2}
        )
        assert payload == load_golden("anthropic_request.json")

    def test_consecutive_tool_results_merge_into_one_user_message(self, convo):
        payload = build_messages_payload(
            "m", CompletionRequest(messages=convo), {}
        )
        tool_result_messages = [
            m
            for m in payload["messages"]
            if isinstance(m["content"], list)
            and any(b["type"] == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_messages) == 1
        assert len(tool_result_messages[0]["content"]) == 2

    def test_is_error_flag_survives(self):
        messages = (
            Message.user("q"),
            Message.assistant(tool_calls=(ToolCall(id="c1", name="t"),)),
            Message.tool_result("c1", "denied", is_error=True),
        )
        payload = build_messages_payload("m", CompletionRequest(messages=messages), {})
        block = payload["messages"][-1]["content"][0]
        assert block["is_error"] is True

    def test_max_tokens_precedence(self):
        request = CompletionRequest(messages=(Message.user("hi"),), max_tokens=99)
        payload = build_messages_payload("m", request, {"max_tokens": 500})
        assert payload["max_tokens"] == 99  # request beats adapter params
        payload = build_messages_payload(
            "m", CompletionRequest(messages=(Message.user("hi"),)), {"max_tokens": 500}
        )
        assert payload["max_tokens"] == 500  # adapter params beat the default


class TestResponseParsing:
    def test_tool_use_blocks_become_tool_calls(self):
        data = {
            "content": [
                {"type": "text", "text": "Searching."},
                {"type": "tool_use", "id": "toolu_01", "name": "grep",
                 "input": {"pattern": "x"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1000, "output_tokens": 50,
                      "cache_read_input_tokens": 2000},
        }
        result = parse_messages_response(data, "claude-opus-4-8", PRICING)
        assert result.message.text == "Searching."
        call = result.message.tool_calls[0]
        assert (call.id, call.name, call.arguments) == ("toolu_01", "grep", {"pattern": "x"})
        assert result.stop_reason == "tool_calls"
        assert result.usage.cache_read_tokens == 2000

    def test_cost_uses_pricing_table(self):
        data = {
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
                      "cache_read_input_tokens": 1_000_000},
        }
        cost = parse_messages_response(data, "claude-opus-4-8", PRICING).cost
        assert cost.usd == Decimal("30.50")  # 5 + 25 + 0.50
        assert cost.known is True

    @pytest.mark.parametrize(
        ("wire", "internal"),
        [("end_turn", "stop"), ("tool_use", "tool_calls"),
         ("max_tokens", "max_tokens"), ("refusal", "refusal"),
         ("pause_turn", "other"), (None, "other")],
    )
    def test_stop_reason_mapping(self, wire, internal):
        data = {"content": [], "stop_reason": wire, "usage": {}}
        assert parse_messages_response(data, "m", PRICING).stop_reason == internal


def _sse(events: list[dict]) -> str:
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    )


STREAM_EVENTS = [
    {"type": "message_start",
     "message": {"usage": {"input_tokens": 40, "cache_read_input_tokens": 100}}},
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "Searching "}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "now."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1,
     "content_block": {"type": "tool_use", "id": "toolu_02", "name": "grep"}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "input_json_delta", "partial_json": '{"patt'}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "input_json_delta", "partial_json": 'ern": "x"}'}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
     "usage": {"output_tokens": 30}},
    {"type": "message_stop"},
]


class TestStreaming:
    async def test_sse_stream(self):
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.headers["x-api-key"] == "sk-test"
            assert req.headers["anthropic-version"] == "2023-06-01"
            return httpx.Response(
                200,
                content=_sse(STREAM_EVENTS).encode(),
                headers={"content-type": "text/event-stream"},
            )

        provider = AnthropicProvider(
            "claude-opus-4-8", api_key="sk-test",
            transport=httpx.MockTransport(handler),
        )
        events = [
            e
            async for e in provider.stream(
                CompletionRequest(messages=(Message.user("hi"),))
            )
        ]
        deltas = [e.text for e in events if isinstance(e, TextDelta)]
        assert deltas == ["Searching ", "now."]

        final = events[-1]
        assert isinstance(final, StreamCompleted)
        result = final.result
        assert result.message.text == "Searching now."
        # tool call accumulated from input_json_delta fragments
        call = result.message.tool_calls[0]
        assert (call.id, call.name, call.arguments) == ("toolu_02", "grep", {"pattern": "x"})
        assert result.stop_reason == "tool_calls"
        assert result.usage == result.usage.__class__(
            input_tokens=40, output_tokens=30, cache_read_tokens=100
        )

    async def test_complete_end_to_end(self):
        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/v1/messages"
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            )

        provider = AnthropicProvider(
            "claude-opus-4-8", api_key="sk-test",
            transport=httpx.MockTransport(handler),
        )
        result = await provider.complete(
            CompletionRequest(messages=(Message.user("hi"),))
        )
        assert result.message.text == "done"
        assert result.provider == "anthropic"
