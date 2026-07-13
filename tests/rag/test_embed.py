"""OllamaEmbedder wire format and error mapping."""

from __future__ import annotations

import json

import httpx
import pytest

from agentloop.rag.embed import OllamaEmbedder
from agentloop.types import ProviderError, TransientProviderError


class TestOllamaEmbedder:
    async def test_request_and_response_shape(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/embed"
            body = json.loads(request.content)
            assert body == {"model": "nomic-embed-text", "input": ["a", "b"]}
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

        embedder = OllamaEmbedder(transport=httpx.MockTransport(handler))
        vectors = await embedder.embed(["a", "b"])
        assert vectors == [[0.1, 0.2], [0.3, 0.4]]

    async def test_empty_input_short_circuits(self):
        def handler(request):  # pragma: no cover - must never be called
            raise AssertionError("no HTTP call expected")

        embedder = OllamaEmbedder(transport=httpx.MockTransport(handler))
        assert await embedder.embed([]) == []

    async def test_5xx_is_transient(self):
        embedder = OllamaEmbedder(
            transport=httpx.MockTransport(lambda r: httpx.Response(503))
        )
        with pytest.raises(TransientProviderError):
            await embedder.embed(["x"])

    async def test_4xx_is_not(self):
        embedder = OllamaEmbedder(
            transport=httpx.MockTransport(lambda r: httpx.Response(404, text="no model"))
        )
        with pytest.raises(ProviderError) as exc_info:
            await embedder.embed(["x"])
        assert not isinstance(exc_info.value, TransientProviderError)
