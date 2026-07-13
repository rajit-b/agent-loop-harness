"""EmbeddingProvider implementations. Default: Ollama /api/embed with
nomic-embed-text (A5). The Protocol itself lives in types.py."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from agentloop.types import ProviderError, TransientProviderError


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    "/api/embed", json={"model": self.model, "input": list(texts)}
                )
        except httpx.HTTPError as exc:
            raise TransientProviderError("ollama", f"embed: {exc}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientProviderError(
                "ollama", f"embed HTTP {response.status_code}"
            )
        if response.status_code != 200:
            raise ProviderError(
                "ollama", f"embed HTTP {response.status_code}: {response.text[:200]}"
            )
        return response.json()["embeddings"]
