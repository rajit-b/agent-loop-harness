"""Shared HTTP plumbing for adapters: error mapping onto the §4 taxonomy."""

from __future__ import annotations

import httpx

from agentloop.types import ProviderError, TransientProviderError


def map_status(provider: str, response: httpx.Response) -> ProviderError:
    """HTTP status → taxonomy. 429/5xx are retryable; other 4xx are not."""
    detail = response.text[:500]
    message = f"HTTP {response.status_code}: {detail}"
    if response.status_code == 429 or response.status_code >= 500:
        return TransientProviderError(provider, message)
    return ProviderError(provider, message)


def map_transport(provider: str, exc: httpx.HTTPError) -> TransientProviderError:
    """Connection failures and timeouts are always retryable."""
    return TransientProviderError(provider, f"{type(exc).__name__}: {exc}")
