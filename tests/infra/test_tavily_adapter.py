"""Tests for stock_analysis_agent.infra.tavily_adapter."""
from __future__ import annotations

import httpx
import pytest

from stock_analysis_agent.infra.tavily_adapter import (
    TavilyAdapter,
    TavilyAPIKeyError,
    TavilySearchError,
)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(TavilyAPIKeyError):
        TavilyAdapter()


def test_search_returns_raw_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query, **kwargs):  # type: ignore[no-untyped-def]
            return {"results": [{"title": "t", "url": "u", "content": "c"}]}

    monkeypatch.setattr(
        "stock_analysis_agent.infra.tavily_adapter.TavilyClient", _FakeClient
    )
    result = TavilyAdapter().search("q")
    assert result["results"][0]["title"] == "t"


def test_search_wraps_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            pass

        def search(self, query, **kwargs):  # type: ignore[no-untyped-def]
            raise httpx.ConnectError("down")

    monkeypatch.setattr(
        "stock_analysis_agent.infra.tavily_adapter.TavilyClient", _FakeClient
    )
    with pytest.raises(TavilySearchError, match="transport failed"):
        TavilyAdapter().search("q")


def test_empty_query_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    with pytest.raises(TavilySearchError, match="non-empty"):
        TavilyAdapter().search("   ")
