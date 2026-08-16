"""Tests for stock_analysis_agent.tools.web_search (Tavily backend)."""
from __future__ import annotations

from pathlib import Path

import pytest

from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.memory import _FileCache
from stock_analysis_agent.tools import web_search as ws


@pytest.fixture(autouse=True)
def _reset_tavily_adapter() -> None:
    """Reset the TavilyAdapter singleton so each test builds a fresh instance."""
    ws._TAVILY_ADAPTER = None
    yield
    ws._TAVILY_ADAPTER = None


def _wire_cache(tmp_path: Path) -> _FileCache:
    cache = _FileCache(tmp_path, ttl_seconds=60.0)
    ws._CACHE_PROVIDER.value = cache
    return cache


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(query)
        return {
            "answer": "summary",
            "results": [{"title": "t", "url": "u", "content": "c"}],
        }


def test_format_tavily_results_with_answer() -> None:
    out = ws._format_tavily_results(
        {"answer": "A", "results": [{"title": "T", "url": "U", "content": "C"}]}
    )
    assert "[answer]" in out and "A" in out
    assert "[1] T" in out and "U" in out and "C" in out


def test_format_tavily_results_no_results() -> None:
    assert ws._format_tavily_results({"results": []}) == "[no results]"


def test_web_search_cache_hit_skips_tavily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _wire_cache(tmp_path)
    cache.set(site="tavily", query="q", text="cached")
    fake = _FakeAdapter()
    monkeypatch.setattr(ws, "TavilyAdapter", lambda: fake)
    assert ws._web_search.invoke({"query": "q"}) == "cached"
    assert fake.calls == []


def test_web_search_cache_miss_calls_tavily_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _wire_cache(tmp_path)
    fake = _FakeAdapter()
    monkeypatch.setattr(ws, "TavilyAdapter", lambda: fake)
    out = ws._web_search.invoke({"query": "q"})
    assert fake.calls == ["q"]
    assert "[1] t" in out
    assert cache.get(site="tavily", query="q") == out


def test_web_search_tavily_error_wraps_tool_execution_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stock_analysis_agent.infra.tavily_adapter import TavilySearchError

    _wire_cache(tmp_path)

    class _Boom:
        def search(self, query: str, **kwargs):  # type: ignore[no-untyped-def]
            raise TavilySearchError("boom")

    monkeypatch.setattr(ws, "TavilyAdapter", _Boom)
    with pytest.raises(ToolExecutionError, match="web_search failed"):
        ws._web_search.invoke({"query": "q"})


def test_web_search_tool_metadata() -> None:
    assert ws._web_search.name == "web_search"
    schema = ws._web_search.args
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    if isinstance(schema, dict) and "properties" in schema:
        properties = schema["properties"]
    else:
        properties = schema
    assert "query" in (properties or {})
