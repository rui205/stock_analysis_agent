"""Tests for StockAnalysisAgent: provider injection, tools, system prompt."""
from __future__ import annotations

from stock_analysis_agent.agent.deepsearch import DEFAULT_SITE_LIST
from stock_analysis_agent.agent.stock_analysis import StockAnalysisAgent
from stock_analysis_agent.tools.market_data import (
    ALL_SOURCES,
    _CACHE_PROVIDER as _MD_CACHE_PROVIDER,
    _SOURCES_PROVIDER,
    _get_stock_snapshot,
)
from stock_analysis_agent.tools.web_search import (
    _CACHE_PROVIDER as _WS_CACHE_PROVIDER,
    _SITE_LIST_PROVIDER,
    _web_search,
)


def test_construction_populates_all_providers() -> None:
    agent = StockAnalysisAgent(symbol="02319.HK")  # noqa: F841
    assert _SOURCES_PROVIDER.get() == ALL_SOURCES
    assert _MD_CACHE_PROVIDER.get() is not None
    assert _WS_CACHE_PROVIDER.get() is not None
    assert _SITE_LIST_PROVIDER.get() == list(DEFAULT_SITE_LIST)


def test_tools_include_both_snapshot_and_web_search() -> None:
    agent = StockAnalysisAgent(symbol="02319.HK")  # noqa: F841
    tool_names = {t.name for t in agent.tools}
    assert "get_stock_snapshot" in tool_names
    assert "web_search" in tool_names


def test_default_system_prompt_contains_symbol() -> None:
    agent = StockAnalysisAgent(symbol="600519.SH")  # noqa: F841
    assert "600519.SH" in agent.system_prompt_value


def test_default_system_prompt_contains_all_json_keys() -> None:
    agent = StockAnalysisAgent(symbol="02319.HK")  # noqa: F841
    prompt = agent.system_prompt_value
    for key in (
        "symbol", "summary", "fundamentals", "technicals",
        "peer_compare", "news", "risks", "recommendation",
    ):
        assert f'"{key}"' in prompt, f"missing key in prompt: {key}"


def test_default_system_prompt_reflects_include_peers_true() -> None:
    agent = StockAnalysisAgent(symbol="02319.HK", include_peers=True)
    assert "include_peers 为 True" in agent.system_prompt_value
    assert "include_peers 为 False" not in agent.system_prompt_value


def test_default_system_prompt_reflects_include_peers_false() -> None:
    agent = StockAnalysisAgent(symbol="02319.HK", include_peers=False)
    assert "include_peers 为 False" in agent.system_prompt_value
    assert "include_peers 为 True" not in agent.system_prompt_value


def test_custom_system_prompt_overrides_default() -> None:
    agent = StockAnalysisAgent(symbol="02319.HK", system_prompt="hello world")
    assert agent.system_prompt_value == "hello world"


def test_underlying_tool_objects_match_module_references() -> None:
    """The two tools must be the same objects the @tool decorators exported."""
    agent = StockAnalysisAgent(symbol="02319.HK")
    tool_objs = list(agent.tools)
    assert _get_stock_snapshot in tool_objs
    assert _web_search in tool_objs
