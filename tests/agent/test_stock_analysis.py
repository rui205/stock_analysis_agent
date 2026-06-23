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


def test_include_web_search_false_omits_web_search_from_tools() -> None:
    """When ``include_web_search=False``, the agent must not expose web_search."""
    agent = StockAnalysisAgent(symbol="02319.HK", include_web_search=False)  # noqa: F841
    tool_names = {t.name for t in agent.tools}
    assert "get_stock_snapshot" in tool_names
    assert "web_search" not in tool_names


def test_include_web_search_false_does_not_initialize_web_search_providers() -> None:
    """The web_search providers stay uninitialized when web_search is off.

    This guarantees that, if the LLM somehow called the tool, it would
    raise a clear RuntimeError ("provider not initialized") instead of
    silently making HTTP calls.
    """
    # Snapshot baseline: web_search providers start unset.
    saved_sites = _SITE_LIST_PROVIDER.value
    saved_ws_cache = _WS_CACHE_PROVIDER.value
    _SITE_LIST_PROVIDER.value = None
    _WS_CACHE_PROVIDER.value = None
    try:
        StockAnalysisAgent(symbol="02319.HK", include_web_search=False)
        assert _SITE_LIST_PROVIDER.value is None
        assert _WS_CACHE_PROVIDER.value is None
    finally:
        _SITE_LIST_PROVIDER.value = saved_sites
        _WS_CACHE_PROVIDER.value = saved_ws_cache


def test_default_system_prompt_reflects_include_web_search_false() -> None:
    """The default prompt must explicitly tell the LLM web_search is unavailable."""
    agent = StockAnalysisAgent(symbol="02319.HK", include_web_search=False)
    prompt = agent.system_prompt_value
    assert "没有 web_search 工具" in prompt
    assert "视需要调用 web_search" not in prompt


def test_default_system_prompt_reflects_include_web_search_true() -> None:
    """The default prompt must include the 'use web_search' clause by default."""
    agent = StockAnalysisAgent(symbol="02319.HK", include_web_search=True)
    prompt = agent.system_prompt_value
    assert "视需要调用 web_search" in prompt
    assert "没有 web_search 工具" not in prompt


def test_include_web_search_property() -> None:
    """The agent exposes its ``include_web_search`` flag as a property."""
    a_on = StockAnalysisAgent(symbol="02319.HK", include_web_search=True)
    a_off = StockAnalysisAgent(symbol="02319.HK", include_web_search=False)
    assert a_on.include_web_search is True
    assert a_off.include_web_search is False


def test_include_web_search_false_with_empty_site_list_does_not_raise() -> None:
    """``site_list`` is unused when web_search is off, so an empty list is fine.

    The default constructor rejects ``site_list=[]`` because web_search
    is on by default; with web_search disabled, the rejection must not
    fire.
    """
    agent = StockAnalysisAgent(  # noqa: F841
        symbol="02319.HK",
        include_web_search=False,
        site_list=[],
    )
    assert "web_search" not in {t.name for t in agent.tools}


def test_load_skill_is_always_in_tools() -> None:
    """The load_skill tool is exposed regardless of web_search setting.

    Skill loading is a core capability of the agent (independent of
    network access), so it must be available even when web_search is off.
    """
    a_on = StockAnalysisAgent(symbol="02319.HK", include_web_search=True)
    a_off = StockAnalysisAgent(symbol="02319.HK", include_web_search=False)
    assert "load_skill" in {t.name for t in a_on.tools}
    assert "load_skill" in {t.name for t in a_off.tools}


def test_default_system_prompt_mentions_load_skill() -> None:
    """The default prompt must tell the LLM when to call load_skill."""
    agent = StockAnalysisAgent(symbol="02319.HK")
    prompt = agent.system_prompt_value
    assert "load_skill" in prompt
    assert "stock-snapshot-format" in prompt
    # The hint should mention the structured-report keywords that trigger loading.
    assert "公司画像" in prompt or "格式化" in prompt


def test_default_system_prompt_explains_structured_tool_output() -> None:
    """The default prompt must describe the structured JSON shape returned
    by get_stock_snapshot so the LLM knows how to read per-source data."""
    agent = StockAnalysisAgent(symbol="02319.HK")
    prompt = agent.system_prompt_value
    assert "tushare" in prompt
    assert "akshare" in prompt
    assert "mootdx" in prompt
    assert "data" in prompt
    assert "error" in prompt
