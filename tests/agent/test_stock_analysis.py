"""Tests for StockAnalysisAgent: provider injection, tools, and prompt contract.

The agent is intentionally schema-agnostic — there is no built-in default
prompt, so these tests focus on what the agent *does* guarantee:

- provider singletons are populated on construction
- the right tools are exposed (and ``load_skill`` is always there)
- the caller's ``system_prompt`` is plumbed through verbatim
- a missing / empty ``system_prompt`` is rejected loudly
"""
from __future__ import annotations

import pytest

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


_TEST_PROMPT = "you are a test prompt for {symbol}"


def _agent(**overrides) -> StockAnalysisAgent:
    """Build a StockAnalysisAgent with the minimum required kwargs.

    Returns an agent instance for tests that don't care about the prompt
    contents. Tests that *do* care should pass ``system_prompt=`` directly.
    """
    return StockAnalysisAgent(symbol="02319.HK", system_prompt=_TEST_PROMPT, **overrides)


# ---------------------------------------------------------------------------
# provider wiring
# ---------------------------------------------------------------------------


def test_construction_populates_all_providers() -> None:
    _agent()
    assert _SOURCES_PROVIDER.get() == ALL_SOURCES
    assert _MD_CACHE_PROVIDER.get() is not None
    assert _WS_CACHE_PROVIDER.get() is not None
    assert _SITE_LIST_PROVIDER.get() == list(DEFAULT_SITE_LIST)


def test_underlying_tool_objects_match_module_references() -> None:
    """The two tools must be the same objects the @tool decorators exported."""
    tool_objs = list(_agent().tools)
    assert _get_stock_snapshot in tool_objs
    assert _web_search in tool_objs


# ---------------------------------------------------------------------------
# tool exposure
# ---------------------------------------------------------------------------


def test_tools_include_both_snapshot_and_web_search_by_default() -> None:
    tool_names = {t.name for t in _agent().tools}
    assert "get_stock_snapshot" in tool_names
    assert "web_search" in tool_names


def test_include_web_search_false_omits_web_search_from_tools() -> None:
    """When ``include_web_search=False``, the agent must not expose web_search."""
    tool_names = {t.name for t in _agent(include_web_search=False).tools}
    assert "get_stock_snapshot" in tool_names
    assert "web_search" not in tool_names


def test_include_web_search_false_does_not_initialize_web_search_providers() -> None:
    """The web_search providers stay uninitialized when web_search is off.

    This guarantees that, if the LLM somehow called the tool, it would
    raise a clear RuntimeError ("provider not initialized") instead of
    silently making HTTP calls.
    """
    saved_sites = _SITE_LIST_PROVIDER.value
    saved_ws_cache = _WS_CACHE_PROVIDER.value
    _SITE_LIST_PROVIDER.value = None
    _WS_CACHE_PROVIDER.value = None
    try:
        _agent(include_web_search=False)
        assert _SITE_LIST_PROVIDER.value is None
        assert _WS_CACHE_PROVIDER.value is None
    finally:
        _SITE_LIST_PROVIDER.value = saved_sites
        _WS_CACHE_PROVIDER.value = saved_ws_cache


def test_include_web_search_false_with_empty_site_list_does_not_raise() -> None:
    """``site_list`` is unused when web_search is off, so an empty list is fine.

    The default constructor rejects ``site_list=[]`` because web_search
    is on by default; with web_search disabled, the rejection must not
    fire.
    """
    agent = _agent(include_web_search=False, site_list=[])
    assert "web_search" not in {t.name for t in agent.tools}


def test_load_skill_is_always_in_tools() -> None:
    """The load_skill tool is exposed regardless of web_search setting.

    Skill loading is a core capability of the agent (independent of
    network access), so it must be available even when web_search is off.
    """
    a_on = _agent(include_web_search=True)
    a_off = _agent(include_web_search=False)
    assert "load_skill" in {t.name for t in a_on.tools}
    assert "load_skill" in {t.name for t in a_off.tools}


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


def test_include_web_search_property() -> None:
    a_on = _agent(include_web_search=True)
    a_off = _agent(include_web_search=False)
    assert a_on.include_web_search is True
    assert a_off.include_web_search is False


# ---------------------------------------------------------------------------
# system_prompt — required, plumbed through verbatim, no default
# ---------------------------------------------------------------------------


def test_system_prompt_is_plumbed_through_verbatim() -> None:
    """The agent must pass ``system_prompt`` to the LLM as-is, with no mutation."""
    agent = StockAnalysisAgent(
        symbol="02319.HK", system_prompt="hello world {symbol}",
    )
    assert agent.system_prompt_value == "hello world {symbol}"


def test_system_prompt_is_required() -> None:
    """``system_prompt`` has no default — the caller must own the schema."""
    with pytest.raises(TypeError):
        StockAnalysisAgent(symbol="02319.HK")  # type: ignore[call-arg]


def test_empty_system_prompt_is_rejected() -> None:
    """An empty string would silently send a blank instruction to the LLM."""
    with pytest.raises(ValueError, match="system_prompt"):
        StockAnalysisAgent(symbol="02319.HK", system_prompt="")


def test_empty_symbol_is_rejected() -> None:
    """Symbol is the primary key — empty must fail loudly."""
    with pytest.raises(ValueError, match="symbol"):
        StockAnalysisAgent(symbol="", system_prompt=_TEST_PROMPT)
