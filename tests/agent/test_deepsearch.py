"""Tests for stock_analysis_agent.agent.deepsearch.DeepSearchAgent."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

from stock_analysis_agent.agent.deepsearch import DeepSearchAgent

from tests.agent.conftest import ToolAwareFakeChatModel, make_ai, make_tool_call


def test_default_construction_uses_module_constants(tmp_path: Path) -> None:
    """No-arg construction uses DEFAULT_SITE_LIST, DEFAULT_SYSTEM_PROMPT, max_retries=3."""
    from stock_analysis_agent.agent.deepsearch import (
        DEFAULT_SITE_LIST,
        DEFAULT_SYSTEM_PROMPT,
    )

    agent = DeepSearchAgent(cache_dir=tmp_path, cache_ttl=None)
    assert agent.site_list == DEFAULT_SITE_LIST
    assert agent.system_prompt_value == DEFAULT_SYSTEM_PROMPT
    assert agent.max_retries == 3
    assert agent.cache_dir == tmp_path.resolve()


def test_custom_site_list_overrides_default(tmp_path: Path) -> None:
    agent = DeepSearchAgent(
        site_list=["https://x.test"], cache_dir=tmp_path, cache_ttl=None
    )
    assert agent.site_list == ["https://x.test"]


def test_custom_system_prompt_overrides_default(tmp_path: Path) -> None:
    agent = DeepSearchAgent(
        system_prompt="custom prompt", cache_dir=tmp_path, cache_ttl=None
    )
    assert agent.system_prompt_value == "custom prompt"


def test_site_list_returns_copy(tmp_path: Path) -> None:
    """Mutating the returned site_list must not affect the agent or DEFAULT_SITE_LIST."""
    from stock_analysis_agent.agent.deepsearch import DEFAULT_SITE_LIST

    agent = DeepSearchAgent(cache_dir=tmp_path, cache_ttl=None)
    snapshot = agent.site_list
    snapshot.append("https://mutated.test")
    assert "https://mutated.test" not in agent.site_list
    assert "https://mutated.test" not in DEFAULT_SITE_LIST


def test_empty_site_list_raises_at_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="site_list cannot be empty"):
        DeepSearchAgent(site_list=[], cache_dir=tmp_path, cache_ttl=None)


def test_kwargs_pass_through_to_base_agent(tmp_path: Path) -> None:
    """model, temperature, name flow through to BaseAgent properties."""
    agent = DeepSearchAgent(
        model="claude-opus-4-8",
        temperature=0.7,
        name="custom",
        cache_dir=tmp_path,
        cache_ttl=None,
    )
    assert agent.model == "claude-opus-4-8"
    assert agent.temperature == 0.7
    assert agent.name == "custom"


def test_cache_dir_expands_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`~` in a str cache_dir is expanded via Path.expanduser()."""
    monkeypatch.setenv("HOME", str(tmp_path))
    agent = DeepSearchAgent(cache_dir="~/my-cache", cache_ttl=None)
    assert agent.cache_dir == (tmp_path / "my-cache").resolve()


def test_cache_ttl_none_disables_expiration(tmp_path: Path) -> None:
    """cache_ttl=None means cache entries never expire."""
    agent = DeepSearchAgent(cache_dir=tmp_path, cache_ttl=None)
    agent._cache.set(site="https://a.test", query="q", text="cached")
    assert agent._cache.get(site="https://a.test", query="q") == "cached"
    assert agent.cache_ttl is None  # use public property


def test_web_search_provider_reflects_latest_construction(tmp_path: Path) -> None:
    """After constructing a DeepSearchAgent with site_list=[B], the
    module-level site provider exposes [B] (single-instance contract)."""
    from stock_analysis_agent.tools.web_search import _SITE_LIST_PROVIDER

    DeepSearchAgent(site_list=["https://b.test"], cache_dir=tmp_path, cache_ttl=None)
    assert _SITE_LIST_PROVIDER.get() == ["https://b.test"]


def _bypass_retry_middleware():  # type: ignore[no-untyped-def]
    """Build an AgentMiddleware that bypasses BaseAgent's retry middleware."""

    class _NoRetry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return handler(request)

        async def awrap_tool_call(self, request, handler):  # type: ignore[no-untyped-def]
            return await handler(request)

    return _NoRetry()


def test_tool_call_reaches_web_search_with_configured_sites(
    tmp_path: Path,
) -> None:
    """When the LLM produces a web_search tool_call, the configured sites
    are actually fetched (verified via MockTransport request log)."""
    from langchain.agents import create_agent

    from stock_analysis_agent.tools.web_search import _web_search

    seen_urls: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, text="<p>snippet</p>")

    transport = httpx.MockTransport(_handler)

    # Build the agent first so providers are populated.
    agent = DeepSearchAgent(
        site_list=["https://a.test", "https://b.test"],
        cache_dir=tmp_path,
        cache_ttl=None,
    )

    # Monkey-patch _fetch_and_concat to inject the MockTransport. This
    # is the smallest surgical way to avoid real HTTP without touching
    # the production API surface.
    import stock_analysis_agent.tools.web_search as ws_mod

    original = ws_mod._fetch_and_concat

    async def _patched(query, sites, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = transport
        return await original(query, sites, **kwargs)

    ws_mod._fetch_and_concat = _patched
    try:
        # Fake model: tool_call, then final answer.
        first = make_ai("")
        first.tool_calls = [make_tool_call("web_search", {"query": "AI safety"}, "call_1")]
        model = ToolAwareFakeChatModel(responses=[first, make_ai("answer")])

        graph = create_agent(
            model=model,
            tools=[_web_search],
            system_prompt=agent.system_prompt_value,
            middleware=[_bypass_retry_middleware()],
        )
        agent._build_graph = lambda: graph  # type: ignore[method-assign]

        # Drain the stream; we don't care about events, just that the
        # tool ran and we got a final answer.
        for _ in agent.stream([HumanMessage(content="research AI safety")]):
            pass
    finally:
        ws_mod._fetch_and_concat = original

    # Both configured sites were hit (with q=AI safety).
    assert len(seen_urls) == 2, f"expected 2 sites, got {seen_urls!r}"
    assert all("q=AI+safety" in u or "q=AI%20safety" in u or "q=AI safety" in u for u in seen_urls), (
        f"query not threaded into URL: {seen_urls!r}"
    )


def test_second_agent_overwrites_provider_keeps_first_agent_intact(
    tmp_path: Path,
) -> None:
    """Single-instance contract: when a second DeepSearchAgent is constructed,
    the module-level _SITE_LIST_PROVIDER is overwritten (so @tool _web_search
    uses the new sites), but the first agent's `agent.site_list` keeps its
    original sites because each instance stores its own config on self."""
    from stock_analysis_agent.tools.web_search import _SITE_LIST_PROVIDER

    agent1 = DeepSearchAgent(
        site_list=["https://first.test"], cache_dir=tmp_path, cache_ttl=None
    )
    assert agent1.site_list == ["https://first.test"]
    assert _SITE_LIST_PROVIDER.get() == ["https://first.test"]

    DeepSearchAgent(
        site_list=["https://second.test"], cache_dir=tmp_path, cache_ttl=None
    )

    # Provider is overwritten — the @tool _web_search will use these sites.
    assert _SITE_LIST_PROVIDER.get() == ["https://second.test"]
    # First agent's snapshot is unchanged — it owns its own config.
    assert agent1.site_list == ["https://first.test"]