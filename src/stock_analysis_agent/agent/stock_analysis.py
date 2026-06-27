"""StockAnalysisAgent: get_stock_snapshot + web_search → LLM-driven analysis.

This is a **low-level reusable agent** — it owns the tool wiring
(snapshot, web_search, load_skill) and the data-source / cache providers,
but it does **not** bake in any output schema or JSON contract. The
caller is responsible for supplying a ``system_prompt`` that defines the
shape the LLM should emit; this class only guarantees that whatever
schema the prompt asks for will reach the LLM, the tools will be
available, and the providers will be correctly initialized.

Typical callers (e.g. ``script.analyze_stock``) load a prompt template
from disk and pass it in.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stock_analysis_agent.agent.base import BaseAgent
from stock_analysis_agent.agent.deepsearch import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_TTL,
    DEFAULT_SITE_LIST,
)
from stock_analysis_agent.memory.file_cache import _FileCache
from stock_analysis_agent.tools.market_data import (
    ALL_SOURCES,
    _CACHE_PROVIDER as _MD_CACHE_PROVIDER,
    _SOURCES_PROVIDER,
    _get_stock_snapshot,
)
from stock_analysis_agent.tools.skill import load_skill
from stock_analysis_agent.tools.web_search import (
    _CACHE_PROVIDER as _WS_CACHE_PROVIDER,
    _SITE_LIST_PROVIDER,
    _web_search,
)

# Defaults (site list, cache dir, cache ttl) are imported from
# ``agent.deepsearch`` so a single source of truth governs both agents.


class StockAnalysisAgent(BaseAgent):
    """LLM-driven stock analysis agent.

    Bundles the ``get_stock_snapshot`` (multi-source quote), ``web_search``,
    and ``load_skill`` tools. The system prompt is **caller-supplied** —
    pass ``system_prompt=`` to define the output contract the LLM should
    follow. This class never infers a default prompt, so different callers
    can target different output schemas (e.g. a terse JSON, a structured
    Markdown report, a multi-section company profile) without subclassing.

    Construction mirrors :class:`DeepSearchAgent`: it writes into the
    module-level ``_Provider`` singletons used by the @tool callables,
    so calling ``StockAnalysisAgent(symbol=..., system_prompt=...)`` is
    enough to make all tools available.

    Concurrent multi-instance construction in one process is not
    supported (matches :class:`DeepSearchAgent`'s constraint).
    """

    def __init__(
        self,
        *,
        symbol: str,
        system_prompt: str,
        include_peers: bool = True,
        peer_count: int = 2,
        include_web_search: bool = True,
        site_list: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
        cache_ttl: float | None = DEFAULT_CACHE_TTL,
        max_retries: int = 3,
        recursion_limit: int = 6,
        **kwargs: Any,
    ) -> None:
        if not symbol:
            raise ValueError("symbol cannot be empty")
        if not system_prompt:
            raise ValueError("system_prompt cannot be empty")

        resolved_sites: list[str] = (
            list(site_list) if site_list is not None else list(DEFAULT_SITE_LIST)
        )
        if include_web_search and not resolved_sites:
            raise ValueError("site_list cannot be empty when web_search is enabled")

        resolved_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else Path(DEFAULT_CACHE_DIR).expanduser().resolve()
        )

        self._cache = _FileCache(resolved_dir, ttl_seconds=cache_ttl)
        self._symbol = symbol
        self._include_peers = include_peers
        self._peer_count = peer_count
        self._include_web_search = include_web_search

        # Single-instance provider writes — both @tool callables read
        # these via .get() on each invocation. ``market_data`` and
        # ``web_search`` each declare their own ``_CACHE_PROVIDER`` (they
        # are different module-level singletons), so we have to write to
        # both. Collapsing them into one shared singleton is a follow-up.
        # When ``include_web_search`` is False, skip the web_search providers
        # entirely so the LLM cannot accidentally call a half-initialized
        # _web_search.
        _SOURCES_PROVIDER.value = ALL_SOURCES
        _MD_CACHE_PROVIDER.value = self._cache
        if include_web_search:
            _WS_CACHE_PROVIDER.value = self._cache
            _SITE_LIST_PROVIDER.value = resolved_sites

        tools = [_get_stock_snapshot, load_skill]
        if include_web_search:
            tools.append(_web_search)

        super().__init__(
            system_prompt=system_prompt,
            max_retries=max_retries,
            recursion_limit=recursion_limit,
            tools=tools,
            **kwargs,
        )

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def include_peers(self) -> bool:
        return self._include_peers

    @property
    def peer_count(self) -> int:
        return self._peer_count

    @property
    def include_web_search(self) -> bool:
        return self._include_web_search


__all__ = ["StockAnalysisAgent"]
