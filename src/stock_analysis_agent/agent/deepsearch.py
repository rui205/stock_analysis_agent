"""DeepSearchAgent: an LLM-driven deep-research agent.

Wraps a single @tool _web_search function that fans out to a configured
list of external search endpoints (httpx + stdlib HTML parser + file cache).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stock_analysis_agent.agent.base import BaseAgent
from stock_analysis_agent.memory.file_cache import _FileCache
from stock_analysis_agent.tools.web_search import (
    _CACHE_PROVIDER,
    _SITE_LIST_PROVIDER,
    _web_search,
)


DEFAULT_SYSTEM_PROMPT: str = (
    "You are a deep research agent. Given a user question, "
    "use the web_search tool to gather information from the "
    "configured sites, then synthesize a concise answer. "
    "Cite the source site in parentheses when you use a fact."
)

DEFAULT_SITE_LIST: list[str] = [
    "https://duckduckgo.com/html/",
    "https://www.bing.com/search",
    "https://html.duckduckgo.com/html/",
]

DEFAULT_CACHE_DIR: str = "~/.cache/stock-analysis-agent"
DEFAULT_CACHE_TTL: float | None = 86400.0  # 24h in seconds


class DeepSearchAgent(BaseAgent):
    """LLM-driven deep-research agent that searches a configured site list.

    Adds a single tool (`web_search`) that fans out to the configured
    external sites, fetches each concurrently via httpx, caches results
    to local JSON files, and returns aggregated plain text. The LLM
    decides what to search and when to synthesize.

    Construction overrides `BaseAgent`'s `max_retries` default from 2 → 3.
    Other BaseAgent parameters (model, temperature, name, ...) flow
    through via **kwargs.

    Single-instance: constructing a second agent updates the module-level
    _SITE_LIST_PROVIDER and _CACHE_PROVIDER used by the @tool _web_search.
    """

    def __init__(
        self,
        *,
        site_list: Sequence[str] | None = None,
        system_prompt: str | None = None,
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
        cache_ttl: float | None = DEFAULT_CACHE_TTL,
        **kwargs: Any,
    ) -> None:
        resolved_sites = list(site_list) if site_list is not None else list(DEFAULT_SITE_LIST)
        if not resolved_sites:
            raise ValueError("site_list cannot be empty")

        resolved_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

        resolved_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else Path(DEFAULT_CACHE_DIR).expanduser().resolve()
        )
        # `cache_ttl` defaults to DEFAULT_CACHE_TTL when omitted; an explicit
        # `None` disables expiration; an explicit float sets a custom TTL.
        # No sentinel needed because the function default IS the resolution.

        self._cache = _FileCache(resolved_dir, ttl_seconds=cache_ttl)
        self._site_list = resolved_sites

        # Single-instance: write into module-level providers so the @tool
        # callable (which is module-level) can read them.
        _SITE_LIST_PROVIDER.value = resolved_sites
        _CACHE_PROVIDER.value = self._cache

        super().__init__(
            system_prompt=resolved_prompt,
            max_retries=max_retries,
            tools=[_web_search],
            **kwargs,
        )

    @property
    def site_list(self) -> list[str]:
        return list(self._site_list)

    @property
    def cache_dir(self) -> Path:
        return self._cache._dir

    @property
    def cache_ttl(self) -> float | None:
        return self._cache._ttl