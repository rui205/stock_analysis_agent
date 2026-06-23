"""StockAnalysisAgent: get_stock_snapshot + web_search → structured JSON analysis."""
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
from stock_analysis_agent.tools.web_search import (
    _CACHE_PROVIDER as _WS_CACHE_PROVIDER,
    _SITE_LIST_PROVIDER,
    _web_search,
)

# Defaults (site list, cache dir, cache ttl) are imported from
# ``agent.deepsearch`` so a single source of truth governs both agents.


_DEFAULT_PROMPT_TEMPLATE: str = """\
你是一个股票分析助手。当前要分析的股票代码是: {symbol}。

你必须:
1. 调用 get_stock_snapshot 工具,参数 symbol="{symbol}",获取实时行情(必做)。
   返回值是结构化 JSON:
     - 顶层键: <symbol>、可选 peers、fetched_at
     - <symbol> 下面有 tushare / akshare / mootdx 三个数据源
     - 每个源要么 {{"data": <row dict>, "row_index": int}} 要么 {{"error": {{...}}}}
   在 fundamentals / technicals / peer_compare 字段里引用数据时,标注来源
   (例如 "tushare 报 PE=11.03")。
2. {web_search_clause}
3. 整合两类信息,产生**严格 JSON**,匹配下面的 schema:
   {{
     "symbol": "{symbol}",
     "summary": "<100~200 字总体观点>",
     "fundamentals": "<行业、PE/PB、市值等基本面要点,2~4 句>",
     "technicals": "<现价、涨跌、量能等技术面要点,2~4 句>",
     "peer_compare": "<同行对比要点,2~4 句;若 {include_clause} 写 'N/A'>",
     "news": "<基于 web_search 的近期关键新闻,1~3 条要点>",
     "risks": "<主要风险点,1~3 条>",
     "recommendation": "<明确操作建议:关注/观望/减仓,1 句>"
   }}
4. 输出**只**包含这一个 JSON,不要 markdown 代码块、不要解释、不要多余文字。
"""


_WEB_SEARCH_ENABLED_CLAUSE: str = (
    "视需要调用 web_search 补充近期新闻/公告/分析师观点。"
)
_WEB_SEARCH_DISABLED_CLAUSE: str = (
    "**没有 web_search 工具**(搜索引擎被屏蔽/不可用)。仅依靠 "
    "get_stock_snapshot 的数据 + 你自己的训练知识,不要尝试调用 web_search,"
    "news 字段若无法补充就写 'N/A'。"
)


def _build_default_prompt(
    symbol: str, include_peers: bool, include_web_search: bool
) -> str:
    """Render the default Chinese system prompt for the given symbol and flags."""
    include_clause = "include_peers 为 True" if include_peers else "include_peers 为 False"
    web_search_clause = (
        _WEB_SEARCH_ENABLED_CLAUSE
        if include_web_search
        else _WEB_SEARCH_DISABLED_CLAUSE
    )
    return _DEFAULT_PROMPT_TEMPLATE.format(
        symbol=symbol,
        include_clause=include_clause,
        web_search_clause=web_search_clause,
    )


class StockAnalysisAgent(BaseAgent):
    """LLM-driven stock analysis agent.

    Bundles the existing ``get_stock_snapshot`` (multi-source quote) and
    ``web_search`` tools. The system prompt directs the LLM to emit a
    strict JSON payload matching :class:`StockAnalysis`.

    Construction mirrors :class:`DeepSearchAgent`: it writes into the
    module-level ``_Provider`` singletons used by the @tool callables,
    so calling ``StockAnalysisAgent(symbol=...)`` is enough to make
    both tools available.

    Concurrent multi-instance construction in one process is not
    supported (matches :class:`DeepSearchAgent`'s constraint).
    """

    def __init__(
        self,
        *,
        symbol: str,
        include_peers: bool = True,
        peer_count: int = 2,
        include_web_search: bool = True,
        site_list: Sequence[str] | None = None,
        cache_dir: str | Path | None = None,
        cache_ttl: float | None = DEFAULT_CACHE_TTL,
        max_retries: int = 3,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not symbol:
            raise ValueError("symbol cannot be empty")

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

        resolved_prompt = (
            system_prompt
            if system_prompt is not None
            else _build_default_prompt(symbol, include_peers, include_web_search)
        )

        tools = [_get_stock_snapshot]
        if include_web_search:
            tools.append(_web_search)

        super().__init__(
            system_prompt=resolved_prompt,
            max_retries=max_retries,
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
