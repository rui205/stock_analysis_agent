"""LangChain tool definitions — thin wrappers over the data sources.

Each tool is a `@tool`-decorated function whose docstring becomes the
description the LLM sees. Tool bodies are short: parse inputs, call the
right data source, format the result as a string.

On any data-source failure, the tool returns a string beginning with
`"ERROR: "`. This lets the agent decide whether to retry, switch market,
or surface the failure to the user — rather than crashing the whole loop.
"""

from __future__ import annotations

from typing import Literal

from langchain.tools import tool

from ..data_sources import get_data_source
from ..errors import DataSourceError
from ..models import Market

MarketArg = Literal["a_share", "us", "hk"]


def _market(arg: str) -> Market:
    return Market(arg)


def _format_quote(q) -> str:
    pct = q.change_pct * 100
    sign = "+" if q.change >= 0 else ""
    as_of = q.as_of.strftime("%Y-%m-%d %H:%M")
    name = f" ({q.company_name})" if q.company_name else ""
    return (
        f"{q.symbol}{name} [{q.market.value}]\n"
        f"  Price:   {q.currency} {q.price:,.2f}\n"
        f"  Change:  {sign}{q.change:,.2f} ({sign}{pct:.2f}%)\n"
        f"  As of:   {as_of}\n"
        f"  Source:  {q.source}"
    )


def _format_ohlcv(symbol: str, market: Market, bars) -> str:
    if not bars:
        return f"No OHLCV data for {symbol} [{market.value}]."
    header = f"{symbol} [{market.value}] — last {len(bars)} daily bars:"
    lines = [header, "  date        open       high       low        close      volume"]
    for b in bars:
        lines.append(
            f"  {b.date.isoformat()}  "
            f"{b.open:>9,.2f}  {b.high:>9,.2f}  {b.low:>9,.2f}  "
            f"{b.close:>9,.2f}  {b.volume:>14,.0f}"
        )
    return "\n".join(lines)


def _format_fundamentals(f) -> str:
    def _fmt(value, suffix=""):
        if value is None:
            return "  n/a"
        return f"  {value:,.2f}{suffix}"

    return (
        f"{f.symbol} [{f.market.value}] fundamentals ({f.source}):\n"
        f"  Currency:        {f.currency or 'n/a'}\n"
        f"  Market cap:      {_fmt(f.market_cap)}\n"
        f"  P/E ratio:       {_fmt(f.pe_ratio)}\n"
        f"  P/B ratio:       {_fmt(f.pb_ratio)}\n"
        f"  Dividend yield:  {_fmt(f.dividend_yield)}\n"
        f"  Revenue:         {_fmt(f.revenue)}\n"
        f"  Net income:      {_fmt(f.net_income)}\n"
        f"  Fiscal period:   {f.fiscal_period or 'n/a'}"
    )


def _format_company_info(c) -> str:
    return (
        f"{c.symbol} [{c.market.value}] — {c.name}\n"
        f"  Local name: {c.name_local or 'n/a'}\n"
        f"  Industry:   {c.industry or 'n/a'}\n"
        f"  Sector:     {c.sector or 'n/a'}\n"
        f"  Exchange:   {c.exchange or 'n/a'}\n"
        f"  Source:     {c.source}\n"
        f"  Description: {c.description or 'n/a'}"
    )


def _format_search(hits) -> str:
    if not hits:
        return "No matching companies found."
    lines = ["Search results:"]
    for h in hits:
        local = f" / {h.name_local}" if h.name_local else ""
        exch = f" ({h.exchange})" if h.exchange else ""
        lines.append(f"  {h.symbol}{exch} — {h.name}{local} [{h.market.value}]")
    return "\n".join(lines)


# --- The four tools ----------------------------------------------------------


@tool
def get_quote(ticker: str, market: MarketArg) -> str:
    """Get the latest price quote for a single stock.

    Use this when the user asks for the current price, latest close, or
    a "what is X trading at" question. Returns the price, change vs.
    previous close, and the timestamp of the quote.

    Args:
        ticker: The ticker symbol. A-share examples: "600519" (Kweichow
            Moutai), "000001" (Ping An Bank). US examples: "AAPL", "MSFT",
            "TSLA". HK examples: "0700" or "0700.HK" (Tencent), "9988"
            or "9988.HK" (Alibaba).
        market: Which market the ticker belongs to. Must be one of
            "a_share" (沪深北), "us" (United States), or "hk" (Hong Kong).
    """
    try:
        src = get_data_source(_market(market))
        quote = src.get_quote(ticker)
    except DataSourceError as exc:
        return f"ERROR: {exc}"
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        return f"ERROR: Unexpected failure fetching quote: {exc}"
    return _format_quote(quote)


@tool
def get_ohlcv(
    ticker: str,
    market: MarketArg,
    period: Literal["1mo", "3mo", "6mo", "1y", "5y"] = "1mo",
    limit: int = 30,
) -> str:
    """Get daily OHLCV (open/high/low/close/volume) bars for a stock.

    Use this when the user asks for price history, charts, "how has X
    performed over the last N months", or moving-average context.

    Args:
        ticker: The ticker symbol (same conventions as get_quote).
        market: One of "a_share", "us", "hk".
        period: How far back to look. One of "1mo", "3mo", "6mo",
            "1y", "5y". The actual number of returned bars is the
            smaller of `limit` and the number of trading days in
            the period.
        limit: Maximum number of daily bars to return (default 30).
    """
    try:
        src = get_data_source(_market(market))
        bars = src.get_ohlcv(ticker, period=period, limit=limit)
    except DataSourceError as exc:
        return f"ERROR: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Unexpected failure fetching OHLCV: {exc}"
    return _format_ohlcv(
        normalize_ticker_for_display(ticker, _market(market)),
        _market(market),
        bars,
    )


@tool
def get_fundamentals(ticker: str, market: MarketArg) -> str:
    """Get fundamental data for a stock (market cap, P/E, P/B, revenue, etc.).

    Use this when the user asks about valuation, financial health,
    earnings, or "is X expensive" questions. Field coverage varies by
    market — missing values are reported as "n/a".

    Args:
        ticker: The ticker symbol.
        market: One of "a_share", "us", "hk".
    """
    try:
        src = get_data_source(_market(market))
        fund = src.get_fundamentals(ticker)
    except DataSourceError as exc:
        return f"ERROR: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Unexpected failure fetching fundamentals: {exc}"
    return _format_fundamentals(fund)


@tool
def search_company(
    query: str,
    market: MarketArg | None = None,
    limit: int = 5,
) -> str:
    """Search for a company by name or ticker across the supported markets.

    Use this when the user mentions a company by name (e.g. "Tencent",
    "Apple", "贵州茅台") and you need to resolve it to a canonical
    ticker before calling get_quote, get_ohlcv, or get_fundamentals.

    Args:
        query: The name (full or partial) or ticker to search for.
            Matching is case-insensitive substring on both code and name.
        market: Optional — restrict the search to one market. If omitted,
            searches all three and interleaves results.
        limit: Maximum number of results to return (default 5).
    """
    if not query or not query.strip():
        return "ERROR: query must be a non-empty string."

    markets: list[Market] = (
        [_market(market)] if market else [Market.A_SHARE, Market.US, Market.HK]
    )
    all_hits = []
    try:
        per_market = max(1, limit // len(markets))
        for m in markets:
            src = get_data_source(m)
            hits = src.search_company(query, limit=per_market)
            all_hits.extend(hits)
    except DataSourceError as exc:
        return f"ERROR: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Unexpected failure during search: {exc}"

    if not all_hits:
        return f"No companies matched {query!r}."
    return _format_search(all_hits[:limit])


# --- helpers ---------------------------------------------------------------


def normalize_ticker_for_display(ticker: str, market: Market) -> str:
    """Normalize a ticker for display in tool output. Tools don't return
    this for the LLM (the LLM already knows what it asked for); it's used
    only to format messages to humans.
    """
    from ..data_sources import normalize_symbol  # local import to avoid cycle

    try:
        return normalize_symbol(ticker, market)
    except Exception:  # noqa: BLE001
        return ticker


# The list the agent factory binds. Order matters only for tool-list
# presentation; semantics don't depend on order.
ALL_TOOLS = [get_quote, get_ohlcv, get_fundamentals, search_company]


__all__ = [
    "ALL_TOOLS",
    "get_fundamentals",
    "get_ohlcv",
    "get_quote",
    "search_company",
]
