"""Hong Kong equity data source — wraps yfinance (`.HK` suffix).

yfinance handles HK shares using the same `Ticker` class as US shares, so
the implementation is nearly identical to `USStockSource`. The only
differences are: the market tag, the currency, and the symbol form
(`0700.HK` rather than `AAPL`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from ..errors import (
    RateLimitError,
    SymbolNotFoundError,
    UpstreamUnavailableError,
)
from ..models import (
    CompanyInfo,
    Fundamentals,
    Market,
    OHLCV,
    Quote,
    SearchResult,
)
from .base import normalize_symbol
from .us_stock import _PERIOD_TO_YF, _yfinance_call, USStockSource


class HKStockSource:
    """yfinance-backed data source for Hong Kong equities."""

    market: Market = Market.HK

    def __init__(self) -> None:
        self._source_name = "yfinance"
        # Reuse the US bars helper — the DataFrame layout is identical.
        self._us_helpers = USStockSource()

    @property
    def market_name(self) -> str:
        return Market.HK.value

    # --- public API ----------------------------------------------------

    def get_quote(self, symbol: str) -> Quote:
        sym = normalize_symbol(symbol, Market.HK)
        ticker = yf.Ticker(sym)
        info: dict[str, Any] = _yfinance_call(ticker.get_info) or {}
        if not info:
            raise SymbolNotFoundError(f"HK symbol not found: {sym!r}")

        if info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            # Fall back to a 5-day history window.
            hist: pd.DataFrame = _yfinance_call(ticker.history, period="5d")
            if hist is None or hist.empty:
                raise SymbolNotFoundError(f"HK symbol not found: {sym!r}")
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) >= 2 else latest
            price = float(latest["Close"])
            prev_close = float(prev["Close"])
            ts = (
                latest.name.to_pydatetime()
                if hasattr(latest.name, "to_pydatetime")
                else datetime.now(timezone.utc)
            )
        else:
            price = float(info.get("regularMarketPrice") or info.get("currentPrice"))
            prev_close = float(
                info.get("regularMarketPreviousClose") or info.get("previousClose") or price
            )
            ts_unix = info.get("regularMarketTime")
            ts = (
                datetime.fromtimestamp(ts_unix, tz=timezone.utc)
                if ts_unix
                else datetime.now(timezone.utc)
            )

        return Quote(
            symbol=sym,
            market=Market.HK,
            company_name=info.get("longName") or info.get("shortName"),
            currency=info.get("currency", "HKD"),
            price=price,
            change=price - prev_close,
            change_pct=(price - prev_close) / prev_close if prev_close else 0.0,
            as_of=ts,
            source=self._source_name,
        )

    def get_ohlcv(
        self, symbol: str, *, period: str = "1mo", limit: int = 30
    ) -> list[OHLCV]:
        sym = normalize_symbol(symbol, Market.HK)
        yf_period = _PERIOD_TO_YF.get(period, "1mo")
        ticker = yf.Ticker(sym)
        hist: pd.DataFrame = _yfinance_call(ticker.history, period=yf_period)
        if hist is None or hist.empty:
            raise SymbolNotFoundError(f"HK symbol not found: {sym!r}")
        return self._us_helpers._df_to_bars(hist.tail(limit))

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        sym = normalize_symbol(symbol, Market.HK)
        info: dict[str, Any] = _yfinance_call(yf.Ticker(sym).get_info) or {}
        if not info:
            raise SymbolNotFoundError(f"HK symbol not found: {sym!r}")
        return Fundamentals(
            symbol=sym,
            market=Market.HK,
            currency=info.get("currency", "HKD"),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            pb_ratio=info.get("priceToBook"),
            dividend_yield=info.get("dividendYield"),
            revenue=info.get("totalRevenue"),
            net_income=info.get("netIncomeToCommon"),
            fiscal_period=info.get("lastFiscalYearEnd"),
            source=self._source_name,
        )

    def get_company_info(self, symbol: str) -> CompanyInfo:
        sym = normalize_symbol(symbol, Market.HK)
        info: dict[str, Any] = _yfinance_call(yf.Ticker(sym).get_info) or {}
        if not info:
            raise SymbolNotFoundError(f"HK symbol not found: {sym!r}")
        return CompanyInfo(
            symbol=sym,
            market=Market.HK,
            name=info.get("longName") or info.get("shortName") or sym,
            name_local=None,
            industry=info.get("industry"),
            sector=info.get("sector"),
            exchange=info.get("exchange"),
            description=info.get("longBusinessSummary"),
            source=self._source_name,
        )

    def search_company(self, query: str, limit: int = 5) -> list[SearchResult]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            search = yf.Search(q, max_results=limit * 2)
        except YFRateLimitError as exc:
            raise RateLimitError(
                f"Yahoo Finance rate limit hit: {exc}", retry_after_seconds=60.0
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise UpstreamUnavailableError(f"yfinance search failed: {exc}") from exc

        quotes = getattr(search, "quotes", None) or []
        results: list[SearchResult] = []
        for q_ in quotes:
            sym = str(q_.get("symbol", ""))
            if not sym or not sym.endswith(".HK"):
                continue
            results.append(
                SearchResult(
                    symbol=sym,
                    market=Market.HK,
                    name=str(q_.get("longname") or q_.get("shortname") or sym),
                    name_local=None,
                    exchange=q_.get("exchange"),
                    match_type="substring",
                )
            )
            if len(results) >= limit:
                break
        return results


__all__ = ["HKStockSource"]
