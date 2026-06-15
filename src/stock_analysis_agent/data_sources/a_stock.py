"""A-share (沪深北) data source — wraps akshare.

akshare is an open-source aggregator for Chinese-market public data. Its
function surface is large and module-level (no classes), so we wrap a
small, stable set of calls here. Tools never call `akshare.*` directly.

Data-source attribution: https://github.com/akfamily/akshare
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import akshare as ak
import pandas as pd

from ..errors import SymbolNotFoundError, UpstreamUnavailableError
from ..models import (
    CompanyInfo,
    Fundamentals,
    Market,
    OHLCV,
    Quote,
    SearchResult,
)
from .base import normalize_symbol

_A_SHARE_CODE_NAME_CACHE: pd.DataFrame | None = None


def _to_akshare_date(d: date) -> str:
    return d.strftime("%Y%m%d")


class AStockSource:
    """akshare-backed data source for A-shares (沪深北)."""

    market: Market = Market.A_SHARE

    def __init__(self) -> None:
        self._source_name = "akshare"

    @property
    def market_name(self) -> str:
        return Market.A_SHARE.value

    # --- private helpers ------------------------------------------------

    def _recent_daily_bars(self, symbol: str, days: int = 10) -> pd.DataFrame:
        """Return the last `days` daily bars for `symbol` as a DataFrame.

        Using a tight date range keeps the response small for what is
        effectively a "latest price" lookup.
        """
        today = date.today()
        start = today - timedelta(days=max(days, 5))
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=_to_akshare_date(start),
                end_date=_to_akshare_date(today),
                adjust="qfq",
            )
        except Exception as exc:  # noqa: BLE001 - we re-wrap with our own type
            raise UpstreamUnavailableError(
                f"akshare.stock_zh_a_hist failed for {symbol!r}: {exc}"
            ) from exc

        if df is None or df.empty:
            raise SymbolNotFoundError(f"A-share symbol not found: {symbol!r}")
        return df

    @staticmethod
    def _load_code_name_table() -> pd.DataFrame:
        """Load and cache the A-share symbol→name table."""
        global _A_SHARE_CODE_NAME_CACHE
        if _A_SHARE_CODE_NAME_CACHE is None:
            _A_SHARE_CODE_NAME_CACHE = ak.stock_info_a_code_name()
        return _A_SHARE_CODE_NAME_CACHE

    # --- public API ----------------------------------------------------

    def get_quote(self, symbol: str) -> Quote:
        sym = normalize_symbol(symbol, Market.A_SHARE)
        df = self._recent_daily_bars(sym, days=10)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        price = float(latest["收盘"])
        prev_close = float(prev["收盘"])
        change = price - prev_close
        change_pct = (change / prev_close) if prev_close else 0.0
        # Parse the date column (akshare returns it as YYYY-MM-DD or YYYYMMDD).
        latest_date = pd.to_datetime(latest["日期"]).to_pydatetime()
        return Quote(
            symbol=sym,
            market=Market.A_SHARE,
            company_name=self._name_for(sym),
            currency="CNY",
            price=price,
            change=change,
            change_pct=change_pct,
            as_of=latest_date,
            source=self._source_name,
        )

    def get_ohlcv(
        self, symbol: str, *, period: str = "1mo", limit: int = 30
    ) -> list[OHLCV]:
        # akshare period strings are different from yfinance's. We map a
        # small set of human-friendly buckets to a date range.
        sym = normalize_symbol(symbol, Market.A_SHARE)
        days = {"1mo": 31, "3mo": 95, "6mo": 185, "1y": 370, "5y": 1850}.get(
            period, 31
        )
        today = date.today()
        start = today - timedelta(days=days)
        try:
            df = ak.stock_zh_a_hist(
                symbol=sym,
                period="daily",
                start_date=_to_akshare_date(start),
                end_date=_to_akshare_date(today),
                adjust="qfq",
            )
        except Exception as exc:  # noqa: BLE001
            raise UpstreamUnavailableError(
                f"akshare.stock_zh_a_hist failed for {sym!r}: {exc}"
            ) from exc
        if df is None or df.empty:
            raise SymbolNotFoundError(f"A-share symbol not found: {sym!r}")

        bars: list[OHLCV] = []
        for _, row in df.tail(limit).iterrows():
            bars.append(
                OHLCV(
                    date=pd.to_datetime(row["日期"]).date(),
                    open=float(row["开盘"]),
                    high=float(row["最高"]),
                    low=float(row["最低"]),
                    close=float(row["收盘"]),
                    volume=float(row["成交量"]),
                )
            )
        return bars

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        sym = normalize_symbol(symbol, Market.A_SHARE)
        try:
            df = ak.stock_individual_info_em(symbol=sym)
        except Exception as exc:  # noqa: BLE001
            raise UpstreamUnavailableError(
                f"akshare.stock_individual_info_em failed for {sym!r}: {exc}"
            ) from exc
        if df is None or df.empty:
            raise SymbolNotFoundError(f"A-share symbol not found: {sym!r}")

        item_value = dict(zip(df["item"].astype(str), df["value"].astype(str)))

        def _f(key: str) -> float | None:
            raw = item_value.get(key)
            if raw is None or raw in ("", "None", "nan"):
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        return Fundamentals(
            symbol=sym,
            market=Market.A_SHARE,
            currency="CNY",
            market_cap=_f("总市值"),
            pe_ratio=_f("市盈率(动)"),
            pb_ratio=_f("市净率"),
            dividend_yield=None,  # not directly provided by individual_info_em
            revenue=None,
            net_income=None,
            fiscal_period=None,
            source=self._source_name,
        )

    def get_company_info(self, symbol: str) -> CompanyInfo:
        sym = normalize_symbol(symbol, Market.A_SHARE)
        try:
            df = ak.stock_individual_info_em(symbol=sym)
        except Exception as exc:  # noqa: BLE001
            raise UpstreamUnavailableError(
                f"akshare.stock_individual_info_em failed for {sym!r}: {exc}"
            ) from exc
        if df is None or df.empty:
            raise SymbolNotFoundError(f"A-share symbol not found: {sym!r}")

        item_value = dict(zip(df["item"].astype(str), df["value"].astype(str)))
        return CompanyInfo(
            symbol=sym,
            market=Market.A_SHARE,
            name=self._name_for(sym) or sym,
            name_local=self._name_for(sym),
            industry=item_value.get("行业"),
            sector=item_value.get("行业"),
            exchange=item_value.get("市场"),
            description=item_value.get("简介"),
            source=self._source_name,
        )

    def search_company(self, query: str, limit: int = 5) -> list[SearchResult]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            df = self._load_code_name_table()
        except Exception as exc:  # noqa: BLE001
            raise UpstreamUnavailableError(
                f"akshare.stock_info_a_code_name failed: {exc}"
            ) from exc

        # The frame has columns "code" and "name"; do a case-insensitive substring match.
        mask = (
            df["code"].astype(str).str.contains(q, case=False, na=False)
            | df["name"].astype(str).str.contains(q, case=False, na=False)
        )
        hits = df[mask].head(limit)
        results: list[SearchResult] = []
        for _, row in hits.iterrows():
            code = str(row["code"])
            name = str(row["name"])
            results.append(
                SearchResult(
                    symbol=normalize_symbol(code, Market.A_SHARE),
                    market=Market.A_SHARE,
                    name=name,
                    name_local=name,
                    exchange=None,
                    match_type="substring",
                )
            )
        return results

    # --- helpers --------------------------------------------------------

    def _name_for(self, symbol: str) -> str | None:
        try:
            df = self._load_code_name_table()
        except Exception:  # noqa: BLE001
            return None
        hit = df[df["code"].astype(str) == symbol]
        if hit.empty:
            return None
        return str(hit.iloc[0]["name"])


__all__ = ["AStockSource"]
