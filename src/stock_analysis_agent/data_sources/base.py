"""`DataSource` Protocol and symbol-normalization helpers.

`Protocol` gives us structural typing — any class that has the right
methods satisfies the contract, with no inheritance required. This is
ideal for wrapping third-party libraries (`akshare`, `yfinance`) that we
don't own and don't want to subclass.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..errors import SymbolNotFoundError
from ..models import CompanyInfo, Fundamentals, Market, OHLCV, Quote, SearchResult

# --- Symbol normalization --------------------------------------------------


def normalize_symbol(raw: str, market: Market) -> str:
    """Normalize a user-supplied symbol to the canonical form for the market.

    The canonical forms are:
      * A_SHARE: 6-digit string, e.g. ``"600519"`` (akshare convention).
      * US:      uppercase ticker, e.g. ``"AAPL"``.
      * HK:      4-digit zero-padded code with ``.HK`` suffix, e.g. ``"0700.HK"``.

    Args:
        raw: The symbol as the user (or LLM) wrote it.
        market: The target market.

    Returns:
        The canonical symbol string.

    Raises:
        SymbolNotFoundError: If the input is empty or cannot be parsed.
    """
    if raw is None:
        raise SymbolNotFoundError("Symbol must be a non-empty string.")
    s = raw.strip()
    if not s:
        raise SymbolNotFoundError("Symbol must be a non-empty string.")

    if market is Market.A_SHARE:
        # Strip common Chinese-style prefixes/suffixes.
        for prefix in ("sh", "sz", "bj"):
            if s.lower().startswith(prefix):
                s = s[len(prefix):]
                break
        if "." in s:
            s = s.split(".")[0]
        if not s.isdigit() or len(s) != 6:
            raise SymbolNotFoundError(
                f"A-share symbol must be 6 digits, got {raw!r}."
            )
        return s

    if market is Market.US:
        return s.upper()

    if market is Market.HK:
        # Strip a trailing ".HK" / ".hk" if present.
        s_low = s.lower()
        if s_low.endswith(".hk"):
            s = s_low[:-3]
        else:
            s = s_low
        # Allow either "700" or "0700" but enforce 4-digit zero-padding.
        if not s.isdigit():
            raise SymbolNotFoundError(
                f"HK symbol must be digits (optionally followed by '.HK'), got {raw!r}."
            )
        s = s.zfill(4)
        return f"{s}.HK"

    raise SymbolNotFoundError(f"Unknown market: {market!r}")


# --- Protocol --------------------------------------------------------------


@runtime_checkable
class DataSource(Protocol):
    """The contract every market-specific data source must satisfy.

    Tools look up the right source via `get_data_source(market)` and call
    these methods. Adding a new market = adding a new class with these
    methods; nothing else needs to change.
    """

    market: Market

    @property
    def market_name(self) -> str: ...

    def get_quote(self, symbol: str) -> Quote: ...

    def get_ohlcv(
        self, symbol: str, *, period: str = "1mo", limit: int = 30
    ) -> list[OHLCV]: ...

    def get_fundamentals(self, symbol: str) -> Fundamentals: ...

    def get_company_info(self, symbol: str) -> CompanyInfo: ...

    def search_company(self, query: str, limit: int = 5) -> list[SearchResult]: ...


__all__ = ["DataSource", "normalize_symbol"]
