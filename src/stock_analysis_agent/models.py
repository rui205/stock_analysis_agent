"""Pydantic models — the stable contract between the data layer and the agent.

Data sources return these typed objects; tools serialize them to compact
strings for the LLM. Using models (rather than raw `pd.DataFrame`) gives
us:

  * Normalized field names across markets (e.g., `pe_ratio: float | None`).
  * A single place to evolve the contract when we add fields.
  * Easy JSON serialization for caching and structured outputs.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Market(str, Enum):
    """The three supported markets.

    `str` mixin lets the LLM pass these as plain strings (e.g. "a_share")
    while the enum still constrains the legal values in our code.
    """

    A_SHARE = "a_share"
    US = "us"
    HK = "hk"

    @property
    def display_name(self) -> str:
        return {
            Market.A_SHARE: "A股 (China A-shares)",
            Market.US: "US (United States equities)",
            Market.HK: "HK (Hong Kong equities)",
        }[self]


class Quote(BaseModel):
    """A snapshot of the latest price for a symbol."""

    symbol: str
    market: Market
    company_name: str | None = None
    currency: str = Field(description="ISO 4217 currency code, e.g. 'CNY' / 'USD' / 'HKD'.")
    price: float
    change: float = Field(description="Absolute change vs. previous close.")
    change_pct: float = Field(description="Percentage change vs. previous close (0.01 = 1%).")
    as_of: datetime
    source: str = Field(description="Which data source produced this quote, e.g. 'akshare'.")


class OHLCV(BaseModel):
    """One bar of OHLCV data."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class Fundamentals(BaseModel):
    """A loose set of fundamentals. All fields optional — coverage varies by market/source."""

    symbol: str
    market: Market
    currency: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    dividend_yield: float | None = None
    revenue: float | None = None
    net_income: float | None = None
    fiscal_period: str | None = None
    source: str


class CompanyInfo(BaseModel):
    """Static / slow-moving company metadata."""

    symbol: str
    market: Market
    name: str
    name_local: str | None = Field(
        default=None, description="Native-script name (e.g. Chinese for A股)."
    )
    industry: str | None = None
    sector: str | None = None
    exchange: str | None = None
    description: str | None = None
    source: str


class SearchResult(BaseModel):
    """A row of a company search result."""

    symbol: str
    market: Market
    name: str
    name_local: str | None = None
    exchange: str | None = None
    match_type: Literal["exact", "prefix", "substring"] = "substring"


__all__ = [
    "CompanyInfo",
    "Fundamentals",
    "Market",
    "OHLCV",
    "Quote",
    "SearchResult",
]
