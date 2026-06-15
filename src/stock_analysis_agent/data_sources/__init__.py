"""Data-source registry and public API.

The `get_data_source(market)` factory is the single point of dispatch
for the tools. To add a new market, implement the `DataSource` Protocol
in a new module, register the class here, and nothing else needs to
change.
"""

from __future__ import annotations

from functools import lru_cache

from ..models import Market
from .a_stock import AStockSource
from .base import DataSource, normalize_symbol
from .hk_stock import HKStockSource
from .us_stock import USStockSource

_REGISTRY: dict[Market, type[DataSource]] = {
    Market.A_SHARE: AStockSource,
    Market.US: USStockSource,
    Market.HK: HKStockSource,
}


@lru_cache(maxsize=None)
def get_data_source(market: Market) -> DataSource:
    """Return a process-wide singleton data source for the given market."""
    cls = _REGISTRY.get(market)
    if cls is None:
        raise ValueError(f"No data source registered for market: {market!r}")
    return cls()


__all__ = [
    "AStockSource",
    "DataSource",
    "HKStockSource",
    "USStockSource",
    "get_data_source",
    "normalize_symbol",
]
