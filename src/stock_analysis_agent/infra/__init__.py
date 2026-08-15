"""Infrastructure adapters for stock_analysis_agent.

Thin wrappers around external SDKs (search, data providers, etc.) so the
rest of the package depends on small, testable surfaces instead of raw
third-party clients.
"""
from __future__ import annotations

from stock_analysis_agent.infra.tavily_adapter import TavilyAdapter, TavilySearchError

__all__ = [
    "TavilyAdapter",
    "TavilySearchError",
]
