"""Exception hierarchy for the data-source layer.

Tools catch these and return a clear string to the LLM so the agent can
decide whether to retry, switch market, or surface the error to the user.
Bare `Exception` should not leak out of the data-source layer.
"""

from __future__ import annotations


class DataSourceError(RuntimeError):
    """Base class for all data-source failures."""


class SymbolNotFoundError(DataSourceError):
    """The requested ticker / symbol is unknown to the data source."""


class RateLimitError(DataSourceError):
    """The data source throttled us. Caller should back off."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class UpstreamUnavailableError(DataSourceError):
    """The data source is reachable but returning errors or partial data."""


__all__ = [
    "DataSourceError",
    "SymbolNotFoundError",
    "RateLimitError",
    "UpstreamUnavailableError",
]
