"""Tavily web-search adapter.

Thin wrapper around ``tavily.TavilyClient`` so the rest of the package
talks to the search backend through one small, testable object instead of
importing the SDK directly. The API key is a hardcoded placeholder in
:data:`TAVILY_API_KEY`; replace it with the real key before use.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    KeylessUnsupportedEndpointError,
    MissingAPIKeyError,
    TimeoutError as TavilyTimeoutError,
    UsageLimitExceededError,
)

logger = logging.getLogger(__name__)

# Hardcoded placeholder key — replace this string with the real Tavily API
# key. Deliberately not read from the environment, per project request.
TAVILY_API_KEY: str = "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"

# API-level failures raised by the Tavily SDK. ``httpx.HTTPError`` is caught
# separately below for transport-level failures (DNS, reset, TLS, timeouts).
_TAVILY_ERRORS: tuple[type[Exception], ...] = (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    KeylessUnsupportedEndpointError,
    MissingAPIKeyError,
    TavilyTimeoutError,
    UsageLimitExceededError,  # superclass of TavilyKeylessLimitError
)


class TavilySearchError(RuntimeError):
    """Raised when a Tavily search request fails.

    The original SDK / transport exception is preserved in ``__cause__``.
    """


class TavilyAdapter:
    """Adapter over the Tavily web-search SDK.

    Centralizes error normalization so the rest of the package depends on
    one small surface instead of the raw SDK.

    Attributes:
        client: The underlying synchronous :class:`TavilyClient`.
    """

    def __init__(self, api_key: str = TAVILY_API_KEY) -> None:
        self.client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        *,
        search_depth: str = "advanced",
        max_results: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a Tavily search for ``query``.

        Args:
            query: Search query. Must be non-empty.
            search_depth: One of ``"basic"``, ``"advanced"``, ``"fast"``,
                ``"ultra-fast"``. Defaults to ``"advanced"``.
            max_results: Optional cap on the number of returned results.
            **kwargs: Any other ``TavilyClient.search`` keyword argument
                (``topic``, ``time_range``, ``include_answer``,
                ``include_raw_content``, ``include_domains``, etc.).

        Returns:
            The raw Tavily response dict.

        Raises:
            TavilySearchError: If ``query`` is empty, or the SDK /
                transport raises during the request.
        """
        if not query.strip():
            raise TavilySearchError("query must be a non-empty string")
        payload: dict[str, Any] = {"search_depth": search_depth}
        if max_results is not None:
            payload["max_results"] = max_results
        payload.update(kwargs)
        try:
            return self.client.search(query=query, **payload)
        except _TAVILY_ERRORS as exc:
            logger.warning("tavily search failed (%s): %s", type(exc).__name__, exc)
            raise TavilySearchError(f"tavily search failed: {exc}") from exc
        except httpx.HTTPError as exc:
            logger.warning("tavily transport failed (%s): %s", type(exc).__name__, exc)
            raise TavilySearchError(f"tavily transport failed: {exc}") from exc
