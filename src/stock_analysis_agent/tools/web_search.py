"""@tool web_search: fan-out concurrent HTTP search over a configured site list."""
from __future__ import annotations

import asyncio
from typing import Any, Generic, TypeVar

import httpx
from langchain.tools import tool

from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.memory.file_cache import _FileCache
from stock_analysis_agent.tools.text_extractor import _extract_text


async def _fetch_and_concat(
    query: str,
    site_list: list[str],
    *,
    cache: _FileCache | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = 10.0,
) -> str:
    """Fetch `query` from each site in `site_list` concurrently and concatenate results.

    Each site is fetched via httpx.AsyncClient with optional `transport`
    (for tests). Cache behavior:
      - If `cache` is None, every site is fetched over HTTP.
      - If `cache` is set, hit returns the cached text without HTTP;
        miss fetches and writes through to the cache.
    Per-site failures are recorded as `[error: ...]` segments rather
    than raised. If every site fails, the function raises
    `ToolExecutionError` so the BaseAgent retry middleware can act.
    """
    if not site_list:
        raise ValueError("site_list cannot be empty")

    async def _one(site: str) -> tuple[str, str]:
        # 1) Try cache first.
        if cache is not None:
            hit = cache.get(site=site, query=query)
            if hit is not None:
                return (site, hit)
        # 2) HTTP fetch.
        try:
            client_kwargs: dict[str, Any] = {"timeout": timeout}
            if transport is not None:
                client_kwargs["transport"] = transport
            async with httpx.AsyncClient(**client_kwargs) as client:
                resp = await client.get(site, params={"q": query})
                resp.raise_for_status()
                text = _extract_text(resp.text)
        except Exception as e:
            return (site, f"[error: {type(e).__name__}: {e}]")
        # 3) Write-through cache (best-effort).
        if cache is not None:
            try:
                cache.set(site=site, query=query, text=text)
            except OSError:
                pass  # cache write failure does not fail the search
        return (site, text)

    results = await asyncio.gather(*(_one(s) for s in site_list))
    if all(text.startswith("[error:") for _, text in results):
        raise ToolExecutionError(f"all sites failed: {[s for s, _ in results]}")

    parts = [f"[{site}]\n{text}\n" for site, text in results]
    return "\n".join(parts)


T = TypeVar("T")


class _Provider(Generic[T]):
    """Module-level singleton holder for a single value.

    The single-instance design (per spec §1) lets us mutate `self.value`
    on every `DeepSearchAgent.__init__` call, and the @tool _web_search
    reads it via `.get()` whenever the LLM invokes the tool. Concurrent
    multi-instance construction is not supported.
    """

    def __init__(self) -> None:
        self.value: T | None = None  # type: ignore[assignment]

    def get(self) -> T:
        if self.value is None:
            raise RuntimeError(
                f"{type(self).__name__}.value was not initialized; "
                "was DeepSearchAgent.__init__ called?"
            )
        return self.value


_SITE_LIST_PROVIDER: _Provider[list[str]] = _Provider()
_CACHE_PROVIDER: _Provider[_FileCache | None] = _Provider()


# The explicit name drops the leading underscore so the LLM sees the
# tool as "web_search", not "_web_search".
@tool("web_search")
async def _web_search(query: str) -> str:
    """Search the configured site list for `query` and return aggregated text.

    Returns a plain-text concatenation of snippets from each configured
    site. Sites that error are mentioned in the output but do not abort
    the search.
    """
    sites = _SITE_LIST_PROVIDER.get()
    cache = _CACHE_PROVIDER.get()
    return await _fetch_and_concat(query, sites, cache=cache)