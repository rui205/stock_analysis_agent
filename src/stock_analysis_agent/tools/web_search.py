"""@tool web_search: Tavily-backed web search with file caching."""
from __future__ import annotations

from typing import Any, Generic, TypeVar

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from stock_analysis_agent.agent.exceptions import ToolExecutionError
from stock_analysis_agent.infra.tavily_adapter import TavilyAdapter, TavilySearchError
from stock_analysis_agent.memory.file_cache import _FileCache


T = TypeVar("T")


class _Provider(Generic[T]):
    """Module-level singleton holder for a single value.

    ``DeepResearchAgent.__init__`` writes ``self.value`` (the file cache);
    the @tool ``_web_search`` reads it via ``.get()`` on every call.
    """

    def __init__(self) -> None:
        self.value: T | None = None  # type: ignore[assignment]

    def get(self) -> T:
        if self.value is None:
            raise RuntimeError(
                f"{type(self).__name__}.value was not initialized; "
                "was DeepResearchAgent.__init__ called?"
            )
        return self.value


_CACHE_PROVIDER: _Provider[_FileCache | None] = _Provider()

#: Fixed cache namespace — Tavily has no per-site fan-out, so every query
#: is stored under a single ``site`` key to reuse ``_FileCache`` unchanged.
_TAVILY_SITE_KEY: str = "tavily"


class WebSearchInput(BaseModel):
    """Input schema for the ``web_search`` tool."""

    query: str = Field(
        description="Search keyword / natural-language query sent to Tavily.",
        min_length=1,
    )


def _format_tavily_results(results: dict[str, Any]) -> str:
    """Render a Tavily search response into LLM-readable plain text.

    Args:
        results: Raw Tavily response dict (``answer`` + ``results`` list).

    Returns:
        A plain-text block: an optional ``[answer]`` summary followed by
        numbered ``[i] title / url / content`` entries; ``[no results]``
        when the result list is empty.
    """
    blocks: list[str] = []
    answer = results.get("answer")
    if answer:
        blocks.append(f"[answer]\n{answer}")
    for i, r in enumerate(results.get("results", []), start=1):
        blocks.append(
            f"[{i}] {r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}"
        )
    if not blocks:
        return "[no results]"
    return "\n\n".join(blocks)


@tool(
    "web_search",
    description=(
        "Search the web via the Tavily search API for `query` and return "
        "aggregated plain text (titles, URLs, and content snippets). "
        "Results are cached to disk via `_FileCache` under the `tavily` "
        "namespace. Raises `ToolExecutionError` when the Tavily request "
        "fails, so the retry middleware can act."
    ),
    args_schema=WebSearchInput,
)
def _web_search(query: str) -> str:  # pyright: ignore[reportUnusedFunction]
    """Search Tavily for `query` and return aggregated text.

    Returns:
        Plain-text concatenation of Tavily results (see
        :func:`_format_tavily_results`).

    Raises:
        ToolExecutionError: The Tavily search failed (wrapped from
            ``TavilySearchError``).
    """
    cache = _CACHE_PROVIDER.get()
    if cache is not None:
        hit = cache.get(site=_TAVILY_SITE_KEY, query=query)
        if hit is not None:
            return hit

    try:
        results = TavilyAdapter().search(query)
    except TavilySearchError as exc:
        raise ToolExecutionError(f"web_search failed: {exc}") from exc

    text = _format_tavily_results(results)
    if cache is not None:
        try:
            cache.set(site=_TAVILY_SITE_KEY, query=query, text=text)
        except OSError:
            pass  # cache write failure does not fail the search
    return text
