"""DeepResearchAgent: an LLM-driven deep-research agent for stock research.

Bundles the ``load_skill`` / ``read_file`` tools plus the Tavily-backed
``web_search`` @tool, and a ``run_command`` tool (on by default) for
executing the mx-* data-skill scripts. The stock symbol and research
dimensions are
injected into the system prompt before the LLM is called — pass
``prompts/deepresearch_system_prompt.md`` for the full contract (think-first
workflow, evidence chain + confidence, ``unknown`` handling).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stock_analysis_agent.agent.base import BaseAgent
from stock_analysis_agent.memory.file_cache import _FileCache
from stock_analysis_agent.tools.read_file import read_file
from stock_analysis_agent.tools.shell import run_command
from stock_analysis_agent.tools.skill import load_skill
from stock_analysis_agent.tools.web_search import (
    _CACHE_PROVIDER,
    _web_search,
)


DEFAULT_SYSTEM_PROMPT: str = (
    "You are a deep research agent. Given a stock code and one or more "
    "research dimensions, first clarify the research direction and break "
    "each dimension into concrete questions, then use the available tools "
    "(load_skill / read_file / web_search, and run_command when enabled) "
    "to gather data, and produce a report in which every conclusion "
    "carries an evidence chain and a confidence level, and unsearchable "
    "questions are marked `unknown` — never fabricate. Pass the bundled "
    "`prompts/deepresearch_system_prompt.md` as system_prompt for the full "
    "contract."
)

DEFAULT_CACHE_DIR: str = "~/.cache/stock-analysis-agent"
DEFAULT_CACHE_TTL: float | None = 86400.0  # 24h in seconds

#: Absolute path to the bundled system prompt template.
_PROMPT_FILE: Path = (
    Path(__file__).resolve().parents[1] / "prompts" / "deepresearch_system_prompt.md"
)


def render_research_prompt(
    template: str,
    *,
    symbol: str | None,
    dimensions: Sequence[str] | None,
) -> str:
    """Inject the stock symbol and research dimensions into a prompt template.

    Args:
        template: Prompt template containing ``<!-- STOCK -->`` and
            ``<!-- DIMENSIONS -->`` placeholders.
        symbol: Stock code to substitute for ``<!-- STOCK -->``.
        dimensions: Research-dimension labels joined with ``、`` and
            substituted for ``<!-- DIMENSIONS -->``.

    Returns:
        The template with both placeholders replaced. Missing placeholders
        are a no-op (``str.replace`` leaves the text unchanged).
    """
    return (
        template.replace("<!-- STOCK -->", symbol or "")
        .replace("<!-- DIMENSIONS -->", "、".join(dimensions or []))
    )


class DeepResearchAgent(BaseAgent):
    """LLM-driven deep-research agent for stock research.

    Bundles three data-discovery tools — ``load_skill``, ``read_file``,
    and the Tavily-backed ``web_search`` @tool (caching results to local
    JSON files) — plus a ``run_command`` (on by default) for executing the
    mx-* data skill scripts. The stock symbol and research dimensions are
    injected into the system prompt before the LLM is called.

    ``load_skill`` and ``read_file`` are always on (mirroring
    ``StockAnalysisAgent``); ``run_command`` is on by default via
    ``include_shell_tool=True`` — opt out with ``include_shell_tool=False``
    because it is a privilege escalation. The
    full deep-research contract lives in
    ``prompts/deepresearch_system_prompt.md`` — loaded automatically when
    ``symbol``/``dimensions`` are given, or pass its contents as
    ``system_prompt`` explicitly.

    Construction overrides `BaseAgent`'s `max_retries` default from 2 → 3.
    Other BaseAgent parameters (model, temperature, name, ...) flow
    through via **kwargs.

    Single-instance: constructing a second agent updates the module-level
    _CACHE_PROVIDER used by the @tool _web_search.
    """

    def __init__(
        self,
        *,
        symbol: str | None = None,
        dimensions: Sequence[str] | None = None,
        system_prompt: str | None = None,
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
        cache_ttl: float | None = DEFAULT_CACHE_TTL,
        include_shell_tool: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the agent.

        Args:
            symbol: Stock code to inject into the system prompt
                ``<!-- STOCK -->`` placeholder. ``None`` renders it empty.
            dimensions: Research-dimension labels injected into the system
                prompt ``<!-- DIMENSIONS -->`` placeholder (joined with
                ``、``). ``None`` renders it empty.
            system_prompt: Caller-owned system prompt defining the deep
                research contract. Defaults to
                :data:`DEFAULT_SYSTEM_PROMPT`; pass the contents of
                ``prompts/deepresearch_system_prompt.md`` for the full
                contract.
            max_retries: Tool-call retry budget for transient errors.
            cache_dir: Directory for the ``web_search`` file cache.
            cache_ttl: Cache TTL in seconds; ``None`` disables expiration.
            include_shell_tool: When ``True`` (default), expose
                ``run_command`` so the agent can execute the mx-* skill
                scripts. Set ``False`` to omit it — the shell tool is a
                privilege escalation.
            **kwargs: Forwarded to :class:`BaseAgent` (``model``,
                ``temperature``, ``name``, ...).
        """
        resolved_prompt = self._resolve_prompt(
            system_prompt=system_prompt,
            symbol=symbol,
            dimensions=dimensions,
        )

        resolved_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else Path(DEFAULT_CACHE_DIR).expanduser().resolve()
        )
        # `cache_ttl` defaults to DEFAULT_CACHE_TTL when omitted; an explicit
        # `None` disables expiration; an explicit float sets a custom TTL.
        # No sentinel needed because the function default IS the resolution.

        self._cache = _FileCache(resolved_dir, ttl_seconds=cache_ttl)
        self._include_shell_tool = include_shell_tool

        # Single-instance: write the cache into the module-level provider
        # so the @tool can read it.
        _CACHE_PROVIDER.value = self._cache

        tools = [load_skill, read_file, _web_search]
        if include_shell_tool:
            tools.append(run_command)

        super().__init__(
            system_prompt=resolved_prompt,
            max_retries=max_retries,
            tools=tools,
            **kwargs,
        )

    @staticmethod
    def _resolve_prompt(
        *,
        system_prompt: str | None,
        symbol: str | None,
        dimensions: Sequence[str] | None,
    ) -> str:
        """Resolve and render the system prompt.

        Precedence: explicit ``system_prompt`` > bundled ``.md`` (when
        ``symbol``/``dimensions`` given) > :data:`DEFAULT_SYSTEM_PROMPT`.
        """
        if system_prompt is None and (symbol is not None or dimensions is not None):
            template = _PROMPT_FILE.read_text(encoding="utf-8")
        else:
            template = (
                system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
            )
        return render_research_prompt(template, symbol=symbol, dimensions=dimensions)

    @property
    def include_shell_tool(self) -> bool:
        """Whether the ``run_command`` tool is exposed to the LLM."""
        return self._include_shell_tool

    @property
    def cache_dir(self) -> Path:
        return self._cache._dir

    @property
    def cache_ttl(self) -> float | None:
        return self._cache._ttl