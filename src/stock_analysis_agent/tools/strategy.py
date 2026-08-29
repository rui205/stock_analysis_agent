"""Strategy-related tools: load a Markdown strategy file + subagent wrapper.

The ``load_strategy`` tool reads a single ``.md`` file under
``src/<package>/conf/strategies/`` and returns its full content (YAML
frontmatter + body). The LLM uses the body as natural-language
selection principles; the frontmatter provides ``name`` / ``version``
that flow into the output report.

The dynamic ``run_analyze_stock`` tool lives in this same module so
both strategy-related tools are colocated; it depends on
:class:`StockAnalysisAgent` and is exercised separately in Task 4.

The sub-agent returns Markdown (per the bundled ``stock-analysis`` skill),
so ``run_analyze_stock`` is a thin wrapper that just forwards the
agent's final text back to the caller — no JSON parsing, no schema
validation, no remapping. The caller (the strategy-match LLM) gets
the raw report and decides what to do with it.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from stock_analysis_agent.agent.stock_analysis import StockAnalysisAgent
from stock_analysis_agent.agent.stream import collect_final_text
from stock_analysis_agent.memory.file_cache import _FileCache
from stock_analysis_agent.tools._paths import PACKAGE_ROOT

# Resolved at import time — points at conf/strategies/. Tests may
# monkeypatch this to a tmp dir.
_STRATEGIES_DIR = PACKAGE_ROOT / "conf" / "strategies"

#: Frontmatter keys the strategy schema recognises. Anything else
#: (e.g. ``tags:`` in ``value-investing.md``) is dropped by
#: :func:`_parse_strategy_frontmatter` so the report sees only the
#: fields the schema binds to.
_STRATEGY_FRONTMATTER_KEYS: frozenset[str] = frozenset({"name", "version", "description"})

#: On-disk cache for sub-agent reports, keyed by (site, query). Re-running
#: the same symbol/dimensions within the TTL returns the previous report
#: instead of re-fetching data and re-paying LLM tokens. Cache misses and
#: write failures degrade silently — caching is an optimization, not a
#: correctness layer.
_SUBAGENT_CACHE_DIR = Path("~/.cache/stock-analysis-agent/subagent-reports").expanduser()
_SUBAGENT_CACHE_TTL: float = 3600.0  # 1 hour

_subagent_cache = _FileCache(_SUBAGENT_CACHE_DIR, ttl_seconds=_SUBAGENT_CACHE_TTL)


def _cache_query(
    symbol: str,
    *,
    dimensions: tuple[str, ...] | None,
    shell: bool,
) -> str:
    """Build the stable cache key for a sub-agent run.

    The shell flag is part of the key because it changes the sub-agent's
    output contract (shell-enabled runs execute the mx-* data scripts;
    without it the report degrades to LLM-knowledge-only).
    """
    if dimensions is None:
        return f"{shell}|{symbol}"
    return f"{shell}|{symbol}|{'、'.join(dimensions)}"


def _list_strategy_names() -> tuple[str, ...]:
    """Return the alphabetical list of ``.md`` strategy file stems.

    Files without the ``.md`` suffix are ignored. Missing directory
    yields an empty tuple (no error).
    """
    if not _STRATEGIES_DIR.is_dir():
        return ()
    return tuple(sorted(p.stem for p in _STRATEGIES_DIR.glob("*.md")))


def _parse_strategy_frontmatter(text: str) -> dict[str, str]:
    """Extract simple ``key: value`` pairs from a YAML frontmatter block.

    Supports single-line values and ``description: |`` literal blocks
    (joined with spaces). Delegates to the shared parser; only keys
    in :data:`_STRATEGY_FRONTMATTER_KEYS` are returned (others dropped).

    Args:
        text: Full strategy Markdown text starting with the ``---`` fence.

    Returns:
        Dict of frontmatter keys. Missing/unparseable input yields ``{}``.
    """
    from stock_analysis_agent.tools._frontmatter import parse_yaml_frontmatter

    return parse_yaml_frontmatter(text, allow=_STRATEGY_FRONTMATTER_KEYS)


class LoadStrategyInput(BaseModel):
    """Input schema for the ``load_strategy`` tool."""

    name: str = Field(
        min_length=1,
        description=(
            "Strategy name — must match a `.md` file under "
            "`src/stock_analysis_agent/conf/strategies/` (without the "
            "`.md` suffix). Example: `value-investing`. Unknown names "
            "raise `FileNotFoundError` with the available list."
        ),
    )


@tool(
    "load_strategy",
    description=(
        "Load a personal stock-selection strategy from "
        "`src/stock_analysis_agent/conf/strategies/<name>.md`. Returns "
        "the full file content (YAML frontmatter + Markdown body). "
        "Use the natural-language principles in the body to drive the "
        "per-criterion strategy matching in your final JSON report. "
        "Raises `FileNotFoundError` for unknown strategy names — the "
        "error message lists the available strategies."
    ),
    args_schema=LoadStrategyInput,
)
def load_strategy(name: str) -> str:
    """Read the full Markdown content of one strategy file.

    Args:
        name: Strategy file stem (no `.md` suffix).

    Returns:
        The full UTF-8 content of the file.

    Raises:
        FileNotFoundError: ``name`` is not a known strategy.
    """
    path = _STRATEGIES_DIR / f"{name}.md"
    if not path.is_file():
        available = ", ".join(_list_strategy_names()) or "(none)"
        raise FileNotFoundError(
            f"strategy {name!r} not found at {path}; available: {available}"
        )
    return path.read_text(encoding="utf-8")


def _run_subagent_and_collect(symbol: str) -> str:
    """Inner helper — builds the subagent, runs it, returns the final text.

    The sub-agent is driven by the same ``prompts/system_prompt.md`` as
    ``script.analyze_stock``; per the bundled ``stock-analysis`` skill,
    its final text is a Markdown report. We return it verbatim — no
    parsing, no remapping.
    """
    from stock_analysis_agent.script.analyze_stock import _load_system_prompt

    # The sub-agent always runs with ``run_command``: the bundled
    # stock-analysis workflow executes its mx-* skill scripts via shell
    # and publishes the report to Feishu (lark-cli), both of which
    # require shell access — without it the sub-agent degrades to an
    # LLM-knowledge-only report with no published URL to thread back.
    shell_enabled = True
    query = _cache_query(symbol, dimensions=None, shell=shell_enabled)
    cached = _subagent_cache.get(site="analyze_stock", query=query)
    if cached is not None:
        return cached

    system_prompt = _load_system_prompt(include_shell_tool=shell_enabled)
    sub = StockAnalysisAgent(
        system_prompt=system_prompt,
        include_shell_tool=shell_enabled,
        # Shell-enabled runs execute the full mx-* workflow: each data
        # fetch costs ~4 graph steps (run_command + read_file, each
        # preceded by an LLM decision round) plus skill loads and
        # reasoning — observed runs exhaust the constructor default of
        # 50 mid-workflow. Mirror ``script.analyze_stock``'s CLI default
        # (100) for the identical workflow.
        recursion_limit=100,
    )
    events = sub.stream([HumanMessage(f"按 system prompt 的 schema 给出 {symbol} 的分析报告。")])
    report = collect_final_text(events)
    try:
        _subagent_cache.set(site="analyze_stock", query=query, text=report)
    except OSError:
        pass  # cache write failure does not fail the sub-agent run
    return report


class RunAnalyzeStockInput(BaseModel):
    """Input schema for the ``run_analyze_stock`` tool."""

    symbol: str = Field(
        min_length=1,
        description=(
            "Stock symbol to analyze, e.g. `600519.SH`, `02319.HK`, "
            "`AAPL.US`. The tool runs the existing `StockAnalysisAgent` "
            "subagent on this symbol and returns its Markdown report verbatim."
        ),
    )


@tool(
    "run_analyze_stock",
    description=(
        "Run the existing `StockAnalysisAgent` subagent on a stock "
        "symbol and return its Markdown analysis verbatim. Returns the "
        "Markdown report on success; an `[ERROR] analyze_stock tool "
        "failed: ...` string when the sub-agent run fails (tool retries "
        "exhausted or recursion budget exceeded)."
    ),
    args_schema=RunAnalyzeStockInput,
)
def run_analyze_stock(symbol: str) -> str:
    """Synchronously run the analyze-stock subagent and forward its Markdown output.

    The sub-agent emits Markdown directly (per ``prompts/system_prompt.md`` +
    the bundled ``stock-analysis`` skill). No JSON parsing or remapping is
    performed here — the caller decides how to consume the report.

    Args:
        symbol: Stock symbol, e.g. ``"600519.SH"``.

    Returns:
        The sub-agent's final Markdown text on success, or an ``[ERROR]``-prefixed
        string when the sub-agent's tool retries are exhausted or its
        graph runs out of recursion budget mid-workflow.
    """
    try:
        return _run_subagent_and_collect(symbol)
    except Exception as e:  # noqa: BLE001 — degrade any sub-agent failure to [ERROR]
        # Covers ``ToolExecutionError`` (tool retries exhausted) and
        # ``RecursionError``/``GraphRecursionError`` (step budget), plus
        # any other unhandled failure (auth, file I/O, JSON, …). Degrade
        # to the soft ``[ERROR]`` contract rather than letting the
        # exception escape and abort the orchestrator run.
        return f"[ERROR] analyze_stock tool failed: {e}"


def _run_deepresearch_and_collect(symbol: str, dimensions: list[str]) -> str:
    """Run the DeepResearchAgent subagent and return its final Markdown text.

    The sub-agent is driven by ``DeepResearchAgent``'s bundled prompt (with
    ``symbol`` + ``dimensions`` injected). We return its final text verbatim —
    no parsing, no remapping. ``DeepResearchAgent`` is imported lazily to
    avoid the ``tools.__init__`` -> ``strategy`` -> ``deepresearch`` ->
    ``tools.web_search`` import cycle documented in ``tools/__init__.py``.
    """
    from stock_analysis_agent.agent.deepresearch import DeepResearchAgent

    shell_enabled = True
    dims = tuple(dimensions)
    query = _cache_query(symbol, dimensions=dims, shell=shell_enabled)
    cached = _subagent_cache.get(site="deepresearch", query=query)
    if cached is not None:
        return cached

    sub = DeepResearchAgent(
        symbol=symbol,
        dimensions=dimensions,
        # Inherit the orchestrator's shell opt-in: the mx-* skill data
        # scripts need ``run_command``; without it the sub-agent can only
        # fall back to ``web_search``.
        include_shell_tool=shell_enabled,
        # DeepResearch's default recursion_limit is None (LangGraph 25) —
        # too small for a multi-skill deep-research run.
        recursion_limit=100,
    )
    events = sub.stream(
        [HumanMessage(f"研究 {symbol} 的以下维度并产出带证据链的报告:{'、'.join(dimensions)}")]
    )
    report = collect_final_text(events)
    try:
        _subagent_cache.set(site="deepresearch", query=query, text=report)
    except OSError:
        pass  # cache write failure does not fail the sub-agent run
    return report


class RunDeepResearchInput(BaseModel):
    """Input schema for the ``run_deepresearch`` tool."""

    symbol: str = Field(
        min_length=1,
        description="Stock symbol to research, e.g. `600519.SH`, `02319.HK`, `AAPL.US`.",
    )
    dimensions: list[str] = Field(
        min_length=1,
        description=(
            "Research dimensions derived from the strategy principles that "
            "lack evidence, e.g. `['盈利质量-ROE', '财务稳健-现金流']`."
        ),
    )


@tool(
    "run_deepresearch",
    description=(
        "Run the `DeepResearchAgent` subagent on a stock symbol and one or "
        "more research dimensions, returning its Markdown report verbatim. "
        "Use when `run_analyze_stock`'s report lacks the data needed to "
        "judge one or more strategy principles. Returns Markdown on success; "
        "an `[ERROR] deepresearch tool failed: ...` string when the sub-agent "
        "run fails."
    ),
    args_schema=RunDeepResearchInput,
)
def run_deepresearch(symbol: str, dimensions: list[str]) -> str:
    """Synchronously run the deep-research subagent and forward its Markdown output.

    Args:
        symbol: Stock symbol, e.g. ``"600519.SH"``.
        dimensions: Research dimensions to pass to the sub-agent.

    Returns:
        The sub-agent's final Markdown text on success, or an ``[ERROR]``-prefixed
        string when the sub-agent's tool retries are exhausted or its graph runs
        out of recursion budget.
    """
    try:
        return _run_deepresearch_and_collect(symbol, dimensions)
    except Exception as e:  # noqa: BLE001 — degrade any sub-agent failure to [ERROR]
        return f"[ERROR] deepresearch tool failed: {e}"


__all__ = [
    "LoadStrategyInput",
    "RunAnalyzeStockInput",
    "RunDeepResearchInput",
    "_list_strategy_names",
    "_parse_strategy_frontmatter",
    "load_strategy",
    "run_analyze_stock",
    "run_deepresearch",
]
